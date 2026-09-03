"""
PatchTST evaluation script — runs inference on the test set using saved checkpoints.

Loads best_model.pt for each trained task and saves predictions + targets + masks
as .npz files for downstream analysis.

Usage:
    # Evaluate a single model:
    python scripts/eval_patchtst.py --task feature --feature-name MAP --horizon 0

    # Evaluate all completed models:
    python scripts/eval_patchtst.py --all

    # Evaluate with a specific checkpoint:
    python scripts/eval_patchtst.py --task feature --feature-name MAP --horizon 0 \
        --feature-horizon-mode gap \
        --target-path outputs/targets/feature_targets_gap_vasopressor_free.npz \
        --checkpoint outputs/patchtst/vasopressor_free_v1_es/feature_MAP_t_plus_0m_gap/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from waveform_baselines.patient_sampler import PatientGroupedSampler

# Import shared definitions from train script
from train_patchtst import (
    ALL_FEATURE_NAMES,
    EVENT_HORIZONS,
    EVENT_NAMES,
    FEATURE_HORIZONS,
    PatchTST,
    TargetNormalizationStats,
    load_target_normalization_stats,
    parse_channel_list,
    SingleTargetDataset,
    SUPPORTED_EVENT_NAMES,
    TargetExtractor,
    TrainConfig,
    _cli_flag_set,
    build_waveform_dataset,
    collate_fn,
)


class LegacyPatchEmbeddingV1(nn.Module):
    """Pre-refactor PatchTST v1 tokenizer with per-channel embeddings."""

    def __init__(self, patch_len: int, stride: int, d_model: int, n_channels: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)
        self.channel_embed = nn.Embedding(n_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> (B, C*T, D)."""
        bsz, n_channels, _ = x.shape
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        tokens = self.proj(patches)
        channel_ids = torch.arange(n_channels, device=x.device)
        tokens = tokens + self.channel_embed(channel_ids)[None, :, None, :]
        return tokens.reshape(bsz, -1, tokens.size(-1))


class LegacyPatchTSTV1(nn.Module):
    """Compatibility model for older flattened-token PatchTST v1 checkpoints."""

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.n_patches = (config.seq_len - config.patch_len) // config.stride + 1
        self.patch_embed = LegacyPatchEmbeddingV1(
            patch_len=config.patch_len,
            stride=config.stride,
            d_model=config.d_model,
            n_channels=config.n_channels,
        )
        total_tokens = self.n_patches * config.n_channels
        self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, config.d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        tokens = self.encoder(tokens + self.pos_embed)
        tokens = self.norm(tokens)
        latent = tokens.mean(dim=1)
        return self.head(latent)


def build_model_for_checkpoint(config: TrainConfig, state_dict: dict[str, torch.Tensor]) -> nn.Module:
    """Instantiate the model variant that matches the saved checkpoint layout."""
    inferred_variant = infer_checkpoint_model_variant(state_dict)
    if inferred_variant == "legacy_patchtst_v1":
        if config.model_variant != "patchtst_v1":
            raise ValueError(
                f"Checkpoint layout is legacy patchtst_v1, but evaluation config requested "
                f"{config.model_variant}."
            )
        return LegacyPatchTSTV1(config)
    if inferred_variant != config.model_variant:
        raise ValueError(
            f"Checkpoint layout indicates model_variant={inferred_variant}, but evaluation config "
            f"requested {config.model_variant}."
        )
    return PatchTST(config)


def infer_checkpoint_model_variant(state_dict: dict[str, torch.Tensor]) -> str:
    keys = set(state_dict)
    if any(key.startswith("v15_") for key in keys):
        return "patchtst_v1_5"
    if (
        "patch_embed.channel_embed.weight" in keys
        and "encoder_norm.weight" not in keys
        and "norm.weight" in keys
    ):
        return "legacy_patchtst_v1"
    if (
        "channel_embed.weight" in keys
        or any(key.startswith("fusion_layers.") for key in keys)
        or any(key.startswith("final_norm.") for key in keys)
    ):
        return "patchtst_v2"
    return "patchtst_v1"


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove torch.compile's _orig_mod. prefix and reject ambiguous collisions."""
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "", 1) if key.startswith("_orig_mod.") else key
        if new_key in cleaned:
            raise ValueError(f"State dict key collision after stripping _orig_mod.: {new_key}")
        cleaned[new_key] = value
    return cleaned


EVAL_OVERRIDE_FIELDS = {
    "--task": "task",
    "--feature-name": "feature_name",
    "--event-name": "event_name",
    "--horizon": "horizon",
    "--feature-horizon-mode": "feature_horizon_mode",
    "--seq-len": "seq_len",
    "--input-window-position": "input_window_position",
    "--model-variant": "model_variant",
    "--dataset-format": "dataset_format",
    "--waveform-dir": "waveform_dir",
    "--splits-path": "splits_path",
    "--target-path": "target_path",
    "--run-tag": "run_tag",
}

COMMON_EVAL_REQUIRED_FIELDS = {
    "task",
    "horizon",
    "channels",
    "n_channels",
    "seq_len",
    "model_variant",
    "patch_len",
    "stride",
    "d_model",
    "n_heads",
    "n_layers",
    "d_ff",
    "dropout",
    "normalize",
    "seed",
    "waveform_dir",
    "splits_path",
    "target_path",
}

VARIANT_REQUIRED_FIELDS = {
    "patchtst_v1": set(),
    "patchtst_v1_5": {
        "attn_dropout",
        "qkv_bias",
        "pool_depth",
        "pool_mlp_ratio",
        "pool_num_queries",
        "pool_complete_block",
        "pool_affine",
    },
    "patchtst_v2": {
        "cross_channel_layers",
        "cross_channel_heads",
        "cross_channel_window",
        "pooling_type",
    },
}


def extract_explicit_eval_overrides(args: argparse.Namespace, explicit_flags: set[str]) -> dict:
    overrides = {
        field_name: getattr(args, field_name)
        for flag_name, field_name in EVAL_OVERRIDE_FIELDS.items()
        if flag_name in explicit_flags
    }
    if "--channels" in explicit_flags:
        channels = parse_channel_list(args.channels)
        overrides["channels"] = channels
        if "--n-channels" in explicit_flags:
            overrides["n_channels"] = args.n_channels
        else:
            overrides["n_channels"] = len(channels)
    elif "--n-channels" in explicit_flags:
        overrides["n_channels"] = args.n_channels
    return overrides


def required_eval_fields(config_values: dict) -> set[str]:
    required = set(COMMON_EVAL_REQUIRED_FIELDS)
    task = config_values.get("task")
    if task == "feature":
        required.add("feature_name")
        required.add("feature_horizon_mode")
    elif task == "event":
        required.add("event_name")
    variant = config_values.get("model_variant")
    required.update(VARIANT_REQUIRED_FIELDS.get(variant, set()))
    return required


def validate_eval_config_values(config_values: dict) -> None:
    missing = [
        field_name
        for field_name in sorted(required_eval_fields(config_values))
        if field_name not in config_values or config_values[field_name] in (None, "")
    ]
    if missing:
        raise ValueError(
            "Saved training configuration is missing required evaluation fields: "
            + ", ".join(missing)
        )


def train_config_from_saved_and_overrides(saved_config: dict | None, overrides: dict) -> TrainConfig:
    config_values: dict = {}
    if saved_config:
        config_values.update({
            field_name: saved_config[field_name]
            for field_name in TrainConfig.__dataclass_fields__
            if field_name in saved_config
        })
    config_values.update(overrides)
    validate_eval_config_values(config_values)
    return TrainConfig(
        **{
            field_name: config_values[field_name]
            for field_name in TrainConfig.__dataclass_fields__
            if field_name in config_values
        }
    )


def load_saved_training_config(checkpoint_path: Path) -> dict:
    config_path = checkpoint_path.parent / "config.json"
    if config_path.exists():
        with open(config_path) as handle:
            return json.load(handle)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_config = ckpt.get("config")
    if isinstance(saved_config, dict):
        return saved_config
    raise FileNotFoundError(
        f"No saved training config found for checkpoint {checkpoint_path}. "
        "Expected either config.json next to the checkpoint or a checkpoint['config'] entry."
    )


def find_all_completed_models(base_dir: Path, skip_existing: bool = False) -> list[dict]:
    """Find all output directories that have a best_model.pt checkpoint."""
    models = []
    if not base_dir.exists():
        return models
    for task_dir in sorted(path.parent for path in base_dir.rglob("best_model.pt")):
        ckpt = task_dir / "best_model.pt"
        config_file = task_dir / "config.json"
        if skip_existing and (task_dir / "test_metrics.json").exists() and (task_dir / "test_predictions.npz").exists():
            print(f"Skipping existing evaluation: {task_dir}", flush=True)
            continue
        if ckpt.exists() and config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            models.append({
                "checkpoint": ckpt,
                "config": config,
                "output_dir": task_dir,
            })
    return models


def safe_roc_auc_score(targets: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(roc_auc_score(targets, probs))
        except ValueError:
            return float("nan")


def safe_average_precision_score(targets: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(average_precision_score(targets, probs))
        except ValueError:
            return float("nan")


def compute_event_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    pred_labels = (probs >= 0.5).astype(int)
    accuracy = float(np.mean(pred_labels == targets))
    prevalence = float(targets.mean())

    tp = float(((pred_labels == 1) & (targets == 1)).sum())
    fp = float(((pred_labels == 1) & (targets == 0)).sum())
    fn = float(((pred_labels == 0) & (targets == 1)).sum())
    tn = float(((pred_labels == 0) & (targets == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "prevalence": prevalence,
        "auroc": safe_roc_auc_score(targets, probs),
        "auprc": safe_average_precision_score(targets, probs),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }
    return metrics


def bootstrap_metric_interval(
    logits: np.ndarray,
    targets: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
) -> dict[str, dict[str, float | int]]:
    if n_bootstrap <= 0 or len(targets) == 0:
        return {}

    metric_names = [
        "accuracy",
        "prevalence",
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
    ]
    samples_by_metric: dict[str, list[float]] = {name: [] for name in metric_names}
    n = len(targets)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_targets = targets[idx]
        sample_logits = logits[idx]
        sample_metrics = compute_event_metrics(sample_logits, sample_targets)
        for name in metric_names:
            value = sample_metrics.get(name, float("nan"))
            if np.isfinite(value):
                samples_by_metric[name].append(float(value))

    ci = {}
    lower_q = 100 * (alpha / 2)
    upper_q = 100 * (1 - alpha / 2)
    for name, values in samples_by_metric.items():
        if not values:
            ci[f"{name}_ci"] = {
                "lower": float("nan"),
                "upper": float("nan"),
                "confidence_level": 1 - alpha,
                "n_bootstrap": n_bootstrap,
                "n_successful": 0,
            }
            continue

        arr = np.asarray(values, dtype=np.float64)
        ci[f"{name}_ci"] = {
            "lower": float(np.percentile(arr, lower_q)),
            "upper": float(np.percentile(arr, upper_q)),
            "confidence_level": 1 - alpha,
            "n_bootstrap": n_bootstrap,
            "n_successful": int(arr.size),
        }
    return ci


def resolve_target_normalization(
    config: TrainConfig,
    checkpoint_path: Path,
    checkpoint: dict,
) -> TargetNormalizationStats | None:
    if not config.normalize_targets:
        return None
    sidecar_path = checkpoint_path.parent / "target_normalization.json"
    if sidecar_path.exists():
        return load_target_normalization_stats(sidecar_path)
    stats = TargetNormalizationStats.from_dict(checkpoint.get("target_normalization"))
    if stats is None:
        raise FileNotFoundError(
            f"Target normalization is enabled for {checkpoint_path}, but no saved target stats were found."
        )
    return stats


def evaluate_single(
    config: TrainConfig,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 4,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
    ci_level: float = 0.95,
) -> dict:
    """
    Evaluate a single model on the test set.

    Returns dict with summary metrics and saves predictions to output_dir/test_predictions.npz.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluating: {config.task} / {config.target_key}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"{'='*60}", flush=True)

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = strip_compile_prefix(ckpt["model_state_dict"])
    model = build_model_for_checkpoint(config, state_dict).to(device)
    model.load_state_dict(state_dict, strict=True)

    best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("nan")))
    train_epochs = ckpt.get("epoch", "?")
    target_normalization = resolve_target_normalization(config, checkpoint_path, ckpt)
    print(f"  Loaded checkpoint: epoch={train_epochs}, best_val_loss={best_val_loss:.6f}", flush=True)
    if target_normalization is not None:
        print(
            "  Loaded target normalization: "
            f"mean={target_normalization.mean:.6f}, std={target_normalization.std:.6f}",
            flush=True,
        )

    # Load test dataset
    target_extractor = TargetExtractor(
        config,
        Path(config.target_path),
        target_normalization=target_normalization,
    )

    test_numpy_ds = build_waveform_dataset(config, split="test")
    test_ds = SingleTargetDataset(test_numpy_ds, target_extractor)
    print(f"  Test set: {len(test_ds):,} windows ({len(test_ds.patient_ids)} patients)", flush=True)

    test_sampler = PatientGroupedSampler(
        patient_boundaries=test_ds.patient_boundaries,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        seed=config.seed,
    )

    test_loader_kwargs = {
        "batch_sampler": test_sampler,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        test_loader_kwargs["prefetch_factor"] = 2
    test_loader = DataLoader(test_ds, **test_loader_kwargs)

    # Run inference
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    all_patient_ids = []
    all_anchor_times = []
    all_anchor_ids = []
    all_segment_ids = []
    all_segment_names = []

    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            waveform = batch["waveform"].to(device, non_blocking=True)
            target = batch["target"]
            mask = batch["mask"]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(waveform)

            all_preds.append(pred.squeeze(-1).float().cpu().numpy())
            all_targets.append(target.numpy())
            all_masks.append(mask.numpy())
            all_patient_ids.extend(batch["patient_id"])
            all_anchor_times.append(batch["anchor_time"].numpy())
            all_anchor_ids.append(batch["anchor_id"].numpy())
            all_segment_ids.extend(batch["segment_id"])
            all_segment_names.extend(batch["segment_name"])

    elapsed = time.time() - t0
    preds = np.concatenate(all_preds).astype(np.float64)
    targets = np.concatenate(all_targets).astype(np.float64)
    masks = np.concatenate(all_masks)
    patient_ids = np.asarray(all_patient_ids).astype(str)
    anchor_times = np.concatenate(all_anchor_times).astype(np.float64)
    anchor_ids = np.concatenate(all_anchor_ids).astype(np.int64)
    segment_ids = np.asarray(all_segment_ids).astype(str)
    segment_names = np.asarray(all_segment_names).astype(str)
    preds_original = preds
    targets_original = targets
    if target_normalization is not None:
        preds_original = target_normalization.denormalize_array(preds)
        targets_original = target_normalization.denormalize_array(targets)

    print(f"  Inference: {len(preds):,} samples in {elapsed:.1f}s "
          f"({len(preds)/elapsed:.0f} samples/s)", flush=True)

    # Compute metrics on valid samples (mask=True AND finite predictions)
    valid = masks.astype(bool) & np.isfinite(preds_original) & np.isfinite(targets_original)
    n_valid = valid.sum()
    n_nan_preds = int(np.isnan(preds).sum())
    metrics = {"n_total": len(preds), "n_valid": int(n_valid), "n_nan_preds": n_nan_preds}

    if n_nan_preds > 0:
        print(f"  Note: {n_nan_preds} NaN predictions filtered out", flush=True)

    if n_valid > 0:
        p_valid = preds_original[valid]
        t_valid = targets_original[valid]

        if config.task == "feature":
            # Regression metrics
            mse = float(np.mean((p_valid - t_valid) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(p_valid - t_valid)))
            # R² score
            ss_res = np.sum((p_valid - t_valid) ** 2)
            ss_tot = np.sum((t_valid - t_valid.mean()) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            # Correlation
            if np.std(p_valid) > 0 and np.std(t_valid) > 0:
                corr = float(np.corrcoef(p_valid, t_valid)[0, 1])
            else:
                corr = 0.0
            # Target stats for context
            target_mean = float(t_valid.mean())
            target_std = float(t_valid.std())

            metrics.update({
                "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "corr": corr,
                "target_mean": target_mean, "target_std": target_std,
            })
            print(f"  Results: R²={r2:.4f}, corr={corr:.4f}, MSE={mse:.4f}, "
                  f"RMSE={rmse:.4f}, MAE={mae:.4f}", flush=True)

        else:
            # Classification metrics (binary)
            try:
                metrics.update(compute_event_metrics(p_valid, t_valid))
            except ImportError as e:
                print(f"  Warning: sklearn metrics failed: {e}", flush=True)
                metrics.update({
                    "accuracy": float("nan"),
                    "prevalence": float("nan"),
                    "auroc": float("nan"),
                    "auprc": float("nan"),
                    "sensitivity": float("nan"),
                    "specificity": float("nan"),
                    "precision": float("nan"),
                    "f1": float("nan"),
                })

            alpha = 1 - ci_level
            bootstrap_rng = np.random.default_rng(bootstrap_seed)
            metrics.update(
                bootstrap_metric_interval(
                    logits=p_valid,
                    targets=t_valid.astype(np.int64),
                    rng=bootstrap_rng,
                    n_bootstrap=bootstrap_samples,
                    alpha=alpha,
                )
            )
            metrics["bootstrap"] = {
                "n_samples": bootstrap_samples,
                "seed": bootstrap_seed,
                "confidence_level": ci_level,
            }
            print(f"  Results: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, F1={metrics['f1']:.4f}, "
                  f"Acc={metrics['accuracy']:.4f}, Sens={metrics['sensitivity']:.4f}, "
                  f"Spec={metrics['specificity']:.4f}, prevalence={metrics['prevalence']:.4f}", flush=True)
            auroc_ci = metrics.get("auroc_ci", {})
            auprc_ci = metrics.get("auprc_ci", {})
            f1_ci = metrics.get("f1_ci", {})
            print(
                f"  Bootstrap {ci_level:.0%} CIs: "
                f"AUROC=[{auroc_ci.get('lower', float('nan')):.4f}, {auroc_ci.get('upper', float('nan')):.4f}], "
                f"AUPRC=[{auprc_ci.get('lower', float('nan')):.4f}, {auprc_ci.get('upper', float('nan')):.4f}], "
                f"F1=[{f1_ci.get('lower', float('nan')):.4f}, {f1_ci.get('upper', float('nan')):.4f}]",
                flush=True,
            )

    # Save predictions
    pred_path = output_dir / "test_predictions.npz"
    np.savez(
        pred_path,
        predictions=preds_original,
        targets=targets_original,
        predictions_normalized=preds,
        targets_normalized=targets,
        masks=masks,
        sample_ids=np.asarray([
            f"anchor_id:{int(anchor_id)}" if int(anchor_id) >= 0 else f"{pid}|{float(anchor_time):.6f}"
            for pid, anchor_time, anchor_id in zip(patient_ids.tolist(), anchor_times.tolist(), anchor_ids.tolist())
        ]).astype(str),
        patient_time_sample_ids=np.asarray([
            f"{pid}|{float(anchor_time):.6f}"
            for pid, anchor_time in zip(patient_ids.tolist(), anchor_times.tolist())
        ]).astype(str),
        patient_ids=patient_ids,
        anchor_times=anchor_times,
        anchor_ids=anchor_ids,
        segment_ids=segment_ids,
        segment_names=segment_names,
        task=config.task,
        target_key=config.target_key,
        best_val_loss=best_val_loss,
        train_epochs=train_epochs,
        **metrics,
    )
    print(f"  Saved: {pred_path} ({preds.nbytes / 1e6:.1f} MB)", flush=True)

    # Save metrics as JSON too for easy reading
    metrics_path = output_dir / "test_metrics.json"
    metrics["target_key"] = config.target_key
    metrics["task"] = config.task
    if target_normalization is not None:
        metrics["target_normalization"] = target_normalization.to_dict()
    metrics["best_val_loss"] = float(best_val_loss)
    metrics["train_epochs"] = int(train_epochs) if isinstance(train_epochs, (int, float)) else train_epochs
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}", flush=True)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate PatchTST on test set")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all completed models in outputs/patchtst/")
    parser.add_argument("--skip-existing", action="store_true",
                        help="When using --all, skip checkpoints that already have test metrics and predictions")
    parser.add_argument("--task", type=str, default="feature", choices=["feature", "event"])
    parser.add_argument("--feature-name", type=str, default="MAP")
    parser.add_argument("--event-name", type=str, default="hypotension", choices=SUPPORTED_EVENT_NAMES)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--feature-horizon-mode", type=str, default="center", choices=["center", "gap"])
    parser.add_argument("--channels", type=str, default="ABP,II,PLETH",
                        help="Comma-separated waveform channel order to load")
    parser.add_argument("--dataset-format", type=str, default="numpy_patient",
                        choices=["numpy_patient", "full_data_segments"])
    parser.add_argument("--seq-len", type=int, default=150_000)
    parser.add_argument(
        "--input-window-position",
        type=str,
        default="center",
        choices=["center", "input_end"],
        help="For full_data_segments, choose the centered window or the trailing window ending at the original 20-minute input boundary.",
    )
    parser.add_argument("--n-channels", type=int, default=3)
    parser.add_argument("--model-variant", type=str, default="patchtst_v1",
                        choices=["patchtst_v1", "patchtst_v1_5", "patchtst_v2"])
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to checkpoint (default: auto-detect from output dir)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--waveform-dir", type=str,
                        default="/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms")
    parser.add_argument("--splits-path", type=str, default="outputs/splits/splits.json")
    parser.add_argument("--target-path", type=str, default="outputs/targets/all_targets.npz")
    parser.add_argument("--output-base", type=str, default="outputs/patchtst",
                        help="Base directory for model outputs")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--bootstrap-samples", type=int, default=1000,
                        help="Number of bootstrap resamples for event-task confidence intervals")
    parser.add_argument("--bootstrap-seed", type=int, default=42,
                        help="Random seed for event-task bootstrap confidence intervals")
    parser.add_argument("--ci-level", type=float, default=0.95,
                        help="Confidence level for event-task bootstrap intervals")
    args = parser.parse_args()
    explicit_flags = _cli_flag_set()
    explicit_overrides = extract_explicit_eval_overrides(args, explicit_flags)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    base_dir = Path(args.output_base)

    if args.all:
        # Evaluate all completed models
        models = find_all_completed_models(base_dir, skip_existing=args.skip_existing)
        if not models:
            print("No completed models found in", base_dir)
            sys.exit(1)

        print(f"Found {len(models)} completed models", flush=True)
        all_metrics = []

        for info in models:
            config = train_config_from_saved_and_overrides(info["config"], explicit_overrides)
            metrics = evaluate_single(
                config=config,
                checkpoint_path=info["checkpoint"],
                output_dir=info["output_dir"],
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                ci_level=args.ci_level,
            )
            all_metrics.append(metrics)

        # Print summary table
        print(f"\n{'='*80}", flush=True)
        print("SUMMARY", flush=True)
        print(f"{'='*80}", flush=True)

        # Separate by task type
        feature_metrics = [m for m in all_metrics if m.get("task") == "feature"]
        event_metrics = [m for m in all_metrics if m.get("task") == "event"]

        if feature_metrics:
            print(f"\n{'Target':<30} {'R²':>8} {'Corr':>8} {'RMSE':>10} {'MAE':>10} {'N_valid':>10}")
            print("-" * 86)
            for m in sorted(feature_metrics, key=lambda x: -x.get("r2", float("-inf"))):
                print(f"{m.get('target_key', '?'):<30} "
                      f"{m.get('r2', float('nan')):>8.4f} "
                      f"{m.get('corr', float('nan')):>8.4f} "
                      f"{m.get('rmse', float('nan')):>10.4f} "
                      f"{m.get('mae', float('nan')):>10.4f} "
                      f"{m.get('n_valid', 0):>10,}")

        if event_metrics:
            print(f"\n{'Target':<30} {'AUROC':>8} {'AUROC CI':>21} {'AUPRC':>8} {'AUPRC CI':>21} {'F1':>8}")
            print("-" * 104)
            for m in sorted(event_metrics, key=lambda x: -x.get("auroc", float("-inf"))):
                auroc_ci = m.get("auroc_ci", {})
                auprc_ci = m.get("auprc_ci", {})
                print(f"{m.get('target_key', '?'):<30} "
                      f"{m.get('auroc', float('nan')):>8.4f} "
                      f"[{auroc_ci.get('lower', float('nan')):>6.4f}, {auroc_ci.get('upper', float('nan')):>6.4f}] "
                      f"{m.get('auprc', float('nan')):>8.4f} "
                      f"[{auprc_ci.get('lower', float('nan')):>6.4f}, {auprc_ci.get('upper', float('nan')):>6.4f}] "
                      f"{m.get('f1', float('nan')):>8.4f}")

        # Save combined results
        summary_path = base_dir / "test_results_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved summary: {summary_path}", flush=True)

    else:
        # Evaluate single model
        cli_channels = parse_channel_list(args.channels)
        cli_config = TrainConfig(
            task=args.task,
            feature_name=args.feature_name,
            event_name=args.event_name,
            horizon=args.horizon,
            feature_horizon_mode=args.feature_horizon_mode,
            run_tag=args.run_tag,
            dataset_format=args.dataset_format,
            channels=cli_channels,
            model_variant=args.model_variant,
            n_channels=len(cli_channels),
            seq_len=args.seq_len,
            input_window_position=args.input_window_position,
            waveform_dir=args.waveform_dir,
            splits_path=args.splits_path,
            target_path=args.target_path,
        )

        output_dir = Path(cli_config.resolve_output_dir())
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = output_dir / "best_model.pt"

        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            print("Has training completed for this target?")
            sys.exit(1)

        saved_config = load_saved_training_config(checkpoint_path)
        config = train_config_from_saved_and_overrides(saved_config, explicit_overrides)
        output_dir = checkpoint_path.parent

        evaluate_single(
            config=config,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            ci_level=args.ci_level,
        )


if __name__ == "__main__":
    main()
