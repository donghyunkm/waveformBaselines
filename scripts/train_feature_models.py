#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waveform_baselines.wf_features import FeaturePreprocessor, HistorySummaryBuilder, load_feature_cache
from waveform_baselines.wf_features.models import GRUSequenceModel, SequenceModelConfig, TCNSequenceModel, TransformerSequenceModel
from waveform_baselines.target_builders import load_waveform_feature_table

try:
    from train_patchtst import TargetExtractor, TrainConfig
except ImportError:  # pragma: no cover - used when imported as scripts.train_feature_models
    from scripts.train_patchtst import TargetExtractor, TrainConfig


ANCHOR_TIME_DECIMALS = 6
DEFAULT_PERSISTENCE_SOURCE_DIR = "/gpfs/data/eh3828lab/derived_datasets/baselines/output_v2"


@dataclass
class RunConfig:
    cache_dir: str
    splits_path: str
    target_path: str
    task: str
    feature_name: str = "MAP"
    event_name: str = "hypotension"
    horizon: int = 0
    feature_horizon_mode: str = "gap"
    model_type: str = "transformer"
    output_dir: str = ""
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    gru_hidden_dim: int = 128
    gru_layers: int = 1
    tcn_hidden_dim: int = 128
    tcn_blocks: int = 3
    tcn_kernel_size: int = 3
    mlp_hidden_dim: int = 512
    mlp_layers: int = 2
    seed: int = 42
    persistence_source_dir: str = DEFAULT_PERSISTENCE_SOURCE_DIR

    @property
    def target_key(self) -> str:
        if self.task == "feature":
            suffix = f"t_plus_{self.horizon}m"
            if self.feature_horizon_mode == "gap":
                suffix = f"{suffix}_gap"
            return f"{self.feature_name}_{suffix}"
        return f"{self.event_name}_within_{self.horizon}m"


class FeatureTargetDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def fit_preprocessor(cache, output_dir: Path) -> FeaturePreprocessor:
    pre = FeaturePreprocessor(cache.feature_names)
    pre.fit(cache.values, cache.mask, cache.split_labels)
    pre.save(output_dir / "preprocessing.json")
    return pre


def split_indices(split_labels: np.ndarray) -> dict[str, np.ndarray]:
    return {split: np.flatnonzero(np.asarray(split_labels) == split) for split in ("train", "val", "test")}


def anchor_key(patient_id: str, anchor_time: float) -> tuple[str, float]:
    return str(patient_id), round(float(anchor_time), ANCHOR_TIME_DECIMALS)


def quantized_anchor_time(anchor_time: float) -> int:
    value = float(anchor_time)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite anchor_time during target alignment: {value}")
    return int(np.rint(value * (10**ANCHOR_TIME_DECIMALS)))


def parse_integral_ids(values, label: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got shape {arr.shape}")
    numeric = arr.astype(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} contains non-finite IDs")
    integral = numeric == np.floor(numeric)
    if not integral.all():
        examples = numeric[~integral][:10].tolist()
        raise ValueError(f"{label} contains non-integral IDs, examples={examples}")
    info = np.iinfo(np.int64)
    if np.any(numeric < info.min) or np.any(numeric > info.max):
        raise ValueError(f"{label} contains values outside int64 range")
    return numeric.astype(np.int64)


def _key_duplicates(keys: list[tuple[object, ...]]) -> list[tuple[object, ...]]:
    seen: set[tuple[object, ...]] = set()
    duplicates: list[tuple[object, ...]] = []
    for key in keys:
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    return duplicates


def _feature_alignment_keys(cache, extractor: TargetExtractor, target_path: str) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]], str]:
    cache_anchor_ids = getattr(cache, "anchor_ids", None)
    target_anchor_ids = getattr(extractor, "anchor_ids", None)
    if cache_anchor_ids is not None and target_anchor_ids is not None:
        cache_ids = parse_integral_ids(cache_anchor_ids, "feature cache anchor_ids")
        target_ids = parse_integral_ids(target_anchor_ids, "target bundle anchor_ids")
        feature_keys = [("anchor_id", int(anchor_id)) for anchor_id in cache_ids.tolist()]
        target_keys = [("anchor_id", int(anchor_id)) for anchor_id in target_ids.tolist()]
        missing_ids = set(feature_keys).difference(target_keys)
        extra_ids = set(target_keys).difference(feature_keys)
        if missing_ids or extra_ids:
            missing_examples = sorted(missing_ids)[:5]
            extra_examples = sorted(extra_ids)[:5]
            raise ValueError(
                f"Target anchor_id set mismatch for {target_path}: "
                f"missing={len(missing_ids)} examples={missing_examples}; "
                f"extra={len(extra_ids)} examples={extra_examples}"
            )
        return feature_keys, target_keys, "anchor_id"

    cache_segment_ids = getattr(cache, "segment_ids", None)
    target_segment_ids = getattr(extractor, "segment_ids", None)
    if cache_segment_ids is not None and target_segment_ids is not None:
        feature_keys = [
            ("patient_segment_time", str(pid), str(segment_id), quantized_anchor_time(ts))
            for pid, segment_id, ts in zip(cache.patient_ids.tolist(), np.asarray(cache_segment_ids).astype(str).tolist(), cache.anchor_times.tolist())
        ]
        target_keys = [
            ("patient_segment_time", str(pid), str(segment_id), quantized_anchor_time(ts))
            for pid, segment_id, ts in zip(extractor.anchor_patient_ids.tolist(), np.asarray(target_segment_ids).astype(str).tolist(), extractor.anchor_times.tolist())
        ]
        return feature_keys, target_keys, f"patient_id+segment_id+anchor_time rounded to {ANCHOR_TIME_DECIMALS} decimals"

    cache_segment_names = getattr(cache, "segment_names", None)
    target_segment_names = getattr(extractor, "segment_names", None)
    if cache_segment_names is not None and target_segment_names is not None:
        feature_keys = [
            ("patient_seg_name_time", str(pid), str(seg_name), quantized_anchor_time(ts))
            for pid, seg_name, ts in zip(cache.patient_ids.tolist(), np.asarray(cache_segment_names).astype(str).tolist(), cache.anchor_times.tolist())
        ]
        target_keys = [
            ("patient_seg_name_time", str(pid), str(seg_name), quantized_anchor_time(ts))
            for pid, seg_name, ts in zip(extractor.anchor_patient_ids.tolist(), np.asarray(target_segment_names).astype(str).tolist(), extractor.anchor_times.tolist())
        ]
        return feature_keys, target_keys, f"patient_id+seg_name+anchor_time rounded to {ANCHOR_TIME_DECIMALS} decimals"

    feature_keys = [
        ("legacy_patient_time", str(pid), quantized_anchor_time(ts))
        for pid, ts in zip(cache.patient_ids.tolist(), cache.anchor_times.tolist())
    ]
    target_keys = [
        ("legacy_patient_time", str(pid), quantized_anchor_time(ts))
        for pid, ts in zip(extractor.anchor_patient_ids.tolist(), extractor.anchor_times.tolist())
    ]
    return feature_keys, target_keys, f"legacy patient_id+anchor_time rounded to {ANCHOR_TIME_DECIMALS} decimals"


def aligned_targets(cache, config: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    target_cfg = TrainConfig(
        task=config.task,
        feature_name=config.feature_name,
        event_name=config.event_name,
        horizon=config.horizon,
        feature_horizon_mode=config.feature_horizon_mode,
        target_path=config.target_path,
    )
    extractor = TargetExtractor(target_cfg, Path(config.target_path))

    feature_keys, target_key_rows, key_label = _feature_alignment_keys(cache, extractor, config.target_path)

    feature_duplicates = _key_duplicates(feature_keys)
    target_duplicates = _key_duplicates(target_key_rows)
    if feature_duplicates:
        raise ValueError(f"Duplicate feature cache anchor keys found during target alignment by {key_label}: {feature_duplicates[:5]}")
    if target_duplicates:
        raise ValueError(f"Duplicate target anchor keys found in {config.target_path} during alignment by {key_label}: {target_duplicates[:5]}")
    target_index = {key: idx for idx, key in enumerate(target_key_rows)}
    target_keys = set(target_index)
    missing_keys = [key for key in feature_keys if key not in target_keys]
    if missing_keys:
        examples = ", ".join(str(key) for key in missing_keys[:5])
        raise ValueError(f"Missing target rows for {len(missing_keys)} feature cache anchors: {examples}")
    extra_keys = target_keys.difference(feature_keys)
    values = np.full(cache.values.shape[0], np.nan, dtype=np.float32)
    valid = np.zeros(cache.values.shape[0], dtype=bool)
    for idx, key in enumerate(feature_keys):
        row_idx = target_index[key]
        target_value = float(extractor.targets[row_idx])
        is_valid = bool(extractor.masks[row_idx]) and np.isfinite(target_value)
        if not is_valid:
            target_value = 0.0
        values[idx] = target_value
        valid[idx] = is_valid
    print(
        f"Aligned targets: samples={len(feature_keys)} valid={int(valid.sum())} "
        f"missing=0 extra_target_rows={len(extra_keys)} alignment_key={key_label}",
        flush=True,
    )
    return values, valid


def target_base_names(config: RunConfig) -> list[str]:
    metadata_path = Path(config.target_path).with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        feature_spec = metadata.get("feature_spec", {})
        feature_names = list(feature_spec.get("feature_names", []))
        correlation_names = list(feature_spec.get("correlation_names", []))
        if feature_names or correlation_names:
            return feature_names + correlation_names
    from waveform_baselines.task_specs import DEFAULT_FEATURE_TASK

    return list(DEFAULT_FEATURE_TASK.base_target_names)


def current_target_style_values(cache, config: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.task != "feature":
        raise ValueError("Persistence baseline is only defined for feature regression tasks.")
    base_names = target_base_names(config)
    if config.feature_name not in base_names:
        raise ValueError(f"Feature {config.feature_name!r} is not in target base names: {base_names}")
    feature_col = base_names.index(config.feature_name)

    patient_ids, anchor_times, value_matrix = load_waveform_feature_table(config.persistence_source_dir)
    source_keys = [anchor_key(pid, ts) for pid, ts in zip(patient_ids.tolist(), anchor_times.tolist())]
    cache_anchor_ids = np.asarray(cache.anchor_ids, dtype=np.int64)

    if len(set(source_keys)) == len(source_keys):
        source_index = {key: idx for idx, key in enumerate(source_keys)}
        cache_keys = [anchor_key(pid, ts) for pid, ts in zip(cache.patient_ids.tolist(), cache.anchor_times.tolist())]
        missing = [key for key in cache_keys if key not in source_index]
        if missing:
            examples = ", ".join(str(key) for key in missing[:5])
            raise ValueError(f"Missing persistence source rows for {len(missing)} feature cache anchors: {examples}")
        source_row_indices = np.asarray([source_index[key] for key in cache_keys], dtype=np.int64)
        alignment_key = f"patient_id+anchor_time rounded to {ANCHOR_TIME_DECIMALS} decimals"
    else:
        if np.any(cache_anchor_ids < 0) or np.any(cache_anchor_ids >= value_matrix.shape[0]):
            raise ValueError("Full-data persistence alignment by anchor_id found source row indices outside the source matrix")
        source_row_indices = cache_anchor_ids
        alignment_key = "anchor_id"

    values = np.full(cache.values.shape[0], np.nan, dtype=np.float32)
    valid = np.zeros(cache.values.shape[0], dtype=bool)
    for idx, source_row_idx in enumerate(source_row_indices.tolist()):
        value = float(value_matrix[source_row_idx, feature_col])
        values[idx] = value
        valid[idx] = np.isfinite(value)
    print(
        f"Aligned persistence source: samples={len(source_row_indices)} valid={int(valid.sum())} "
        f"missing=0 feature={config.feature_name} source_dir={config.persistence_source_dir} alignment_key={alignment_key}",
        flush=True,
    )
    return values, valid


def select_rows(x: np.ndarray, y: np.ndarray, valid: np.ndarray, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_valid = valid[row_ids] & np.isfinite(y[row_ids])
    return x[row_ids][row_valid], y[row_ids][row_valid]


def valid_row_ids(y: np.ndarray, valid: np.ndarray, row_ids: np.ndarray) -> np.ndarray:
    row_valid = valid[row_ids] & np.isfinite(y[row_ids])
    return row_ids[row_valid]


def valid_split_rows(y: np.ndarray, valid: np.ndarray, rows: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        valid_row_ids(y, valid, rows["train"]),
        valid_row_ids(y, valid, rows["val"]),
        valid_row_ids(y, valid, rows["test"]),
    )


def transform_rows(preprocessor: FeaturePreprocessor, cache, row_ids: np.ndarray) -> np.ndarray:
    return preprocessor.transform(cache.values[row_ids], cache.mask[row_ids])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    metrics = {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "n_evaluated": int(y_true.size),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
    }
    thresholds = np.unique(y_score)[::-1]
    best_threshold = 0.5
    best_spec = -1.0
    target_sensitivity = 0.85
    for threshold in thresholds:
        pred = (y_score >= threshold).astype(np.int64)
        tp = ((pred == 1) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if sens >= target_sensitivity and spec > best_spec:
            best_spec = float(spec)
            best_threshold = float(threshold)
    metrics["threshold_at_85_sensitivity"] = best_threshold
    metrics["specificity_at_85_sensitivity"] = best_spec
    return metrics


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def sequence_model_metadata(model: nn.Module, seq_len: int) -> dict[str, object]:
    metadata: dict[str, object] = {
        "trainable_parameters": count_trainable_parameters(model),
        "input_sequence_length": int(seq_len),
    }
    if isinstance(model, TCNSequenceModel):
        metadata.update({
            "tcn_dilations": list(model.dilations),
            "tcn_receptive_field": int(model.receptive_field),
            "tcn_receptive_field_covers_sequence": bool(model.receptive_field >= seq_len),
        })
    return metadata


def validate_run_config(config: RunConfig) -> None:
    if config.epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {config.epochs}")
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {config.batch_size}")
    if config.lr <= 0:
        raise ValueError(f"lr must be positive, got {config.lr}")
    if config.weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {config.weight_decay}")
    if not (0.0 <= config.dropout < 1.0):
        raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {config.dropout}")
    if config.mlp_hidden_dim <= 0:
        raise ValueError(f"mlp_hidden_dim must be positive, got {config.mlp_hidden_dim}")
    if config.mlp_layers < 1:
        raise ValueError(f"mlp_layers must be at least 1, got {config.mlp_layers}")


def validate_sequence_arrays(config: RunConfig, x_train: np.ndarray, x_val: np.ndarray) -> int:
    if x_train.ndim != 3 or x_val.ndim != 3:
        raise ValueError(f"Sequence models require 3D arrays (N, T, F), got train={x_train.shape} val={x_val.shape}")
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError(f"Train/val sequence length mismatch: {x_train.shape[1]} vs {x_val.shape[1]}")
    if x_train.shape[2] != x_val.shape[2]:
        raise ValueError(f"Train/val feature dimension mismatch: {x_train.shape[2]} vs {x_val.shape[2]}")
    if x_train.shape[1] != 20:
        raise ValueError(f"Expected extracted-feature sequence length 20, got {x_train.shape[1]}")
    return int(x_train.shape[1])


def save_test_predictions(
    output_dir: Path,
    config: RunConfig,
    cache,
    test_row_ids: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
    logits: np.ndarray | None = None,
) -> None:
    row_indices = np.asarray(test_row_ids, dtype=np.int64)
    patient_ids = np.asarray(cache.patient_ids[row_indices]).astype(str)
    anchor_times = np.asarray(cache.anchor_times[row_indices], dtype=np.float64)
    anchor_ids = np.asarray(cache.anchor_ids[row_indices], dtype=np.int64)
    patient_time_sample_ids = np.asarray([f"{pid}|{round(float(ts), ANCHOR_TIME_DECIMALS):.6f}" for pid, ts in zip(patient_ids, anchor_times)]).astype(str)
    if len(set(patient_time_sample_ids.tolist())) == patient_time_sample_ids.size:
        sample_ids = patient_time_sample_ids
    else:
        sample_ids = np.asarray([f"anchor_id:{int(anchor_id)}" for anchor_id in anchor_ids.tolist()]).astype(str)
    if len(set(sample_ids.tolist())) != sample_ids.size:
        raise ValueError(f"Duplicate test prediction sample IDs found for {config.model_type} {config.target_key}")
    payload = {
        "predictions": np.asarray(predictions, dtype=np.float32),
        "targets": np.asarray(y_test, dtype=np.float32),
        "masks": np.ones(np.asarray(y_test).shape[0], dtype=bool),
        "row_indices": row_indices,
        "sample_ids": sample_ids,
        "patient_time_sample_ids": patient_time_sample_ids,
        "anchor_ids": anchor_ids,
        "patient_ids": patient_ids,
        "anchor_times": anchor_times,
        "split_labels": np.asarray(cache.split_labels[row_indices]).astype(str),
        "task": np.asarray(config.task),
        "target_key": np.asarray(config.target_key),
        "model_type": np.asarray(config.model_type),
    }
    if logits is not None:
        payload["logits"] = np.asarray(logits, dtype=np.float32)
    np.savez_compressed(output_dir / "test_predictions.npz", **payload)


class FullSequenceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be at least 1, got {n_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_torch_tabular_model(config: RunConfig, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullSequenceMLP(
        input_dim=x_train.shape[1],
        hidden_dim=config.mlp_hidden_dim,
        n_layers=config.mlp_layers,
        dropout=config.dropout,
    ).to(device)
    train_generator = torch.Generator().manual_seed(config.seed)
    train_loader = torch.utils.data.DataLoader(FeatureTargetDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True, generator=train_generator)
    val_loader = torch.utils.data.DataLoader(FeatureTargetDataset(x_val, y_val), batch_size=config.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss() if config.task == "event" else nn.MSELoss()
    best_state = None
    best_val = float("inf")
    for _ in range(config.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb).squeeze(-1)
                losses.append(float(loss_fn(pred, yb).item()))
        val_loss = float(np.mean(losses)) if losses else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_torch_tabular_model(model: nn.Module, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(torch.from_numpy(x.astype(np.float32)), batch_size=512, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            preds.append(model(xb).squeeze(-1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_torch_sequence_model(config: RunConfig, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = validate_sequence_arrays(config, x_train, x_val)
    model_cfg = SequenceModelConfig(
        input_dim=x_train.shape[2],
        max_seq_len=seq_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        gru_hidden_dim=config.gru_hidden_dim,
        gru_layers=config.gru_layers,
        tcn_hidden_dim=config.tcn_hidden_dim,
        tcn_blocks=config.tcn_blocks,
        tcn_kernel_size=config.tcn_kernel_size,
    )
    if config.model_type == "transformer":
        model = TransformerSequenceModel(model_cfg)
    elif config.model_type == "tcn":
        model = TCNSequenceModel(model_cfg)
    else:
        model = GRUSequenceModel(model_cfg)
    model.to(device)
    train_generator = torch.Generator().manual_seed(config.seed)
    train_loader = torch.utils.data.DataLoader(FeatureTargetDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True, generator=train_generator)
    val_loader = torch.utils.data.DataLoader(FeatureTargetDataset(x_val, y_val), batch_size=config.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.task == "event":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    for _ in range(config.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb).squeeze(-1)
                losses.append(float(loss_fn(pred, yb).item()))
        val_loss = float(np.mean(losses)) if losses else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_torch_model(model: nn.Module, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(torch.from_numpy(x.astype(np.float32)), batch_size=512, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            preds.append(model(xb).squeeze(-1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_xgboost_like(config: RunConfig, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required for current_state and history_summary baselines.") from exc
    if config.task == "event":
        best_model = None
        best_val = -np.inf
        for params in ParameterGrid({"max_depth": [3, 5], "n_estimators": [100, 200], "learning_rate": [0.05]}):
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=config.seed,
                tree_method="hist",
                n_jobs=2,
                **params,
            )
            model.fit(x_train, y_train)
            val_score = average_precision_score(y_val, model.predict_proba(x_val)[:, 1])
            if val_score > best_val:
                best_val = float(val_score)
                best_model = model
        return best_model
    best_model = None
    best_val = float("inf")
    for params in ParameterGrid({"max_depth": [3, 5], "n_estimators": [100, 200], "learning_rate": [0.05]}):
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=config.seed,
            tree_method="hist",
            n_jobs=2,
            **params,
        )
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_model = model
    return best_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train feature-sequence baselines and temporal models.")
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--splits-path", type=str, required=True)
    parser.add_argument("--target-path", type=str, required=True)
    parser.add_argument("--task", choices=["feature", "event"], required=True)
    parser.add_argument("--feature-name", type=str, default="MAP")
    parser.add_argument("--event-name", type=str, default="hypotension")
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--feature-horizon-mode", choices=["center", "gap"], default="gap")
    parser.add_argument("--model-type", "--model", dest="model_type", choices=["transformer", "gru", "tcn", "history_xgb", "current_state_xgb", "full_sequence_xgb", "full_sequence_mlp", "current_state_linear", "persistence"], required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--tcn-hidden-dim", type=int, default=128)
    parser.add_argument("--tcn-blocks", type=int, default=3)
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--persistence-source-dir", type=str, default=DEFAULT_PERSISTENCE_SOURCE_DIR)
    args = parser.parse_args()
    config = RunConfig(**vars(args))
    validate_run_config(config)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = load_feature_cache(Path(config.cache_dir), require_success=True)
    y, valid = aligned_targets(cache, config)
    rows = split_indices(cache.split_labels)

    test_row_ids = np.array([], dtype=np.int64)
    test_predictions = None
    metrics: dict[str, float | int | bool | list[int]] = {}
    if config.model_type == "persistence":
        current_values, current_valid = current_target_style_values(cache, config)
        test_valid = valid & current_valid & np.isfinite(current_values) & np.isfinite(y)
        test_row_ids = valid_row_ids(y, test_valid, rows["test"])
        preds = current_values[test_row_ids]
        y_test = y[test_row_ids]
        metrics = regression_metrics(y_test, preds)
        test_predictions = preds
    else:
        preprocessor = fit_preprocessor(cache, output_dir)
        train_row_ids, val_row_ids, test_row_ids = valid_split_rows(y, valid, rows)
        y_train = y[train_row_ids]
        y_val = y[val_row_ids]
        y_test = y[test_row_ids]
        if config.model_type in {"transformer", "gru", "tcn", "full_sequence_mlp"}:
            x_train_seq = transform_rows(preprocessor, cache, train_row_ids)
            x_val_seq = transform_rows(preprocessor, cache, val_row_ids)
            x_test_seq = transform_rows(preprocessor, cache, test_row_ids)

    prediction_logits = None
    if config.model_type in {"transformer", "gru", "tcn"}:
        model = train_torch_sequence_model(config, x_train_seq, y_train, x_val_seq, y_val)
        metrics.update(sequence_model_metadata(model, x_train_seq.shape[1]))
        if config.task == "event":
            val_logits = predict_torch_model(model, x_val_seq)
            prediction_logits = predict_torch_model(model, x_test_seq)
            val_scores = 1.0 / (1.0 + np.exp(-val_logits))
            test_scores = 1.0 / (1.0 + np.exp(-prediction_logits))
            metrics = {**metrics, **classification_metrics(y_test.astype(int), test_scores)}
            metrics["validation_auprc"] = float(average_precision_score(y_val.astype(int), val_scores))
            test_predictions = test_scores
        else:
            preds = predict_torch_model(model, x_test_seq)
            metrics = {**metrics, **regression_metrics(y_test, preds)}
            test_predictions = preds
        torch.save(model.state_dict(), output_dir / "model.pt")
    elif config.model_type == "history_xgb":
        builder = HistorySummaryBuilder(cache.feature_names)
        x_train = builder.transform(cache.values[train_row_ids], cache.mask[train_row_ids])
        x_val = builder.transform(cache.values[val_row_ids], cache.mask[val_row_ids])
        x_test = builder.transform(cache.values[test_row_ids], cache.mask[test_row_ids])
        model = train_xgboost_like(config, x_train, y_train, x_val, y_val)
        with (output_dir / "model.pkl").open("wb") as f:
            pickle.dump(model, f)
        if config.task == "event":
            scores = model.predict_proba(x_test)[:, 1]
            metrics = classification_metrics(y_test.astype(int), scores)
            test_predictions = scores
        else:
            preds = model.predict(x_test)
            metrics = regression_metrics(y_test, preds)
            test_predictions = preds
    elif config.model_type == "current_state_xgb":
        x_train = transform_rows(preprocessor, cache, train_row_ids)[:, -1, :]
        x_val = transform_rows(preprocessor, cache, val_row_ids)[:, -1, :]
        x_test = transform_rows(preprocessor, cache, test_row_ids)[:, -1, :]
        model = train_xgboost_like(config, x_train, y_train, x_val, y_val)
        with (output_dir / "model.pkl").open("wb") as f:
            pickle.dump(model, f)
        if config.task == "event":
            test_predictions = model.predict_proba(x_test)[:, 1]
            metrics = classification_metrics(y_test.astype(int), test_predictions)
        else:
            test_predictions = model.predict(x_test)
            metrics = regression_metrics(y_test, test_predictions)
    elif config.model_type == "full_sequence_xgb":
        x_train_seq = transform_rows(preprocessor, cache, train_row_ids)
        x_val_seq = transform_rows(preprocessor, cache, val_row_ids)
        x_test_seq = transform_rows(preprocessor, cache, test_row_ids)
        x_train = x_train_seq.reshape(x_train_seq.shape[0], -1)
        x_val = x_val_seq.reshape(x_val_seq.shape[0], -1)
        x_test = x_test_seq.reshape(x_test_seq.shape[0], -1)
        model = train_xgboost_like(config, x_train, y_train, x_val, y_val)
        with (output_dir / "model.pkl").open("wb") as f:
            pickle.dump(model, f)
        if config.task == "event":
            test_predictions = model.predict_proba(x_test)[:, 1]
            metrics = classification_metrics(y_test.astype(int), test_predictions)
        else:
            test_predictions = model.predict(x_test)
            metrics = regression_metrics(y_test, test_predictions)
    elif config.model_type == "full_sequence_mlp":
        full_x = seq_x.reshape(seq_x.shape[0], -1)
        x_train, y_train = select_rows(full_x, y, valid, rows["train"])
        x_val, y_val = select_rows(full_x, y, valid, rows["val"])
        test_row_ids = valid_row_ids(y, valid, rows["test"])
        x_test = full_x[test_row_ids]
        y_test = y[test_row_ids]
        model = train_torch_tabular_model(config, x_train, y_train, x_val, y_val)
        metrics["trainable_parameters"] = count_trainable_parameters(model)
        logits_or_preds = predict_torch_tabular_model(model, x_test)
        if config.task == "event":
            prediction_logits = logits_or_preds
            test_predictions = 1.0 / (1.0 + np.exp(-logits_or_preds))
            metrics = {**metrics, **classification_metrics(y_test.astype(int), test_predictions)}
            val_logits = predict_torch_tabular_model(model, x_val)
            val_scores = 1.0 / (1.0 + np.exp(-val_logits))
            metrics["validation_auprc"] = float(average_precision_score(y_val.astype(int), val_scores))
        else:
            test_predictions = logits_or_preds
            metrics = {**metrics, **regression_metrics(y_test, test_predictions)}
        torch.save(model.state_dict(), output_dir / "model.pt")
    elif config.model_type == "current_state_linear":
        current_x = seq_x[:, -1, :]
        x_train, y_train = select_rows(current_x, y, valid, rows["train"])
        test_row_ids = valid_row_ids(y, valid, rows["test"])
        x_test = current_x[test_row_ids]
        y_test = y[test_row_ids]
        if config.task == "event":
            model = LogisticRegression(max_iter=1000).fit(x_train, y_train.astype(int))
            test_predictions = model.predict_proba(x_test)[:, 1]
            metrics = classification_metrics(y_test.astype(int), test_predictions)
        else:
            model = Ridge(alpha=1.0).fit(x_train, y_train)
            test_predictions = model.predict(x_test)
            metrics = regression_metrics(y_test, test_predictions)
        with (output_dir / "model.pkl").open("wb") as f:
            pickle.dump(model, f)
    if test_predictions is not None:
        save_test_predictions(output_dir, config, cache, test_row_ids, y_test, test_predictions, logits=prediction_logits)
        metrics["n_test_predictions"] = int(np.asarray(test_predictions).shape[0])
        metrics["n_test_patients"] = int(np.unique(np.asarray(cache.patient_ids[test_row_ids]).astype(str)).size)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
