"""
Compute baseline metrics for all configured PatchTST tasks.

Baselines computed:
  - Regression: "Mean predictor" — always predicts the training-set mean for that target.
    This gives R²=0 by definition on the train set, but may differ slightly on test.
  - Classification: "Prevalence predictor" — always predicts the training-set event prevalence.
    Also computes a random baseline (random predictions at prevalence rate).

Usage:
    python scripts/compute_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from waveform_baselines.task_specs import DEFAULT_EVENT_TASK, DEFAULT_FEATURE_TASK


def main():
    base_dir = Path("outputs/patchtst")
    target_path = Path("outputs/targets/all_targets.npz")
    splits_path = Path("outputs/splits/splits.json")
    metadata = {}
    metadata_path = target_path.with_suffix(".json")
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    # Load target bundle and splits
    print("Loading target bundle...", flush=True)
    targets_data = np.load(target_path, allow_pickle=True)
    feature_targets = targets_data["feature_targets"]
    feature_mask = targets_data["feature_mask"]
    event_targets = targets_data["event_targets"]
    event_mask = targets_data["event_mask"]
    patient_ids = targets_data["anchor_patient_ids"]  # (N,)

    with open(splits_path) as f:
        splits = json.load(f)
    train_pids = set(splits["train"])

    # Build train-patient mask
    train_idx = np.array([p in train_pids for p in patient_ids])
    print(f"Train anchors: {train_idx.sum():,} / {len(patient_ids):,}", flush=True)

    # --- Feature column mapping ---
    # From train_patchtst.py:
    WAVEFORM_FEATURE_NAMES = [
        "HR", "RR", "SBP", "DBP", "PP",
        "MAP", "ABP_area", "PLETH_ACDC", "PLETH_amp", "ECG_Ramp",
        "HRV_RMSSD", "HR_range", "ShockIdx", "PPV", "PVI",
        "PTT", "dPdt_max", "ABP_tau", "RESP_amp",
    ]
    CORRELATION_FEATURE_NAMES = [
        "PLETH_ACDC_PLETH_amp", "ABP_area_ABP_tau", "ABP_area_ShockIdx",
        "PLETH_amp_ShockIdx", "PLETH_ACDC_ShockIdx", "ShockIdx_ABP_tau",
        "PLETH_ACDC_ABP_tau",
    ]
    ALL_FEATURE_NAMES = WAVEFORM_FEATURE_NAMES + CORRELATION_FEATURE_NAMES
    FEATURE_HORIZONS = list(DEFAULT_FEATURE_TASK.horizons_min)
    EVENT_NAMES = ["hypotension", "tachycardia"]
    EVENT_HORIZONS = list(DEFAULT_EVENT_TASK.horizons_min)
    FEATURE_TARGET_NAMES = metadata.get("feature_target_names")
    EVENT_TARGET_NAMES = metadata.get("event_target_names")

    def feature_col_index(name: str, horizon: int) -> int:
        if FEATURE_TARGET_NAMES:
            return FEATURE_TARGET_NAMES.index(f"{name}_t_plus_{horizon}m")
        h_idx = FEATURE_HORIZONS.index(horizon)
        f_idx = ALL_FEATURE_NAMES.index(name)
        return h_idx * len(ALL_FEATURE_NAMES) + f_idx

    def event_col_index(name: str, horizon: int) -> int:
        if EVENT_TARGET_NAMES:
            return EVENT_TARGET_NAMES.index(f"{name}_within_{horizon}m")
        h_idx = EVENT_HORIZONS.index(horizon)
        e_idx = EVENT_NAMES.index(name)
        return h_idx * len(EVENT_NAMES) + e_idx

    # --- Compute training-set statistics ---
    print("\nComputing training-set statistics...", flush=True)

    # Feature means and stds from training set
    train_feature_targets = feature_targets[train_idx]
    train_feature_mask = feature_mask[train_idx]

    # Event prevalences from training set
    train_event_targets = event_targets[train_idx]
    train_event_mask = event_mask[train_idx]

    # --- Evaluate baselines on test set ---
    print("\n" + "=" * 80, flush=True)
    print("BASELINE RESULTS", flush=True)
    print("=" * 80, flush=True)

    all_baselines = []

    # Process each model output directory
    for task_dir in sorted(base_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        pred_file = task_dir / "test_predictions.npz"
        metrics_file = task_dir / "test_metrics.json"
        if not pred_file.exists() or not metrics_file.exists():
            continue

        # Load model metrics for comparison
        with open(metrics_file) as f:
            model_metrics = json.load(f)

        task_type = model_metrics["task"]
        target_key = model_metrics["target_key"]

        # Load test predictions (we need targets and masks)
        pred_data = np.load(pred_file, allow_pickle=True)
        test_targets = pred_data["targets"].astype(np.float64)
        test_masks = pred_data["masks"]
        test_preds = pred_data["predictions"].astype(np.float64)

        # Valid test samples (same filtering as eval script)
        valid = test_masks.astype(bool) & np.isfinite(test_preds) & np.isfinite(test_targets)
        t_valid = test_targets[valid]

        if len(t_valid) == 0:
            continue

        baseline = {
            "target_key": target_key,
            "task": task_type,
            "n_valid": int(valid.sum()),
        }

        if task_type == "feature":
            # Parse feature name and horizon from target_key
            # target_key format: "HR_t_plus_0m" or "PLETH_ACDC_PLETH_amp_t_plus_0m"
            parts = target_key.rsplit("_t_plus_", 1)
            feat_name = parts[0]
            horizon = int(parts[1].replace("m", ""))

            col_idx = feature_col_index(feat_name, horizon)

            # Training mean for this feature
            train_col = train_feature_targets[:, col_idx]
            train_mask_col = train_feature_mask[:, col_idx].astype(bool)
            train_valid = train_col[train_mask_col & np.isfinite(train_col)]
            train_mean = float(np.mean(train_valid))
            train_std = float(np.std(train_valid))

            # --- Mean predictor baseline ---
            # Predict train mean for every test sample
            mean_preds = np.full_like(t_valid, train_mean)
            mse_mean = float(np.mean((mean_preds - t_valid) ** 2))
            rmse_mean = float(np.sqrt(mse_mean))
            mae_mean = float(np.mean(np.abs(mean_preds - t_valid)))
            ss_res_mean = np.sum((mean_preds - t_valid) ** 2)
            ss_tot = np.sum((t_valid - t_valid.mean()) ** 2)
            r2_mean = float(1 - ss_res_mean / ss_tot) if ss_tot > 0 else 0.0

            baseline.update({
                "train_mean": train_mean,
                "train_std": train_std,
                "baseline_mse": mse_mean,
                "baseline_rmse": rmse_mean,
                "baseline_mae": mae_mean,
                "baseline_r2": r2_mean,
                # Model metrics for comparison
                "model_r2": model_metrics.get("r2"),
                "model_rmse": model_metrics.get("rmse"),
                "model_mae": model_metrics.get("mae"),
                "model_corr": model_metrics.get("corr"),
            })

        else:
            # Classification baseline
            parts = target_key.split("_within_")
            event_name = parts[0]
            horizon = int(parts[1].replace("m", ""))

            col_idx = event_col_index(event_name, horizon)

            # Training prevalence
            train_col = train_event_targets[:, col_idx]
            train_mask_col = train_event_mask[:, col_idx].astype(bool)
            train_valid_events = train_col[train_mask_col]
            train_prevalence = float(np.mean(train_valid_events))
            test_prevalence = float(t_valid.mean())

            # --- Always-negative baseline ---
            acc_neg = float(1 - test_prevalence)

            # --- Always-positive baseline ---
            acc_pos = float(test_prevalence)

            # --- Prevalence predictor: predict probability = train_prevalence ---
            # AUROC for constant predictor is undefined (0.5 by convention)
            # AUPRC for random = prevalence
            from sklearn.metrics import roc_auc_score, average_precision_score

            # Random baseline with proper seed
            rng = np.random.default_rng(42)
            random_probs = rng.random(len(t_valid))
            try:
                auroc_random = float(roc_auc_score(t_valid, random_probs))
            except ValueError:
                auroc_random = 0.5
            auprc_random = float(average_precision_score(t_valid, random_probs))

            # Prevalence-calibrated random: predict train_prevalence for everyone
            # Threshold at 0.5 → if prevalence < 0.5, always predict 0
            prev_pred_label = 1 if train_prevalence >= 0.5 else 0
            if prev_pred_label == 0:
                f1_prev = 0.0
                sens_prev = 0.0
                spec_prev = 1.0
            else:
                sens_prev = 1.0
                spec_prev = 0.0
                precision_prev = test_prevalence
                f1_prev = 2 * precision_prev / (1 + precision_prev)

            baseline.update({
                "train_prevalence": train_prevalence,
                "test_prevalence": test_prevalence,
                "baseline_auroc_random": auroc_random,
                "baseline_auprc_random": auprc_random,
                "baseline_auroc_constant": 0.5,
                "baseline_auprc_constant": test_prevalence,
                "baseline_acc_always_neg": acc_neg,
                "baseline_f1_always_neg": 0.0,
                "baseline_f1_prev": f1_prev,
                # Model metrics
                "model_auroc": model_metrics.get("auroc"),
                "model_auprc": model_metrics.get("auprc"),
                "model_f1": model_metrics.get("f1"),
                "model_sensitivity": model_metrics.get("sensitivity"),
                "model_specificity": model_metrics.get("specificity"),
            })

        all_baselines.append(baseline)

    # --- Print summary tables ---
    print("\n" + "=" * 90, flush=True)
    print("FEATURE REGRESSION — PatchTST vs Mean Predictor Baseline", flush=True)
    print("=" * 90, flush=True)

    feature_baselines = [b for b in all_baselines if b["task"] == "feature"]
    feature_baselines.sort(key=lambda x: -(x.get("model_r2") or -999))

    print(f"\n{'Target':<28} {'Model R²':>9} {'Base R²':>8} {'Model RMSE':>11} {'Base RMSE':>10} {'Model MAE':>10} {'Base MAE':>9}")
    print("-" * 95)
    for b in feature_baselines:
        print(f"{b['target_key']:<28} "
              f"{b.get('model_r2', 0):>9.4f} "
              f"{b.get('baseline_r2', 0):>8.4f} "
              f"{b.get('model_rmse', 0):>11.4f} "
              f"{b.get('baseline_rmse', 0):>10.4f} "
              f"{b.get('model_mae', 0):>10.4f} "
              f"{b.get('baseline_mae', 0):>9.4f}")

    print("\n" + "=" * 90, flush=True)
    print("EVENT CLASSIFICATION — PatchTST vs Random/Prevalence Baseline", flush=True)
    print("=" * 90, flush=True)

    event_baselines = [b for b in all_baselines if b["task"] == "event"]
    print(f"\n{'Target':<28} {'Model':>8} {'Random':>8} {'Const':>8} | {'Model':>8} {'Random':>8} {'Prev':>8} | {'Model':>7} {'Base':>7}")
    print(f"{'':28} {'AUROC':>8} {'AUROC':>8} {'AUROC':>8} | {'AUPRC':>8} {'AUPRC':>8} {'AUPRC':>8} | {'F1':>7} {'F1':>7}")
    print("-" * 105)
    for b in event_baselines:
        print(f"{b['target_key']:<28} "
              f"{b.get('model_auroc', 0):>8.4f} "
              f"{b.get('baseline_auroc_random', 0):>8.4f} "
              f"{b.get('baseline_auroc_constant', 0):>8.4f} | "
              f"{b.get('model_auprc', 0):>8.4f} "
              f"{b.get('baseline_auprc_random', 0):>8.4f} "
              f"{b.get('baseline_auprc_constant', 0):>8.4f} | "
              f"{b.get('model_f1', 0):>7.4f} "
              f"{b.get('baseline_f1_always_neg', 0):>7.4f}")

    # Save results
    output_path = base_dir / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(all_baselines, f, indent=2)
    print(f"\nSaved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
