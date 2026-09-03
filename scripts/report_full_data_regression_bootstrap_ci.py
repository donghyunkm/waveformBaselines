#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waveform_baselines.regression_bootstrap import bootstrap_regression_metrics

DEFAULT_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/full_data/regression")
DEFAULT_DOC = Path("docs/full_data/extractedFeaturesRegressionFullData.md")
DEFAULT_SUMMARY_PREFIX = Path("outputs/feature_models/full_data_regression_patient_bootstrap_ci_2026-09-01")
MODELS = ["history_xgb", "full_sequence_xgb", "transformer"]
TARGETS = [
    "HR", "RR", "SBP", "DBP", "PP", "MAP", "ABP_area", "PLETH_ACDC", "PLETH_amp",
    "ECG_Ramp", "HRV_RMSSD", "HR_range", "ShockIdx", "PPV", "PVI", "PTT",
    "dPdt_max", "ABP_tau", "RESP_amp", "PLETH_ACDC_PLETH_amp", "ABP_area_ABP_tau",
    "ABP_area_ShockIdx", "PLETH_amp_ShockIdx", "PLETH_ACDC_ShockIdx", "ShockIdx_ABP_tau",
    "PLETH_ACDC_ABP_tau",
]


def load_prediction_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        files = set(data.files)
        pred_key = "predictions" if "predictions" in files else "y_pred"
        target_key = "targets" if "targets" in files else "y_true"
        patient_key = "patient_ids" if "patient_ids" in files else "patient_id"
        missing = [key for key in (pred_key, target_key, patient_key) if key not in files]
        if missing:
            raise ValueError(f"{path} missing required arrays: {missing}")
        y_pred = np.asarray(data[pred_key], dtype=np.float64)
        y_true = np.asarray(data[target_key], dtype=np.float64)
        patient_ids = np.asarray(data[patient_key]).astype(str)
        if "masks" in files:
            mask = np.asarray(data["masks"], dtype=bool)
            y_pred = y_pred[mask]
            y_true = y_true[mask]
            patient_ids = patient_ids[mask]
    return y_true, y_pred, patient_ids


def run_dir(root: Path, model: str, target: str) -> Path:
    return root / f"{model}_feature_{target}_t_plus_0m_gap_v7"


def format_ci(value: float, lower: float, upper: float) -> str:
    return f"{value:.4f} [{lower:.4f}, {upper:.4f}]"


def load_existing_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summarize_run(root: Path, model: str, target: str, n_bootstrap: int, seed: int) -> dict[str, Any]:
    directory = run_dir(root, model, target)
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "test_predictions.npz"
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(f"missing metrics or predictions for {model} {target}: {directory}")
    if model in {"history_xgb", "full_sequence_xgb"} and not (directory / "model.pkl").exists():
        raise FileNotFoundError(f"missing XGBoost model.pkl for {model} {target}: {directory}")
    metrics = load_existing_metrics(metrics_path)
    y_true, y_pred, patient_ids = load_prediction_archive(predictions_path)
    ci = bootstrap_regression_metrics(y_true, y_pred, patient_ids, n_bootstrap=n_bootstrap, seed=seed).to_dict()
    for key in ("mae", "rmse", "r2"):
        if not np.isclose(float(metrics[key]), float(ci[key]), rtol=0.0, atol=5e-5):
            raise ValueError(f"point metric mismatch for {model} {target} {key}: metrics.json={metrics[key]} bootstrap={ci[key]}")
        ci[key] = float(metrics[key])
    if int(metrics.get("n_test_predictions", ci["n_test_predictions"])) != int(ci["n_test_predictions"]):
        raise ValueError(f"prediction count mismatch for {model} {target}")
    if int(metrics.get("n_test_patients", ci["n_test_patients"])) != int(ci["n_test_patients"]):
        raise ValueError(f"patient count mismatch for {model} {target}")
    enriched = {
        **metrics,
        **ci,
        "model": model,
        "target": f"{target}_t_plus_0m_gap",
        "target_base": target,
        "output_dir": str(directory),
        "predictions_path": str(predictions_path),
        "ci_method": "patient_cluster_percentile_bootstrap",
    }
    (directory / "metrics_with_ci.json").write_text(json.dumps(enriched, indent=2, sort_keys=True))
    return enriched


def markdown_model_table(model: str, rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: float(row["r2"]), reverse=True)
    lines = [
        f"### `{model}`",
        "",
        "Completed targets: `26/26`.",
        "",
        "| Target | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) | Test Predictions | Test Patients |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| `{row['target']}` | "
            f"{format_ci(row['mae'], row['mae_ci_lower'], row['mae_ci_upper'])} | "
            f"{format_ci(row['rmse'], row['rmse_ci_lower'], row['rmse_ci_upper'])} | "
            f"{format_ci(row['r2'], row['r2_ci_lower'], row['r2_ci_upper'])} | "
            f"{int(row['n_test_predictions'])} | {int(row['n_test_patients'])} |"
        )
    return "\n".join(lines)


def markdown_best_table(rows_by_model: dict[str, list[dict[str, Any]]]) -> tuple[str, dict[str, int]]:
    by_model_target = {model: {row["target_base"]: row for row in rows} for model, rows in rows_by_model.items()}
    best_rows = []
    counts = {model: 0 for model in MODELS}
    for target in TARGETS:
        best_model = max(MODELS, key=lambda model: float(by_model_target[model][target]["r2"]))
        counts[best_model] += 1
        best_rows.append((best_model, by_model_target[best_model][target]))
    lines = [
        "### Best Completed Model Per Target",
        "",
        "All three submitted model families completed. Best-model counts by held-out R2: "
        f"`history_xgb` {counts['history_xgb']}/26, `full_sequence_xgb` {counts['full_sequence_xgb']}/26, `transformer` {counts['transformer']}/26.",
        "",
        "| Target | Best Completed Model | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) | Test Predictions | Test Patients |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, row in best_rows:
        lines.append(
            f"| `{row['target']}` | `{model}` | "
            f"{format_ci(row['mae'], row['mae_ci_lower'], row['mae_ci_upper'])} | "
            f"{format_ci(row['rmse'], row['rmse_ci_lower'], row['rmse_ci_upper'])} | "
            f"{format_ci(row['r2'], row['r2_ci_lower'], row['r2_ci_upper'])} | "
            f"{int(row['n_test_predictions'])} | {int(row['n_test_patients'])} |"
        )
    return "\n".join(lines), counts


def markdown_result_section(
    rows_by_model: dict[str, list[dict[str, Any]]],
    n_bootstrap: int,
    seed: int,
    summary_prefix: Path,
) -> tuple[str, dict[str, int]]:
    best_table, counts = markdown_best_table(rows_by_model)
    invalid_total = sum(int(row["r2_invalid_bootstrap_replicates"]) for rows in rows_by_model.values() for row in rows)
    patient_counts = sorted({int(row["n_test_patients"]) for rows in rows_by_model.values() for row in rows})
    lines = [
        "## Result Tables",
        "",
        f"Full-data results refreshed on `2026-09-01` with `{n_bootstrap}` patient-cluster bootstrap replicates and seed `{seed}`. Tables include runs with completed `metrics.json`, `test_predictions.npz`, and XGBoost `model.pkl` artifacts where applicable. Point estimates are the original global test-set metrics; confidence intervals are percentile 95% CIs from resampling test patients with replacement and including all windows for sampled patients.",
        "",
        f"Unique test-patient counts across model-target runs: `{', '.join(str(v) for v in patient_counts)}`. Invalid R2 bootstrap replicates across all runs: `{invalid_total}`.",
        "",
        f"Machine-readable CI outputs are saved at `{summary_prefix.with_suffix('.json')}` and `{summary_prefix.with_suffix('.csv')}`; each run directory also has a `metrics_with_ci.json` sidecar with separate numeric CI fields.",
        "",
    ]
    lines.extend(markdown_model_table(model, rows_by_model[model]) for model in MODELS)
    lines.append(best_table)
    return "\n\n".join(lines) + "\n", counts


def write_summaries(prefix: Path, rows: list[dict[str, Any]], best_counts: dict[str, int]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {"runs": rows, "best_model_counts_by_r2": best_counts}
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    fieldnames = [
        "model", "target", "mae", "mae_ci_lower", "mae_ci_upper", "rmse", "rmse_ci_lower", "rmse_ci_upper",
        "r2", "r2_ci_lower", "r2_ci_upper", "n_test_predictions", "n_test_patients", "n_bootstrap", "seed",
        "r2_valid_bootstrap_replicates", "r2_invalid_bootstrap_replicates", "output_dir",
    ]
    with prefix.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def update_doc(doc_path: Path, result_section: str) -> None:
    text = doc_path.read_text()
    text = re.sub(r"## Result Tables\n.*?\n## Verification\n", result_section + "\n## Verification\n", text, flags=re.S)
    doc_path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add patient-cluster bootstrap CIs to full-data regression reports.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--summary-prefix", type=Path, default=DEFAULT_SUMMARY_PREFIX)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-doc-update", action="store_true")
    args = parser.parse_args()

    rows_by_model = {model: [] for model in MODELS}
    for model in MODELS:
        for target in TARGETS:
            rows_by_model[model].append(summarize_run(args.root, model, target, args.n_bootstrap, args.seed))
    section, best_counts = markdown_result_section(rows_by_model, args.n_bootstrap, args.seed, args.summary_prefix)
    all_rows = [row for model in MODELS for row in rows_by_model[model]]
    write_summaries(args.summary_prefix, all_rows, best_counts)
    args.summary_prefix.with_suffix(".md").write_text(section)
    if not args.no_doc_update:
        update_doc(args.doc, section)
    invalid = sum(int(row["r2_invalid_bootstrap_replicates"]) for row in all_rows)
    print(json.dumps({
        "runs": len(all_rows),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "best_counts": best_counts,
        "r2_invalid_bootstrap_replicates": invalid,
        "summary_json": str(args.summary_prefix.with_suffix(".json")),
        "summary_csv": str(args.summary_prefix.with_suffix(".csv")),
        "summary_md": str(args.summary_prefix.with_suffix(".md")),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
