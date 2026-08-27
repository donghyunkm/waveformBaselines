"""
Compute descriptive statistics for each regression target.

Writes a JSON summary with per-target stats across `train`, `val`, `test`, and
`all`, plus a flat CSV for easier inspection.

Usage:
    /gpfs/home/dk5565/.conda/envs/physiojepa/bin/python scripts/compute_target_stats.py \
        --target-path outputs/targets/all_targets.npz \
        --splits-path outputs/splits/splits.json \
        --output-json outputs/targets/feature_target_stats.json \
        --output-csv outputs/targets/feature_target_stats.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_feature_target_names(target_path: Path, n_cols: int) -> list[str]:
    metadata_path = target_path.with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        names = metadata.get("feature_target_names")
        if names and len(names) == n_cols:
            return [str(name) for name in names]
    return [f"feature_col_{idx}" for idx in range(n_cols)]


def patient_split_masks(patient_ids: np.ndarray, splits: dict) -> dict[str, np.ndarray]:
    patient_ids = patient_ids.astype(str)
    masks = {}
    for split_name in ("train", "val", "test"):
        split_patients = set(splits.get(split_name, []))
        masks[split_name] = np.array([pid in split_patients for pid in patient_ids], dtype=bool)
    masks["all"] = np.ones(len(patient_ids), dtype=bool)
    return masks


def summarize_values(values: np.ndarray, n_total_anchors: int) -> dict[str, float | int | None]:
    n_valid = int(values.size)
    coverage = float(n_valid / n_total_anchors) if n_total_anchors > 0 else 0.0
    if n_valid == 0:
        return {
            "n_total_anchors": int(n_total_anchors),
            "n_valid": 0,
            "coverage": coverage,
            "mean": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "range": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "iqr": None,
            "lower_fence": None,
            "upper_fence": None,
            "n_outliers_iqr": 0,
            "outlier_fraction_iqr": 0.0,
        }

    values = values.astype(np.float64, copy=False)
    q01, q05, q25, q50, q75, q95, q99 = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    iqr = float(q75 - q25)
    lower_fence = float(q25 - 1.5 * iqr)
    upper_fence = float(q75 + 1.5 * iqr)
    outlier_mask = (values < lower_fence) | (values > upper_fence)
    n_outliers = int(outlier_mask.sum())

    return {
        "n_total_anchors": int(n_total_anchors),
        "n_valid": n_valid,
        "coverage": coverage,
        "mean": float(values.mean()),
        "std": float(values.std()),
        "variance": float(values.var()),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "p01": float(q01),
        "p05": float(q05),
        "p25": float(q25),
        "p50": float(q50),
        "p75": float(q75),
        "p95": float(q95),
        "p99": float(q99),
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "n_outliers_iqr": n_outliers,
        "outlier_fraction_iqr": float(n_outliers / n_valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute descriptive stats for each regression target")
    parser.add_argument("--target-path", type=Path, required=True)
    parser.add_argument("--splits-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    bundle = np.load(args.target_path, allow_pickle=True)
    feature_targets = bundle["feature_targets"]
    feature_mask = bundle["feature_mask"].astype(bool)
    patient_ids = bundle["anchor_patient_ids"].astype(str)

    splits = json.loads(args.splits_path.read_text())
    split_masks = patient_split_masks(patient_ids, splits)
    target_names = load_feature_target_names(args.target_path, feature_targets.shape[1])

    results = {
        "target_path": str(args.target_path),
        "splits_path": str(args.splits_path),
        "n_anchors": int(feature_targets.shape[0]),
        "n_targets": int(feature_targets.shape[1]),
        "targets": [],
    }
    csv_rows: list[dict[str, object]] = []

    for col_idx, target_name in enumerate(target_names):
        target_entry = {
            "target_key": target_name,
            "column_index": col_idx,
            "splits": {},
        }
        target_col = feature_targets[:, col_idx]
        mask_col = feature_mask[:, col_idx] & np.isfinite(target_col)

        for split_name, split_mask in split_masks.items():
            joint_mask = split_mask & mask_col
            values = target_col[joint_mask]
            split_stats = summarize_values(values, int(split_mask.sum()))
            target_entry["splits"][split_name] = split_stats

            csv_row = {
                "target_key": target_name,
                "column_index": col_idx,
                "split": split_name,
            }
            csv_row.update(split_stats)
            csv_rows.append(csv_row)

        results["targets"].append(target_entry)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))

    fieldnames = [
        "target_key",
        "column_index",
        "split",
        "n_total_anchors",
        "n_valid",
        "coverage",
        "mean",
        "std",
        "variance",
        "min",
        "max",
        "range",
        "p01",
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
        "iqr",
        "lower_fence",
        "upper_fence",
        "n_outliers_iqr",
        "outlier_fraction_iqr",
    ]
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
