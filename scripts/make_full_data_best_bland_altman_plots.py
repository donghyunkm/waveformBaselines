#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.make_bland_altman_plots import compute_bland_altman_stats

DEFAULT_SUMMARY_CSV = Path("outputs/feature_models/full_data_regression_patient_bootstrap_ci_2026-09-01.csv")
DEFAULT_OUTPUT_DIR = Path("blandaltman_full_features")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_rows(summary_csv: Path) -> list[dict[str, Any]]:
    with summary_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def select_best_by_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = row["target"]
        if target not in best or float(row["r2"]) > float(best[target]["r2"]):
            best[target] = row
    return [best[target] for target in sorted(best)]


def load_predictions(pred_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(pred_path, allow_pickle=True) as data:
        predictions = np.asarray(data["predictions"], dtype=np.float64)
        targets = np.asarray(data["targets"], dtype=np.float64)
        patient_ids = np.asarray(data["patient_ids"]).astype(str)
        masks = np.asarray(data["masks"], dtype=bool) if "masks" in data.files else np.ones(targets.shape[0], dtype=bool)
    valid = masks & np.isfinite(predictions) & np.isfinite(targets)
    if not np.any(valid):
        raise ValueError(f"No valid prediction rows in {pred_path}")
    return predictions[valid], targets[valid], patient_ids[valid]


def save_plot(
    predictions: np.ndarray,
    targets: np.ndarray,
    stats: dict[str, float],
    title: str,
    output_path: Path,
) -> None:
    differences = predictions - targets
    means = 0.5 * (predictions + targets)
    target_percentiles = np.percentile(targets, [1, 5, 25, 50, 75, 95, 99])
    target_mean = float(np.mean(targets))
    target_std = float(np.std(targets, ddof=1)) if targets.size > 1 else 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(means, differences, s=3, alpha=0.10, edgecolors="none", rasterized=True)
    ax.axhline(stats["bias"], color="tab:red", linestyle="-", linewidth=1.5, label=f"Bias = {stats['bias']:.3f}")
    ax.axhline(stats["loa_upper"], color="tab:orange", linestyle="--", linewidth=1.25, label=f"+1.96 SD = {stats['loa_upper']:.3f}")
    ax.axhline(stats["loa_lower"], color="tab:orange", linestyle="--", linewidth=1.25, label=f"-1.96 SD = {stats['loa_lower']:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Mean of prediction and target")
    ax.set_ylabel("Prediction - target")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    summary_text = "\n".join([
        f"n: {targets.size}",
        f"target mean +/- std: {target_mean:.3f} +/- {target_std:.3f}",
        f"p01: {target_percentiles[0]:.3f}",
        f"p05: {target_percentiles[1]:.3f}",
        f"p25: {target_percentiles[2]:.3f}",
        f"p50: {target_percentiles[3]:.3f}",
        f"p75: {target_percentiles[4]:.3f}",
        f"p95: {target_percentiles[5]:.3f}",
        f"p99: {target_percentiles[6]:.3f}",
    ])
    ax.text(
        0.98,
        0.02,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def process_best_row(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = Path(row["output_dir"]) / "test_predictions.npz"
    predictions, targets, patient_ids = load_predictions(pred_path)
    stats = compute_bland_altman_stats(predictions, targets)
    target = row["target"]
    model = row["model"]
    stem = f"{safe_name(target)}__{safe_name(model)}"
    plot_path = output_dir / f"{stem}.png"
    stats_path = output_dir / f"{stem}.json"
    save_plot(predictions, targets, stats, f"Bland-Altman: {target} ({model})", plot_path)
    summary = {
        "target": target,
        "best_model": model,
        "prediction_file": str(pred_path),
        "plot_path": str(plot_path),
        "stats_path": str(stats_path),
        "n_test_predictions": int(predictions.size),
        "n_test_patients": int(np.unique(patient_ids).size),
        **stats,
    }
    stats_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Bland-Altman plots for best full-data extracted-feature regression models.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--combined-summary", type=str, default="bland_altman_full_features_summary.json")
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    best_rows = select_best_by_target(rows)
    if len(best_rows) != 26:
        raise ValueError(f"Expected 26 best target rows, found {len(best_rows)}")
    summaries = [process_best_row(row, args.output_dir) for row in best_rows]
    combined_path = args.output_dir / args.combined_summary
    combined_path.write_text(json.dumps(summaries, indent=2, sort_keys=True))
    print(json.dumps({"plots": len(summaries), "output_dir": str(args.output_dir), "combined_summary": str(combined_path)}, indent=2))


if __name__ == "__main__":
    main()
