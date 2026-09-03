#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


DEFAULT_DOC = Path("docs/v7_extracted_features/extractedFeaturesRegression.md")
DEFAULT_SIGNIFICANCE_JSONS = [
    Path("outputs/feature_models/regression_patchtst_v7_significance_with_mlp_2026-08-29.json"),
    Path("outputs/feature_models/regression_patchtst_v7_significance_tcn_only_2026-08-29.json"),
]
DEFAULT_OUTPUT_DIR = Path("blandaltman_v7_extractedfeatures")
TARGET_SUFFIX = "_t_plus_0m_gap"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_doc_best_rows(doc_path: Path) -> list[dict[str, str]]:
    text = doc_path.read_text()
    marker = "### Best Completed v7 Per Target"
    if marker not in text:
        raise ValueError(f"Could not find {marker!r} in {doc_path}")
    section = text.split(marker, 1)[1].split("\n## ", 1)[0]
    rows: list[dict[str, str]] = []
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        target_match = re.search(r"`([^`]+)`", cells[0])
        model_match = re.search(r"`([^`]+)`", cells[1])
        if not target_match or not model_match:
            continue
        target_base = target_match.group(1)
        model = model_match.group(1)
        rows.append({
            "target_base": target_base,
            "target": f"{target_base}{TARGET_SUFFIX}",
            "best_model": model,
        })
    if len(rows) != 26:
        raise ValueError(f"Expected 26 best rows in {doc_path}, found {len(rows)}")
    return rows


def load_significance_results(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload.get("results", []):
            target = str(row["target"])
            model = str(row["model"])
            results[(target, model)] = row
    return results


def resolve_prediction_file(best_row: dict[str, str], significance_rows: dict[tuple[str, str], dict[str, Any]]) -> Path:
    key = (best_row["target_base"], best_row["best_model"])
    if key not in significance_rows:
        raise ValueError(f"No significance result found for {key}")
    prediction_file = Path(significance_rows[key]["prediction_file"])
    if not prediction_file.exists():
        raise FileNotFoundError(f"Missing prediction file for {key}: {prediction_file}")
    return prediction_file


def load_predictions(prediction_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(prediction_file, allow_pickle=True) as data:
        predictions = np.asarray(data["predictions"], dtype=np.float64)
        targets = np.asarray(data["targets"], dtype=np.float64)
        patient_ids = np.asarray(data["patient_ids"]).astype(str)
        masks = np.asarray(data["masks"], dtype=bool) if "masks" in data.files else np.ones(targets.shape[0], dtype=bool)
    valid = masks & np.isfinite(predictions) & np.isfinite(targets)
    if not np.any(valid):
        raise ValueError(f"No valid prediction rows in {prediction_file}")
    return predictions[valid], targets[valid], patient_ids[valid]


def save_plot(predictions: np.ndarray, targets: np.ndarray, stats: dict[str, float], title: str, output_path: Path) -> None:
    differences = predictions - targets
    means = 0.5 * (predictions + targets)
    target_percentiles = np.percentile(targets, [1, 5, 25, 50, 75, 95, 99])
    target_mean = float(np.mean(targets))
    target_std = float(np.std(targets, ddof=1)) if targets.size > 1 else 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(means, differences, s=8, alpha=0.25, edgecolors="none", rasterized=True)
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
        f"bias: {stats['bias']:.3f}",
        f"LoA: [{stats['loa_lower']:.3f}, {stats['loa_upper']:.3f}]",
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


def process_row(best_row: dict[str, str], prediction_file: Path, output_dir: Path) -> dict[str, Any]:
    predictions, targets, patient_ids = load_predictions(prediction_file)
    stats = compute_bland_altman_stats(predictions, targets)
    stem = f"{safe_name(best_row['target'])}__{safe_name(best_row['best_model'])}"
    plot_path = output_dir / f"{stem}.png"
    stats_path = output_dir / f"{stem}.json"
    save_plot(predictions, targets, stats, f"Bland-Altman: {best_row['target']} ({best_row['best_model']})", plot_path)
    summary = {
        "target": best_row["target"],
        "target_base": best_row["target_base"],
        "best_model": best_row["best_model"],
        "prediction_file": str(prediction_file),
        "plot_path": str(plot_path),
        "stats_path": str(stats_path),
        "n_test_predictions": int(predictions.size),
        "n_test_patients": int(np.unique(patient_ids).size),
        **stats,
    }
    stats_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Bland-Altman plots for documented best v7 extracted-feature regression models.")
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--significance-json", type=Path, nargs="+", default=DEFAULT_SIGNIFICANCE_JSONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--combined-summary", type=str, default="bland_altman_v7_extractedfeatures_summary.json")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_rows = parse_doc_best_rows(args.doc)
    significance_rows = load_significance_results(args.significance_json)
    summaries = [
        process_row(row, resolve_prediction_file(row, significance_rows), args.output_dir)
        for row in best_rows
    ]
    combined_path = args.output_dir / args.combined_summary
    combined_path.write_text(json.dumps(summaries, indent=2, sort_keys=True))
    print(json.dumps({"plots": len(summaries), "output_dir": str(args.output_dir), "combined_summary": str(combined_path)}, indent=2))


if __name__ == "__main__":
    main()
