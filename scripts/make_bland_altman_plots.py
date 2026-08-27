from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_regression_prediction_files(base_dir: Path) -> list[Path]:
    prediction_files: list[Path] = []
    for pred_path in sorted(base_dir.rglob("test_predictions.npz")):
        if pred_path.parent.name.startswith("feature_"):
            prediction_files.append(pred_path)
    return prediction_files


def compute_bland_altman_stats(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    differences = predictions - targets
    means = 0.5 * (predictions + targets)

    bias = float(np.mean(differences))
    diff_std = float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
    loa_delta = 1.96 * diff_std
    loa_lower = float(bias - loa_delta)
    loa_upper = float(bias + loa_delta)

    return {
        "n_valid": int(differences.size),
        "bias": bias,
        "difference_std": diff_std,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
        "mean_of_means": float(np.mean(means)),
        "std_of_means": float(np.std(means, ddof=1)) if means.size > 1 else 0.0,
    }


def save_bland_altman_plot(
    predictions: np.ndarray,
    targets: np.ndarray,
    stats: dict[str, float],
    target_key: str,
    output_path: Path,
) -> None:
    differences = predictions - targets
    means = 0.5 * (predictions + targets)
    target_mean = float(np.mean(targets))
    target_std = float(np.std(targets, ddof=1)) if targets.size > 1 else 0.0
    target_percentiles = np.percentile(targets, [1, 5, 25, 50, 75, 95, 99])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(means, differences, s=8, alpha=0.25, edgecolors="none")
    ax.axhline(stats["bias"], color="tab:red", linestyle="-", linewidth=1.5, label=f"Bias = {stats['bias']:.3f}")
    ax.axhline(
        stats["loa_upper"],
        color="tab:orange",
        linestyle="--",
        linewidth=1.25,
        label=f"+1.96 SD = {stats['loa_upper']:.3f}",
    )
    ax.axhline(
        stats["loa_lower"],
        color="tab:orange",
        linestyle="--",
        linewidth=1.25,
        label=f"-1.96 SD = {stats['loa_lower']:.3f}",
    )
    ax.set_title(f"Bland-Altman: {target_key}")
    ax.set_xlabel("Mean of prediction and target")
    ax.set_ylabel("Prediction - target")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    summary_text = "\n".join(
        [
            f"mean +/- std: {target_mean:.3f} +/- {target_std:.3f}",
            f"p01: {target_percentiles[0]:.3f}",
            f"p05: {target_percentiles[1]:.3f}",
            f"p25: {target_percentiles[2]:.3f}",
            f"p50: {target_percentiles[3]:.3f}",
            f"p75: {target_percentiles[4]:.3f}",
            f"p95: {target_percentiles[5]:.3f}",
            f"p99: {target_percentiles[6]:.3f}",
        ]
    )
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


def process_prediction_file(
    pred_path: Path,
    plot_name: str,
    summary_name: str,
    output_dir: Path | None,
) -> dict[str, object] | None:
    with np.load(pred_path, allow_pickle=True) as data:
        task = str(data["task"])
        if task != "feature":
            return None

        predictions = data["predictions"].astype(np.float64)
        targets = data["targets"].astype(np.float64)
        masks = data["masks"].astype(bool)
        target_key = str(data["target_key"])

    valid = masks & np.isfinite(predictions) & np.isfinite(targets)
    if not np.any(valid):
        return {
            "target_key": target_key,
            "prediction_file": str(pred_path),
            "status": "no_valid_samples",
        }

    pred_valid = predictions[valid]
    target_valid = targets[valid]
    stats = compute_bland_altman_stats(pred_valid, target_valid)

    if output_dir is None:
        task_output_dir = pred_path.parent
        plot_path = task_output_dir / plot_name
        summary_path = task_output_dir / summary_name
    else:
        task_output_dir = output_dir
        task_output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = task_output_dir / f"{target_key}_{plot_name}"
        summary_path = task_output_dir / f"{target_key}_{summary_name}"

    save_bland_altman_plot(pred_valid, target_valid, stats, target_key, plot_path)

    summary = {
        "target_key": target_key,
        "prediction_file": str(pred_path),
        "plot_path": str(plot_path),
        **stats,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Bland-Altman plots from saved PatchTST regression predictions."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="outputs/patchtst/vasopressor_free",
        help="Directory containing per-task test_predictions.npz files.",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="bland_altman.png",
        help="Filename to use for each saved plot inside a task directory.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="bland_altman_stats.json",
        help="Filename to use for each saved summary JSON inside a task directory.",
    )
    parser.add_argument(
        "--combined-summary",
        type=str,
        default="bland_altman_summary.json",
        help="Filename for the combined summary at the base directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory for a flat export of plots and summaries.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    prediction_files = find_regression_prediction_files(base_dir)
    if not prediction_files:
        raise SystemExit(f"No regression prediction files found under {base_dir}")

    combined_results: list[dict[str, object]] = []
    for pred_path in prediction_files:
        result = process_prediction_file(pred_path, args.plot_name, args.summary_name, output_dir)
        if result is not None:
            combined_results.append(result)
            print(f"Saved Bland-Altman outputs for {pred_path.parent.name}", flush=True)

    combined_summary_root = output_dir if output_dir is not None else base_dir
    combined_summary_root.mkdir(parents=True, exist_ok=True)
    combined_summary_path = combined_summary_root / args.combined_summary
    with open(combined_summary_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"Saved combined summary: {combined_summary_path}", flush=True)


if __name__ == "__main__":
    main()
