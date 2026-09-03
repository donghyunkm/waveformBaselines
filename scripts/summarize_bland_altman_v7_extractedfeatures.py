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

from scripts.summarize_bland_altman_full_features import (
    BLAND_ALTMAN_TOLERANCES,
    best_possible_row,
    bland_altman_stats,
    bootstrap_cis,
    format_estimate,
    format_pct,
    load_prediction_data,
    write_csv,
)


SUMMARY_JSON = "bland_altman_v7_extractedfeatures_summary.json"
DEFAULT_OUTPUT_DIR = Path("blandaltman_v7_extractedfeatures")
DEFAULT_DOC = Path("docs/v7_extracted_features/extractedFeaturesRegression.md")
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_SEED = 42
ANCHOR_TIME_DECIMALS = 6


def load_existing_analyses(output_dir: Path) -> list[dict[str, Any]]:
    summary_path = output_dir / SUMMARY_JSON
    analyses = json.loads(summary_path.read_text())
    if not isinstance(analyses, list):
        raise ValueError(f"{summary_path} must contain a list")
    for row in analyses:
        for key in ("target", "target_base", "best_model", "prediction_file", "plot_path", "stats_path"):
            if key not in row:
                raise ValueError(f"{summary_path} row missing {key}: {row}")
    return sorted(analyses, key=lambda row: row["target_base"])


TRAIN_MEAN_CACHE: dict[tuple[str, str], dict[str, float]] = {}


def training_means_original_units(target_path: Path, cache_dir: Path) -> dict[str, float]:
    cache_key = (str(target_path), str(cache_dir))
    if cache_key in TRAIN_MEAN_CACHE:
        return TRAIN_MEAN_CACHE[cache_key]
    metadata_path = target_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    target_names = list(metadata.get("feature_target_names", []))
    if not target_names:
        raise ValueError(f"No feature_target_names found in {metadata_path}")
    with np.load(target_path, allow_pickle=True) as target_data:
        targets = np.asarray(target_data["feature_targets"], dtype=np.float64)
        target_mask = np.asarray(target_data["feature_mask"], dtype=bool)
        target_patient_ids = np.asarray(target_data["anchor_patient_ids"]).astype(str)
        target_anchor_times = np.asarray(target_data["anchor_times"], dtype=np.float64)
    cache_patient_ids = np.asarray(np.load(cache_dir / "patient_ids.npy", allow_pickle=True)).astype(str)
    cache_anchor_times = np.asarray(np.load(cache_dir / "anchor_times.npy"), dtype=np.float64)
    split_labels = np.asarray(np.load(cache_dir / "split_labels.npy", allow_pickle=True)).astype(str)
    if not (targets.shape[0] == target_mask.shape[0] == target_patient_ids.shape[0] == target_anchor_times.shape[0]):
        raise ValueError(f"Target bundle row-count mismatch in {target_path}")
    if not (cache_patient_ids.shape == cache_anchor_times.shape == split_labels.shape):
        raise ValueError(f"Cache metadata shape mismatch in {cache_dir}")
    if target_patient_ids.shape == cache_patient_ids.shape and np.array_equal(target_patient_ids, cache_patient_ids) and np.allclose(target_anchor_times, cache_anchor_times, rtol=0.0, atol=5e-7):
        train_rows = split_labels == "train"
    else:
        cache_keys = [
            (pid, int(np.rint(float(anchor_time) * (10**ANCHOR_TIME_DECIMALS))))
            for pid, anchor_time in zip(cache_patient_ids.tolist(), cache_anchor_times.tolist())
        ]
        split_by_key = dict(zip(cache_keys, split_labels.tolist()))
        target_keys = [
            (pid, int(np.rint(float(anchor_time) * (10**ANCHOR_TIME_DECIMALS))))
            for pid, anchor_time in zip(target_patient_ids.tolist(), target_anchor_times.tolist())
        ]
        train_rows = np.asarray([split_by_key.get(key) == "train" for key in target_keys], dtype=bool)
        if int(train_rows.sum()) == 0:
            raise ValueError(f"No train rows aligned between {target_path} and {cache_dir}")
    means: dict[str, float] = {}
    for col, target in enumerate(target_names):
        valid = train_rows & target_mask[:, col] & np.isfinite(targets[:, col])
        if not np.any(valid):
            raise ValueError(f"No valid training targets for {target}")
        means[target] = float(np.mean(targets[valid, col]))
    TRAIN_MEAN_CACHE[cache_key] = means
    return means


def train_mean_for_analysis(analysis: dict[str, Any]) -> tuple[float, str, str]:
    config_path = Path(analysis["prediction_file"]).parent / "config.json"
    config = json.loads(config_path.read_text())
    target_path = Path(config["target_path"])
    cache_dir = Path(config["cache_dir"])
    means = training_means_original_units(target_path, cache_dir)
    target = analysis["target"]
    if target not in means:
        raise ValueError(f"Target {target!r} not found in training means from {target_path}")
    return means[target], str(target_path), str(cache_dir)


def null_model_row(analysis: dict[str, Any], references: np.ndarray, patient_ids: np.ndarray, tolerance: float | None, n_bootstrap: int, seed: int) -> dict[str, Any]:
    train_mean, target_path, cache_dir = train_mean_for_analysis(analysis)
    null_predictions = np.full(references.shape, train_mean, dtype=np.float64)
    point = bland_altman_stats(null_predictions, references, tolerance)
    boot = bootstrap_cis(null_predictions, references, patient_ids, tolerance, n_bootstrap=n_bootstrap, seed=seed)
    test_mean = float(np.mean(references))
    bias_expected = train_mean - test_mean
    slope_expected = -2.0
    return {
        "target": analysis["target"],
        "model": "TRAIN-MEAN NULL MODEL",
        "row_type": "train_mean_null_model",
        "baseline_type": "train_mean",
        "baseline_prediction_value": train_mean,
        "training_target_mean": train_mean,
        "target_path": target_path,
        "cache_dir": cache_dir,
        "test_reference_mean": test_mean,
        "bias_expected_from_train_minus_test_mean": bias_expected,
        "bias_matches_train_minus_test_mean": bool(abs(float(point["bias"]) - bias_expected) <= 1e-10),
        "null_slope_expected": slope_expected,
        "null_slope_abs_diff_from_minus_two": abs(float(point["proportional_bias_slope"]) - slope_expected),
        "null_slope_sanity_check_passed": bool(abs(float(point["proportional_bias_slope"]) - slope_expected) <= 1e-10),
        "tolerance": tolerance,
        "n_predictions": int(references.size),
        "n_patients": int(np.unique(patient_ids).size),
        "prediction_file": analysis["prediction_file"],
        "plot_path": None,
        "bootstrap_replicates": n_bootstrap,
        **point,
        **boot,
    }


def model_row(analysis: dict[str, Any], n_bootstrap: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    target = analysis["target"]
    model = analysis["best_model"]
    tolerance = BLAND_ALTMAN_TOLERANCES.get(target)
    predictions, references, patient_ids = load_prediction_data(Path(analysis["prediction_file"]))
    point = bland_altman_stats(predictions, references, tolerance)
    boot = bootstrap_cis(predictions, references, patient_ids, tolerance, n_bootstrap=n_bootstrap, seed=seed)
    actual = {
        "target": target,
        "target_base": analysis["target_base"],
        "model": model,
        "row_type": "model",
        "tolerance": tolerance,
        "n_predictions": int(predictions.size),
        "n_patients": int(np.unique(patient_ids).size),
        "prediction_file": analysis["prediction_file"],
        "plot_path": analysis["plot_path"],
        "bootstrap_replicates": n_bootstrap,
        **point,
        **boot,
    }
    null_row = null_model_row(analysis, references, patient_ids, tolerance, n_bootstrap=n_bootstrap, seed=seed)
    null_row["target_base"] = analysis["target_base"]
    return actual, null_row


def validate_against_existing_plot_stats(rows: list[dict[str, Any]], analyses: list[dict[str, Any]], required_targets: set[str]) -> list[dict[str, Any]]:
    model_rows = {(row["target"], row["model"]): row for row in rows if row["row_type"] == "model"}
    checks = []
    for analysis in analyses:
        target = analysis["target"]
        if target not in required_targets:
            continue
        stats_path = Path(analysis["stats_path"])
        existing = json.loads(stats_path.read_text())
        row = model_rows[(target, analysis["best_model"])]
        bias = float(row["bias"])
        lower = bias - float(row["loa_half_width"])
        upper = bias + float(row["loa_half_width"])
        check = {
            "target": target,
            "model": analysis["best_model"],
            "stats_path": str(stats_path),
            "bias_abs_diff": abs(bias - float(existing["bias"])),
            "lower_loa_abs_diff": abs(lower - float(existing["loa_lower"])),
            "upper_loa_abs_diff": abs(upper - float(existing["loa_upper"])),
        }
        check["matches_existing_plot_stats"] = all(check[key] <= 1e-9 for key in ("bias_abs_diff", "lower_loa_abs_diff", "upper_loa_abs_diff"))
        checks.append(check)
    found = {row["target"] for row in checks}
    missing = required_targets - found
    if missing:
        raise ValueError(f"Missing validation targets: {sorted(missing)}")
    return checks


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "## Bland-Altman Agreement Summary",
        "",
        "Generated on `2026-09-01` for the documented best completed v7 extracted-feature regression model per target in the table above. Statistics use `difference = prediction - reference` and `average = (prediction + reference) / 2`, matching the plot-generation convention. Model rows use patient-cluster bootstrap percentile 95% CIs with patients resampled as clusters and all windows retained for sampled patients. The `TRAIN-MEAN NULL MODEL` predicts the training-set mean of the corresponding target for every test observation, in original physical units; brackets on model and null rows are patient-cluster bootstrap percentile 95% CIs.",
        "",
        f"Bootstrap replicates: `{DEFAULT_BOOTSTRAP_REPLICATES}` by default. Random seed: `{DEFAULT_SEED}` by default. No repository-defined clinically meaningful tolerances were found for these regression targets, so `Within Tolerance` is `NA` until `BLAND_ALTMAN_TOLERANCES` in `scripts/summarize_bland_altman_full_features.py` is populated with explicit defensible thresholds.",
        "",
        "**Reference rows.** `PERFECT PREDICTION` denotes exact agreement (`prediction = reference`), giving zero bias, zero LoA width, zero proportional-bias slope, and 100% coverage. `TRAIN-MEAN NULL MODEL` predicts the target's training-set mean for every test observation and represents a model with no patient- or window-specific predictive information. The null prediction is derived exclusively from the training data. Lower absolute bias is better, lower LoA half-width is better, proportional-bias slope closer to zero is better, and higher coverage is better. Do not compare raw Bias or LoA half-width across targets with different physical units; compare those primarily within a target.",
        "",
        "Artifacts: `blandaltman_v7_extractedfeatures/` contains `26` PNG plots, per-plot JSON files, `bland_altman_v7_extractedfeatures_summary.json`, and machine-readable agreement summaries `bland_altman_agreement_summary.{csv,json,md}`.",
        "",
        "| Target | Model | Bias [95% CI] | LoA Half-Width [95% CI] | Proportional-Bias Slope [95% CI] | Within Tolerance [95% CI] | N Predictions | N Patients |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        display_target = row.get("target_base", row["target"])
        if row["row_type"] == "theoretical_best":
            bias = "0"
            loa = "0"
            slope = "0"
            coverage = "100%"
            n_predictions = "-"
            n_patients = "-"
        else:
            bias = format_estimate(row, "bias")
            loa = format_estimate(row, "loa_half_width")
            slope = format_estimate(row, "proportional_bias_slope")
            coverage = format_pct(row, "coverage_probability")
            n_predictions = str(row["n_predictions"])
            n_patients = str(row["n_patients"])
        lines.append(f"| `{display_target}` | `{row['model']}` | {bias} | {loa} | {slope} | {coverage} | {n_predictions} | {n_patients} |")
    return "\n".join(lines) + "\n"


def update_doc(doc_path: Path, section: str) -> None:
    text = doc_path.read_text()
    if "## Bland-Altman Agreement Summary" in text:
        text = re.sub(r"## Bland-Altman Agreement Summary\n.*?(?=\n## |\Z)", section.rstrip() + "\n", text, flags=re.S)
    else:
        marker = "\n## Results Analysis\n"
        if marker not in text:
            raise ValueError(f"Could not find insertion marker in {doc_path}")
        text = text.replace(marker, "\n" + section + marker)
    doc_path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v7 extracted-feature best-model Bland-Altman analyses with train-mean null rows.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-doc-update", action="store_true")
    args = parser.parse_args()

    analyses = load_existing_analyses(args.output_dir)
    rows: list[dict[str, Any]] = [best_possible_row()]
    rows[0]["target_base"] = "ALL TARGETS"
    train_mean_checks = []
    for analysis in analyses:
        train_mean, target_path, cache_dir = train_mean_for_analysis(analysis)
        train_mean_checks.append({
            "target": analysis["target"],
            "target_base": analysis["target_base"],
            "training_mean": train_mean,
            "target_path": target_path,
            "cache_dir": cache_dir,
        })
        rows.extend(model_row(analysis, args.bootstrap_replicates, args.seed))
    validation = validate_against_existing_plot_stats(rows, analyses, {"HR_t_plus_0m_gap", "RR_t_plus_0m_gap", "SBP_t_plus_0m_gap"})

    summary_json = args.output_dir / "bland_altman_agreement_summary.json"
    summary_csv = args.output_dir / "bland_altman_agreement_summary.csv"
    summary_md = args.output_dir / "bland_altman_agreement_summary.md"
    payload = {
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "tolerances": BLAND_ALTMAN_TOLERANCES,
        "training_mean_checks": train_mean_checks,
        "validation_checks": validation,
        "rows": rows,
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_csv(summary_csv, rows)
    section = markdown_table(rows)
    summary_md.write_text(section)
    if not args.no_doc_update:
        update_doc(args.doc, section)
    print(json.dumps({
        "analyses_found": len(analyses),
        "rows_written": len(rows),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "validation_checks": validation,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
