#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.summarize_bland_altman_full_features import (
    BLAND_ALTMAN_TOLERANCES,
    bland_altman_stats,
    bootstrap_cis,
    format_estimate,
    load_prediction_data,
    patient_sufficient_stats,
    stats_from_patient_counts,
)
from scripts.summarize_bland_altman_v7_extractedfeatures import load_existing_analyses


DEFAULT_OUTPUT_DIR = Path("blandaltman_v7_extractedfeatures")
DEFAULT_DOC = Path("docs/v7_extracted_features/extractedFeaturesRegression.md")
DEFAULT_V7_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/regression")
DEFAULT_PATCHTST_ROOT = Path("outputs/patchtst/vasopressor_free_v1_es")
DEFAULT_TARGET_PATH = Path("outputs/targets/feature_targets_gap_vasopressor_free.npz")
DEFAULT_SPLITS_PATH = Path("outputs/splits/vasopressor_free_splits.json")
DEFAULT_WAVEFORM_DIR = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms")
DEFAULT_CHANNELS = "ABP,II,PLETH"
DEFAULT_SEQ_LEN = 150000
DEFAULT_BOOTSTRAP_REPLICATES = 10000
DEFAULT_SEED = 20260901
V7_SOURCE_MODEL = "gru"
V7_MODEL_LABEL = "V7"
PATCHTST_MODEL_NAME = "PatchTST v1 raw waveform"
ANCHOR_TIME_DECIMALS = 6
METRIC_ORDER = ("absolute_bias", "loa_half_width", "absolute_proportional_bias_slope", "within_tolerance")
METRIC_LABELS = {
    "absolute_bias": "Absolute Bias",
    "loa_half_width": "LoA Half-Width",
    "absolute_proportional_bias_slope": "Absolute Proportional-Bias Slope",
    "within_tolerance": "Within Tolerance",
}


@dataclass(frozen=True)
class PredictionSet:
    sample_ids: np.ndarray
    patient_ids: np.ndarray
    anchor_times: np.ndarray
    y: np.ndarray
    pred: np.ndarray


@dataclass(frozen=True)
class PairedPredictions:
    target: str
    target_base: str
    v7_model: str
    sample_ids: np.ndarray
    patient_ids: np.ndarray
    truth: np.ndarray
    v7_predictions: np.ndarray
    patchtst_predictions: np.ndarray
    n_unmatched_v7: int
    n_unmatched_patchtst: int
    n_dropped_nonfinite: int
    v7_prediction_file: str
    patchtst_prediction_file: str


def format_num(value: float | None, decimals: int = 4, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"


def v7_prediction_path(v7_root: Path, target: str) -> Path:
    return v7_root / f"{V7_SOURCE_MODEL}_feature_{target}_t_plus_0m_gap_v7" / "test_predictions.npz"


def load_v7_prediction_set(path: Path) -> PredictionSet:
    with np.load(path, allow_pickle=True) as data:
        for key in ("predictions", "targets", "patient_ids", "anchor_times"):
            if key not in data.files:
                raise ValueError(f"{path} missing {key}")
        predictions = np.asarray(data["predictions"], dtype=np.float64)
        targets = np.asarray(data["targets"], dtype=np.float64)
        patient_ids = np.asarray(data["patient_ids"]).astype(str)
        anchor_times = np.asarray(data["anchor_times"], dtype=np.float64)
        masks = np.asarray(data["masks"], dtype=bool) if "masks" in data.files else np.ones(targets.shape, dtype=bool)
    valid = masks & np.isfinite(predictions) & np.isfinite(targets)
    sample_ids = np.asarray([
        f"{pid}|{round(float(t), ANCHOR_TIME_DECIMALS):.6f}"
        for pid, t in zip(patient_ids[valid].tolist(), anchor_times[valid].tolist())
    ])
    return PredictionSet(sample_ids, patient_ids[valid], anchor_times[valid], targets[valid], predictions[valid])


def load_patchtst_prediction_set(target_base: str, patch_args: SimpleNamespace) -> PredictionSet:
    from scripts.regression_significance import load_patchtst

    return load_patchtst(target_base, patch_args)


def align_prediction_sets(
    *,
    target: str,
    target_base: str,
    v7_model: str,
    v7: PredictionSet,
    patchtst: PredictionSet,
    v7_prediction_file: Path,
    patchtst_prediction_file: Path,
    target_atol: float = 1e-6,
) -> PairedPredictions:
    v7_ids = np.asarray(v7.sample_ids).astype(str)
    patch_ids = np.asarray(patchtst.sample_ids).astype(str)
    if len(set(v7_ids.tolist())) != len(v7_ids):
        raise ValueError(f"{target}: duplicate V7 sample IDs")
    if len(set(patch_ids.tolist())) != len(patch_ids):
        raise ValueError(f"{target}: duplicate PatchTST sample IDs")
    patch_index = {sid: idx for idx, sid in enumerate(patch_ids.tolist())}
    paired_v7_idx: list[int] = []
    paired_patch_idx: list[int] = []
    for idx, sid in enumerate(v7_ids.tolist()):
        patch_idx = patch_index.get(sid)
        if patch_idx is not None:
            paired_v7_idx.append(idx)
            paired_patch_idx.append(patch_idx)
    if not paired_v7_idx:
        raise ValueError(f"{target}: no paired observations")
    vi = np.asarray(paired_v7_idx, dtype=np.int64)
    pi = np.asarray(paired_patch_idx, dtype=np.int64)
    if not np.array_equal(v7.patient_ids[vi].astype(str), patchtst.patient_ids[pi].astype(str)):
        raise ValueError(f"{target}: patient IDs differ after sample-ID alignment")
    if not np.array_equal(v7_ids[vi], patch_ids[pi]):
        raise ValueError(f"{target}: sample IDs differ after alignment")
    if not np.allclose(v7.y[vi].astype(float), patchtst.y[pi].astype(float), rtol=0.0, atol=target_atol):
        raise ValueError(f"{target}: ground-truth values differ after sample-ID alignment")
    finite = np.isfinite(v7.pred[vi]) & np.isfinite(patchtst.pred[pi]) & np.isfinite(v7.y[vi])
    if not np.any(finite):
        raise ValueError(f"{target}: no finite paired predictions")
    vi = vi[finite]
    pi = pi[finite]
    return PairedPredictions(
        target=target,
        target_base=target_base,
        v7_model=v7_model,
        sample_ids=v7_ids[vi],
        patient_ids=v7.patient_ids[vi].astype(str),
        truth=v7.y[vi].astype(np.float64),
        v7_predictions=v7.pred[vi].astype(np.float64),
        patchtst_predictions=patchtst.pred[pi].astype(np.float64),
        n_unmatched_v7=int(len(v7_ids) - len(paired_v7_idx)),
        n_unmatched_patchtst=int(len(patch_ids) - len(paired_patch_idx)),
        n_dropped_nonfinite=int(len(paired_v7_idx) - int(finite.sum())),
        v7_prediction_file=str(v7_prediction_file),
        patchtst_prediction_file=str(patchtst_prediction_file),
    )


def _advantage_from_stats(v7_stats: dict[str, np.ndarray], patch_stats: dict[str, np.ndarray], metric: str) -> np.ndarray:
    if metric == "absolute_bias":
        return np.abs(patch_stats["bias"]) - np.abs(v7_stats["bias"])
    if metric == "loa_half_width":
        return patch_stats["loa_half_width"] - v7_stats["loa_half_width"]
    if metric == "absolute_proportional_bias_slope":
        return np.abs(patch_stats["proportional_bias_slope"]) - np.abs(v7_stats["proportional_bias_slope"])
    if metric == "within_tolerance":
        if v7_stats["coverage_probability"] is None or patch_stats["coverage_probability"] is None:
            raise ValueError("within_tolerance requires a defined tolerance")
        return v7_stats["coverage_probability"] - patch_stats["coverage_probability"]
    raise ValueError(f"Unknown metric: {metric}")


def point_comparison_metrics(
    v7_predictions: np.ndarray,
    patchtst_predictions: np.ndarray,
    truth: np.ndarray,
    tolerance: float | None,
) -> dict[str, dict[str, float | None]]:
    v7 = bland_altman_stats(v7_predictions, truth, tolerance)
    patch = bland_altman_stats(patchtst_predictions, truth, tolerance)
    out = {
        "absolute_bias": {
            "v7": abs(float(v7["bias"])),
            "patchtst": abs(float(patch["bias"])),
            "advantage": abs(float(patch["bias"])) - abs(float(v7["bias"])),
        },
        "loa_half_width": {
            "v7": float(v7["loa_half_width"]),
            "patchtst": float(patch["loa_half_width"]),
            "advantage": float(patch["loa_half_width"]) - float(v7["loa_half_width"]),
        },
        "absolute_proportional_bias_slope": {
            "v7": abs(float(v7["proportional_bias_slope"])),
            "patchtst": abs(float(patch["proportional_bias_slope"])),
            "advantage": abs(float(patch["proportional_bias_slope"])) - abs(float(v7["proportional_bias_slope"])),
        },
    }
    if tolerance is None:
        out["within_tolerance"] = {"v7": None, "patchtst": None, "advantage": None}
    else:
        out["within_tolerance"] = {
            "v7": float(v7["coverage_probability"]),
            "patchtst": float(patch["coverage_probability"]),
            "advantage": float(v7["coverage_probability"]) - float(patch["coverage_probability"]),
        }
    return out


def bootstrap_advantage_cis(
    v7_predictions: np.ndarray,
    patchtst_predictions: np.ndarray,
    truth: np.ndarray,
    patient_ids: np.ndarray,
    tolerance: float | None,
    *,
    n_bootstrap: int,
    seed: int,
    sampled_patient_indices: np.ndarray | None = None,
) -> tuple[dict[str, dict[str, float | int | None]], np.ndarray]:
    v7_stats = patient_sufficient_stats(v7_predictions, truth, patient_ids, tolerance)
    patch_stats = patient_sufficient_stats(patchtst_predictions, truth, patient_ids, tolerance)
    patients = v7_stats["patients"]
    if not np.array_equal(patients, patch_stats["patients"]):
        raise AssertionError("V7 and PatchTST patient groups differ")
    n_patients = int(patients.size)
    if sampled_patient_indices is None:
        rng = np.random.default_rng(seed)
        sampled_patient_indices = rng.integers(0, n_patients, size=(n_bootstrap, n_patients))
    counts = np.zeros((sampled_patient_indices.shape[0], n_patients), dtype=np.float64)
    for idx, sampled in enumerate(sampled_patient_indices):
        counts[idx] = np.bincount(sampled, minlength=n_patients)
    sampled_v7 = stats_from_patient_counts(v7_stats, counts, tolerance)
    sampled_patch = stats_from_patient_counts(patch_stats, counts, tolerance)
    out: dict[str, dict[str, float | int | None]] = {}
    for metric in ("absolute_bias", "loa_half_width", "absolute_proportional_bias_slope"):
        values = _advantage_from_stats(sampled_v7, sampled_patch, metric)
        valid = values[np.isfinite(values)]
        out[metric] = {
            "ci_lower": float(np.percentile(valid, 2.5)),
            "ci_upper": float(np.percentile(valid, 97.5)),
            "valid_bootstrap_replicates": int(valid.size),
            "invalid_bootstrap_replicates": int(values.size - valid.size),
        }
    if tolerance is None:
        out["within_tolerance"] = {
            "ci_lower": None,
            "ci_upper": None,
            "valid_bootstrap_replicates": 0,
            "invalid_bootstrap_replicates": int(counts.shape[0]),
        }
    else:
        values = _advantage_from_stats(sampled_v7, sampled_patch, "within_tolerance")
        valid = values[np.isfinite(values)]
        out["within_tolerance"] = {
            "ci_lower": float(np.percentile(valid, 2.5)),
            "ci_upper": float(np.percentile(valid, 97.5)),
            "valid_bootstrap_replicates": int(valid.size),
            "invalid_bootstrap_replicates": int(values.size - valid.size),
        }
    return out, counts


def result_label(advantage: float | None, ci_lower: float | None, ci_upper: float | None) -> str:
    if advantage is None or ci_lower is None or ci_upper is None:
        return "Not defined"
    if not (np.isfinite(advantage) and np.isfinite(ci_lower) and np.isfinite(ci_upper)):
        return "Not defined"
    if advantage > 0 and ci_lower > 0:
        return "V7 significantly better"
    if advantage < 0 and ci_upper < 0:
        return "PatchTST significantly better"
    return "Not significant"


def comparison_rows_for_target(paired: PairedPredictions, *, n_bootstrap: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tolerance = BLAND_ALTMAN_TOLERANCES.get(paired.target)
    point = point_comparison_metrics(paired.v7_predictions, paired.patchtst_predictions, paired.truth, tolerance)
    cis, bootstrap_counts = bootstrap_advantage_cis(
        paired.v7_predictions,
        paired.patchtst_predictions,
        paired.truth,
        paired.patient_ids,
        tolerance,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    rows = []
    for metric in METRIC_ORDER:
        adv = point[metric]["advantage"]
        if adv is None:
            continue
        ci_lower = cis[metric]["ci_lower"]
        ci_upper = cis[metric]["ci_upper"]
        rows.append({
            "target": paired.target,
            "target_base": paired.target_base,
            "v7_model": paired.v7_model,
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "v7": point[metric]["v7"],
            "patchtst": point[metric]["patchtst"],
            "advantage": adv,
            "advantage_ci_lower": ci_lower,
            "advantage_ci_upper": ci_upper,
            "result": result_label(adv, ci_lower, ci_upper),
            "n_paired_predictions": int(paired.truth.size),
            "n_patients": int(np.unique(paired.patient_ids).size),
            "n_unmatched_v7": paired.n_unmatched_v7,
            "n_unmatched_patchtst": paired.n_unmatched_patchtst,
            "n_dropped_nonfinite": paired.n_dropped_nonfinite,
            "tolerance": tolerance,
            "bootstrap_replicates": int(n_bootstrap),
            "bootstrap_valid_replicates": cis[metric]["valid_bootstrap_replicates"],
            "bootstrap_invalid_replicates": cis[metric]["invalid_bootstrap_replicates"],
            "bootstrap_unit": "patient",
            "inference": "95% paired patient-cluster bootstrap CI",
        })
    validation = {
        "target": paired.target,
        "target_base": paired.target_base,
        "v7_model": paired.v7_model,
        "n_paired_predictions": int(paired.truth.size),
        "n_unique_patients": int(np.unique(paired.patient_ids).size),
        "n_unmatched_v7_predictions": paired.n_unmatched_v7,
        "n_unmatched_patchtst_predictions": paired.n_unmatched_patchtst,
        "n_dropped_nonfinite_predictions": paired.n_dropped_nonfinite,
        "same_sample_ids": True,
        "same_patient_ids": True,
        "same_ground_truth_values": True,
        "finite_predictions_both_models": True,
        "bootstrap_unit": "patient",
        "bootstrap_counts_shape": list(bootstrap_counts.shape),
        "duplicate_patient_selections_retained": bool(np.any(bootstrap_counts > 1.0)),
    }
    return rows, validation


def descriptive_row_from_arrays(
    *,
    target: str,
    target_base: str,
    model: str,
    row_type: str,
    predictions: np.ndarray,
    truth: np.ndarray,
    patient_ids: np.ndarray,
    prediction_file: Path,
    plot_path: Path | None,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    tolerance = BLAND_ALTMAN_TOLERANCES.get(target)
    point = bland_altman_stats(predictions, truth, tolerance)
    boot = bootstrap_cis(predictions, truth, patient_ids, tolerance, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "target": target,
        "target_base": target_base,
        "model": model,
        "row_type": row_type,
        "tolerance": tolerance,
        "n_predictions": int(predictions.size),
        "n_patients": int(np.unique(patient_ids).size),
        "prediction_file": str(prediction_file),
        "plot_path": str(plot_path) if plot_path is not None else None,
        "bootstrap_replicates": int(n_bootstrap),
        **point,
        **boot,
    }


def build_patchtst_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        patchtst_root=str(args.patchtst_root),
        target_path=str(args.target_path),
        splits_path=str(args.splits_path),
        waveform_dir=str(args.waveform_dir),
        channels=args.channels,
        seq_len=int(args.seq_len),
    )


def collect_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    analyses = load_existing_analyses(args.output_dir)
    patch_args = build_patchtst_args(args)
    descriptive_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    target_summaries: list[dict[str, Any]] = []
    for target_idx, analysis in enumerate(analyses):
        target = analysis["target"]
        target_base = analysis["target_base"]
        v7_path = v7_prediction_path(args.v7_root, target_base)
        if not v7_path.exists():
            raise FileNotFoundError(f"Missing GRU/V7 prediction file for {target_base}: {v7_path}")
        v7_set = load_v7_prediction_set(v7_path)
        patch_set = load_patchtst_prediction_set(target_base, patch_args)
        patch_path = args.patchtst_root / f"feature_{target}" / "test_predictions.npz"
        paired = align_prediction_sets(
            target=target,
            target_base=target_base,
            v7_model=V7_SOURCE_MODEL,
            v7=v7_set,
            patchtst=patch_set,
            v7_prediction_file=v7_path,
            patchtst_prediction_file=patch_path,
        )
        seed = int(args.seed) + target_idx * 1009
        descriptive_rows.append(descriptive_row_from_arrays(
            target=target,
            target_base=target_base,
            model=V7_MODEL_LABEL,
            row_type="v7_gru",
            predictions=paired.v7_predictions,
            truth=paired.truth,
            patient_ids=paired.patient_ids,
            prediction_file=v7_path,
            plot_path=None,
            n_bootstrap=args.bootstrap_replicates,
            seed=seed,
        ))
        descriptive_rows.append(descriptive_row_from_arrays(
            target=target,
            target_base=target_base,
            model=PATCHTST_MODEL_NAME,
            row_type="patchtst_v1_raw_waveform",
            predictions=paired.patchtst_predictions,
            truth=paired.truth,
            patient_ids=paired.patient_ids,
            prediction_file=patch_path,
            plot_path=None,
            n_bootstrap=args.bootstrap_replicates,
            seed=seed,
        ))
        rows, validation = comparison_rows_for_target(
            paired,
            n_bootstrap=args.bootstrap_replicates,
            seed=seed,
        )
        comparison_rows.extend(rows)
        validations.append(validation)
        evaluable = [row for row in rows if row["advantage"] is not None]
        favoring = [row for row in evaluable if float(row["advantage"]) > 0]
        significant = [row for row in evaluable if row["advantage_ci_lower"] is not None and float(row["advantage_ci_lower"]) > 0]
        target_summaries.append({
            "target": target,
            "target_base": target_base,
            "metrics_evaluated": int(len(evaluable)),
            "metrics_favoring_v7": int(len(favoring)),
            "significantly_favoring_v7": int(len(significant)),
            "all_evaluated_metrics_significantly_favor_v7": bool(len(evaluable) > 0 and len(significant) == len(evaluable)),
        })
    return descriptive_rows, comparison_rows, target_summaries, validations


def descriptive_markdown(rows: list[dict[str, Any]], *, n_bootstrap: int, seed: int) -> list[str]:
    lines = [
        "## Bland-Altman Agreement Summary",
        "",
        "Generated on `2026-09-01` for the v7 extracted-feature `gru` predictions, labeled `V7`, and the corresponding `PatchTST v1 raw waveform` baseline. Statistics use `difference = prediction - reference` and `average = (prediction + reference) / 2`, matching the plot-generation convention. Brackets are patient-cluster bootstrap percentile 95% CIs with patients resampled as clusters and all windows retained for sampled patients.",
        "",
        f"Bootstrap replicates: `{n_bootstrap}`. Random seed: `{seed}`.",
        "",
        "Each target has two rows: `V7` (`gru`) followed by `PatchTST v1 raw waveform`, evaluated on the same paired observations. Lower absolute bias is better, lower LoA half-width is better, and proportional-bias slope closer to zero is better. The descriptive table keeps the signed bias and signed proportional-bias slope; the paired comparison table below uses absolute magnitudes for bias and slope.",
        "",
        "Artifacts: `blandaltman_v7_extractedfeatures/` contains machine-readable agreement and paired comparison summaries `bland_altman_agreement_summary.{csv,json,md}`, `bland_altman_paired_significance.{csv,json,md}`, and `bland_altman_paired_validation.csv`.",
        "",
        "| Target | Model | Bias [95% CI] | LoA Half-Width [95% CI] | Proportional-Bias Slope [95% CI] | N Predictions | N Patients |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['target_base']}` | `{row['model']}` | "
            f"{format_estimate(row, 'bias')} | "
            f"{format_estimate(row, 'loa_half_width')} | "
            f"{format_estimate(row, 'proportional_bias_slope')} | "
            f"{row['n_predictions']} | {row['n_patients']} |"
        )
    return lines


def comparison_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "### Paired V7 vs PatchTST Significance",
        "",
        "| Target | Metric | V7 | PatchTST | V7 Advantage [95% CI] | Result |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["advantage"] is None:
            v7 = patch = adv = "NA"
        else:
            v7 = format_num(row["v7"])
            patch = format_num(row["patchtst"])
            adv = f"{format_num(row['advantage'], signed=True)} [{format_num(row['advantage_ci_lower'], signed=True)}, {format_num(row['advantage_ci_upper'], signed=True)}]"
        lines.append(f"| `{row['target_base']}` | {row['metric_label']} | {v7} | {patch} | {adv} | {row['result']} |")
    lines.extend([
        "",
        "**Paired model comparison:** Positive V7 Advantage values consistently favor V7. Advantage is defined as |Bias_PatchTST| - |Bias_V7| for absolute bias, LoA_PatchTST - LoA_V7 for LoA half-width, |Slope_PatchTST| - |Slope_V7| for proportional-bias magnitude, and WithinTolerance_V7 - WithinTolerance_PatchTST when a target-specific tolerance exists. Confidence intervals are 95% paired patient-cluster bootstrap intervals obtained by resampling patients with replacement and retaining all paired observations from each selected patient. A metric is considered significantly better for V7 when the entire 95% CI for V7 Advantage is above zero. Undefined optional metrics are omitted from this paired comparison table.",
    ])
    return lines


def summary_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "### Target-Level Summary",
        "",
        "| Target | Metrics Evaluated | Metrics Favoring V7 | Significantly Favoring V7 | All Evaluated Metrics Significantly Favor V7? |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        all_sig = "Yes" if row["all_evaluated_metrics_significantly_favor_v7"] else "No"
        lines.append(f"| `{row['target_base']}` | {row['metrics_evaluated']} | {row['metrics_favoring_v7']} | {row['significantly_favoring_v7']} | {all_sig} |")
    return lines


def section_markdown(descriptive_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]], target_summaries: list[dict[str, Any]], *, n_bootstrap: int, seed: int) -> str:
    lines = []
    lines.extend(descriptive_markdown(descriptive_rows, n_bootstrap=n_bootstrap, seed=seed))
    lines.extend(comparison_markdown(comparison_rows))
    lines.extend(summary_markdown(target_summaries))
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], *, exclude_fields: set[str] | None = None) -> None:
    exclude_fields = exclude_fields or set()
    fieldnames = sorted({key for row in rows for key in row.keys() if key not in exclude_fields})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def update_doc(doc_path: Path, section: str) -> None:
    text = doc_path.read_text()
    text = re.sub(r"## Bland-Altman Agreement Summary\n.*?(?=\n## |\Z)", section.rstrip() + "\n", text, flags=re.S)
    doc_path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add paired patient-cluster bootstrap V7 vs PatchTST Bland-Altman comparison tables.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--v7-root", type=Path, default=DEFAULT_V7_ROOT)
    parser.add_argument("--patchtst-root", type=Path, default=DEFAULT_PATCHTST_ROOT)
    parser.add_argument("--target-path", type=Path, default=DEFAULT_TARGET_PATH)
    parser.add_argument("--splits-path", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--waveform-dir", type=Path, default=DEFAULT_WAVEFORM_DIR)
    parser.add_argument("--channels", default=DEFAULT_CHANNELS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-doc-update", action="store_true")
    args = parser.parse_args()

    descriptive_rows, comparison_rows, target_summaries, validations = collect_rows(args)
    section = section_markdown(
        descriptive_rows,
        comparison_rows,
        target_summaries,
        n_bootstrap=args.bootstrap_replicates,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agreement_payload = {
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "tolerances": BLAND_ALTMAN_TOLERANCES,
        "v7_source_model": V7_SOURCE_MODEL,
        "rows": descriptive_rows,
    }
    significance_payload = {
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_unit": "patient",
        "primary_inference": "V7 Advantage with 95% paired patient-cluster bootstrap CI",
        "v7_source_model": V7_SOURCE_MODEL,
        "metric_definitions": {
            "bias": "mean(prediction - truth); comparison uses absolute value",
            "loa_half_width": "1.96 * sample SD(prediction - truth)",
            "proportional_bias_slope": "least-squares slope of difference on centered Bland-Altman average; comparison uses absolute value",
            "within_tolerance": "mean(abs(prediction - truth) <= target-specific tolerance); undefined when tolerance is absent",
            "v7_advantage": "positive values always favor V7",
        },
        "tolerances": BLAND_ALTMAN_TOLERANCES,
        "validation": validations,
        "rows": comparison_rows,
        "target_summaries": target_summaries,
    }
    (args.output_dir / "bland_altman_agreement_summary.json").write_text(json.dumps(agreement_payload, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "bland_altman_agreement_summary.md").write_text("\n".join(descriptive_markdown(descriptive_rows, n_bootstrap=args.bootstrap_replicates, seed=args.seed)) + "\n")
    write_csv(
        args.output_dir / "bland_altman_agreement_summary.csv",
        descriptive_rows,
        exclude_fields={"coverage_probability", "coverage_probability_lower", "coverage_probability_upper"},
    )
    (args.output_dir / "bland_altman_paired_significance.json").write_text(json.dumps(significance_payload, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "bland_altman_paired_significance.md").write_text("\n".join(comparison_markdown(comparison_rows) + summary_markdown(target_summaries)) + "\n")
    write_csv(args.output_dir / "bland_altman_paired_significance.csv", comparison_rows)
    write_csv(args.output_dir / "bland_altman_paired_validation.csv", validations)
    if not args.no_doc_update:
        update_doc(args.doc, section)
    print(json.dumps({
        "targets": len(target_summaries),
        "descriptive_rows": len(descriptive_rows),
        "comparison_rows": len(comparison_rows),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "doc_updated": not args.no_doc_update,
        "unmatched_v7_total": int(sum(v["n_unmatched_v7_predictions"] for v in validations)),
        "unmatched_patchtst_total": int(sum(v["n_unmatched_patchtst_predictions"] for v in validations)),
        "dropped_nonfinite_total": int(sum(v["n_dropped_nonfinite_predictions"] for v in validations)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
