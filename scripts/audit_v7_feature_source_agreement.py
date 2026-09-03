#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waveform_baselines.task_specs import FOCUSED_CORRELATION_NAMES, WAVEFORM_FEATURE_NAMES  # noqa: E402
from waveform_baselines.wf_features.cache import load_feature_cache  # noqa: E402


DEFAULT_V7_CACHE = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/"
    "waveformBaselines/featureExtraction/v7/vasopressor_free_waveform_features_v7"
)
DEFAULT_SOURCE_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/output_v2")
DEFAULT_OUTPUT_PREFIX = Path("outputs/feature_models/v7_source_agreement_2026-09-02")

TIME_DECIMALS = 6
TIME_SCALE = 10**TIME_DECIMALS


COMPARISONS = [
    {
        "source_name": "HR",
        "v7_feature": "ecg_hr_bpm",
        "source_kind": "X_stats",
        "notes": "Heart rate comparator; source implementation may use a different detector/smoothing path.",
    },
    {
        "source_name": "RR",
        "v7_feature": "resp_rate_bpm",
        "source_kind": "X_stats",
        "notes": "Respiratory-rate comparator; source and v7 RESP cycle acceptance may differ.",
    },
    {
        "source_name": "SBP",
        "v7_feature": "abp_sbp_median_mmhg",
        "source_kind": "X_stats",
        "notes": "Arterial systolic pressure comparator.",
    },
    {
        "source_name": "DBP",
        "v7_feature": "abp_dbp_median_mmhg",
        "source_kind": "X_stats",
        "notes": "Arterial diastolic pressure comparator.",
    },
    {
        "source_name": "MAP",
        "v7_feature": "abp_map_median_mmhg",
        "source_kind": "X_stats",
        "notes": "MAP comparator; v7 uses direct beat-mean MAP, not (SBP + 2*DBP)/3.",
    },
    {
        "source_name": "PP",
        "v7_feature": "abp_pulse_pressure_median_mmhg",
        "source_kind": "X_stats",
        "notes": "Pulse-pressure comparator.",
    },
    {
        "source_name": "ABP_area",
        "v7_feature": "abp_pulse_area_median",
        "source_kind": "X_stats",
        "notes": "ABP area comparator; exact beat boundary and baseline definitions may differ.",
    },
    {
        "source_name": "PLETH_amp",
        "v7_feature": "pleth_amplitude_median",
        "source_kind": "X_stats",
        "notes": "Native-scale PLETH amplitude comparator.",
    },
    {
        "source_name": "HRV_RMSSD",
        "v7_feature": "ecg_hrv_rmssd_s",
        "source_kind": "X_stats",
        "source_scale": 0.001,
        "notes": "RMSSD comparator; source is assumed milliseconds and converted to seconds when this improves unit scale.",
    },
    {
        "source_name": "dPdt_max",
        "v7_feature": "abp_dpdt_max_median",
        "source_kind": "X_stats",
        "notes": "Maximum positive ABP dP/dt comparator.",
    },
    {
        "source_name": "RESP_amp",
        "v7_feature": "resp_amplitude_median",
        "source_kind": "X_stats",
        "notes": "Native-scale RESP amplitude comparator.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frozen v7 extracted waveform features against same-window source feature arrays."
    )
    parser.add_argument("--v7-cache", type=Path, default=DEFAULT_V7_CACHE)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for smoke/debug runs.")
    return parser.parse_args()


def _quantize_time(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite timestamp encountered")
    return np.rint(arr * TIME_SCALE).astype(np.int64)


def _load_source_matrix(source_root: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    patient_ids = np.load(source_root / "patient_ids.npy", allow_pickle=True).astype(str)
    window_times = np.load(source_root / "window_times.npy").astype(np.float64)
    x_stats = np.load(source_root / "X_stats.npy", mmap_mode="r")
    if x_stats.shape[0] != len(patient_ids) or len(window_times) != len(patient_ids):
        raise ValueError("Source patient_ids/window_times/X_stats rows are not aligned")
    if x_stats.ndim == 2:
        x_values = np.asarray(x_stats, dtype=np.float32)
    elif x_stats.ndim == 3:
        feature_axes = [axis for axis in (1, 2) if x_stats.shape[axis] == len(WAVEFORM_FEATURE_NAMES)]
        if len(feature_axes) != 1:
            raise ValueError(f"Could not identify feature axis in X_stats shape {x_stats.shape}")
        aggregation_axis = 2 if feature_axes[0] == 1 else 1
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            x_values = np.nanmean(x_stats, axis=aggregation_axis, dtype=np.float64).astype(np.float32)
    else:
        raise ValueError(f"Unsupported X_stats shape: {x_stats.shape}")
    if x_values.shape != (len(patient_ids), len(WAVEFORM_FEATURE_NAMES)):
        raise ValueError(f"Unexpected aggregated X_stats shape: {x_values.shape}")
    frame = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "qtime": _quantize_time(window_times),
            "source_row": np.arange(len(patient_ids), dtype=np.int64),
        }
    )
    if frame.duplicated(["patient_id", "qtime"]).any():
        examples = frame.loc[frame.duplicated(["patient_id", "qtime"], keep=False)].head(10).to_dict("records")
        raise ValueError(f"Duplicate source rows by patient/time, examples={examples}")
    return frame, x_values, {name: idx for idx, name in enumerate(WAVEFORM_FEATURE_NAMES)}


def _history_summaries(values: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    valid = mask & np.isfinite(values)
    arr = np.where(valid, values, np.nan).astype(np.float64, copy=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        return {
            "mean": np.nanmean(arr, axis=1),
            "median": np.nanmedian(arr, axis=1),
            "last": arr[:, -1, :],
        }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson(ra, rb)


def _metrics(
    source: np.ndarray,
    extracted: np.ndarray,
    source_name: str,
    v7_feature: str,
    summary: str,
    notes: str,
) -> dict[str, object]:
    ok = np.isfinite(source) & np.isfinite(extracted)
    n = int(ok.sum())
    row: dict[str, object] = {
        "source_name": source_name,
        "v7_feature": v7_feature,
        "v7_summary": summary,
        "n_common": n,
        "source_valid_fraction": float(np.isfinite(source).mean()) if source.size else 0.0,
        "v7_valid_fraction": float(np.isfinite(extracted).mean()) if extracted.size else 0.0,
        "notes": notes,
    }
    if n == 0:
        return row
    src = source[ok].astype(np.float64, copy=False)
    ext = extracted[ok].astype(np.float64, copy=False)
    diff = ext - src
    abs_diff = np.abs(diff)
    row.update(
        {
            "source_median": float(np.median(src)),
            "v7_median": float(np.median(ext)),
            "bias_v7_minus_source": float(np.mean(diff)),
            "median_diff_v7_minus_source": float(np.median(diff)),
            "mae": float(np.mean(abs_diff)),
            "rmse": float(np.sqrt(np.mean(diff * diff))),
            "p95_abs_error": float(np.percentile(abs_diff, 95.0)),
            "pearson_r": _pearson(src, ext),
            "spearman_r": _spearman(src, ext),
        }
    )
    return row


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    def fmt(value: object) -> str:
        if isinstance(value, float):
            if not np.isfinite(value):
                return ""
            return f"{value:.4g}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cache = load_feature_cache(args.v7_cache, require_success=True)
    n_rows = cache.values.shape[0] if args.max_rows is None else min(int(args.max_rows), cache.values.shape[0])
    source_frame, source_values_all, source_index = _load_source_matrix(args.source_root)

    cache_frame = pd.DataFrame(
        {
            "cache_row": np.arange(n_rows, dtype=np.int64),
            "patient_id": np.asarray(cache.patient_ids[:n_rows], dtype=str),
            "qtime": _quantize_time(np.asarray(cache.anchor_times[:n_rows], dtype=np.float64)),
        }
    )
    aligned = cache_frame.merge(source_frame, on=["patient_id", "qtime"], how="left", validate="one_to_one")
    matched = aligned["source_row"].notna().to_numpy()
    if not matched.any():
        raise ValueError("No v7 cache rows matched source rows by patient_id and anchor_time/window_time")
    source_rows = aligned.loc[matched, "source_row"].to_numpy(dtype=np.int64)
    cache_rows = aligned.loc[matched, "cache_row"].to_numpy(dtype=np.int64)

    summaries = _history_summaries(np.asarray(cache.values[:n_rows]), np.asarray(cache.mask[:n_rows], dtype=bool))
    feature_index = {name: idx for idx, name in enumerate(cache.feature_names)}
    rows: list[dict[str, object]] = []
    for comparison in COMPARISONS:
        source_name = comparison["source_name"]
        v7_feature = comparison["v7_feature"]
        if source_name not in source_index or v7_feature not in feature_index:
            continue
        source = source_values_all[source_rows, source_index[source_name]].astype(np.float64, copy=False)
        scale = float(comparison.get("source_scale", 1.0))
        source = source * scale
        v7_idx = feature_index[v7_feature]
        for summary_name, summary_values in summaries.items():
            extracted = summary_values[cache_rows, v7_idx]
            rows.append(
                _metrics(
                    source=source,
                    extracted=extracted,
                    source_name=source_name,
                    v7_feature=v7_feature,
                    summary=summary_name,
                    notes=str(comparison["notes"]),
                )
            )

    best_rows = []
    for source_name, group in pd.DataFrame(rows).groupby("source_name", sort=False):
        ranked = group.sort_values(["pearson_r", "mae"], ascending=[False, True], na_position="last")
        best_rows.append(ranked.iloc[0].to_dict())

    report = {
        "v7_cache": str(args.v7_cache),
        "source_root": str(args.source_root),
        "comparison_scope": {
            "v7_rows_checked": int(n_rows),
            "source_rows": int(len(source_frame)),
            "matched_rows": int(matched.sum()),
            "match_rate": float(matched.mean()),
            "time_quantization_decimals": TIME_DECIMALS,
            "v7_shape": list(cache.values.shape),
            "source_feature_names": list(WAVEFORM_FEATURE_NAMES),
            "source_correlation_names": list(FOCUSED_CORRELATION_NAMES),
        },
        "comparisons": rows,
        "best_summary_by_source_name": best_rows,
        "interpretation": (
            "These are same-window comparators from historical source feature arrays, not adjudicated clinical ground truth. "
            "Disagreement can reflect different detector algorithms, tokenization/aggregation, or source feature definitions."
        ),
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2))
    table = pd.DataFrame(rows)
    table.to_csv(csv_path, index=False)
    best_table = pd.DataFrame(best_rows)
    md_path.write_text(
        "# V7 Source Agreement Audit\n\n"
        f"- v7 cache: `{args.v7_cache}`\n"
        f"- source root: `{args.source_root}`\n"
        f"- matched rows: `{int(matched.sum())} / {n_rows}` (`{matched.mean():.6f}`)\n\n"
        "## Best V7 Summary Per Source Comparator\n\n"
        + _markdown_table(
            best_table,
            [
                "source_name",
                "v7_feature",
                "v7_summary",
                "n_common",
                "pearson_r",
                "spearman_r",
                "bias_v7_minus_source",
                "mae",
                "rmse",
                "p95_abs_error",
            ],
        )
        + "\n\nThese are source-feature agreement checks, not adjudicated clinical ground truth.\n"
    )
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(md_path),
                "matched_rows": int(matched.sum()),
                "match_rate": float(matched.mean()),
                "comparisons": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
