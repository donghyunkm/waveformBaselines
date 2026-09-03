#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waveform_baselines.task_specs import (  # noqa: E402
    DEFAULT_FEATURE_TASK,
    FOCUSED_CORRELATION_NAMES,
    WAVEFORM_FEATURE_NAMES,
    FeatureRegressionTaskSpec,
)
from waveform_baselines.target_builders import save_target_bundle  # noqa: E402

DEFAULT_FULL_DATA_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/physionet_restricted/mimic_derived_data/data_m3_120s_prediction")
DEFAULT_CACHE_DIR = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v7/full_data_vasopressor_free_waveform_features_v7")
DEFAULT_OUTPUT = Path("outputs/targets/feature_targets_gap_full_data.npz")

TIME_DECIMALS = 6
TIME_SCALE = 10**TIME_DECIMALS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build segment-aware full-data regression target bundles aligned to the full-data v7 feature cache."
    )
    parser.add_argument("--full-data-root", type=Path, default=DEFAULT_FULL_DATA_ROOT)
    parser.add_argument("--feature-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_FEATURE_TASK.horizons_min))
    parser.add_argument("--feature-horizon-mode", choices=["center", "gap"], default="gap")
    parser.add_argument("--min-source-match-rate", type=float, default=None)
    return parser.parse_args()


def load_string_vector(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=True))
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D string array at {path}, got shape {values.shape}")
    return values.astype(str)


def validate_identifier_array(values: pd.Series | np.ndarray, label: str) -> np.ndarray:
    series = pd.Series(values)
    if series.isna().any():
        rows = series.index[series.isna()][:10].tolist()
        raise ValueError(f"{label} contains missing values, row examples={rows}")
    out = series.astype(str).str.strip()
    invalid = out.eq("") | out.str.lower().isin({"nan", "none", "null"})
    if invalid.any():
        examples = out[invalid].head(10).tolist()
        raise ValueError(f"{label} contains invalid identifiers, examples={examples}")
    return out.to_numpy(dtype=str)


def parse_integral_ids(values: pd.Series | np.ndarray, label: str) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} contains non-finite IDs")
    integral = np.equal(numeric, np.floor(numeric))
    if not integral.all():
        examples = numeric[~integral][:10].tolist()
        raise ValueError(f"{label} contains non-integral IDs, examples={examples}")
    info = np.iinfo(np.int64)
    if np.any(numeric < info.min) or np.any(numeric > info.max):
        raise ValueError(f"{label} contains values outside int64 range")
    return numeric.astype(np.int64)


def load_identifier_vector(path: Path, label: str) -> np.ndarray:
    return validate_identifier_array(load_string_vector(path), label)


def quantize_time_seconds(value: float) -> int:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite timestamp: {value}")
    return int(np.rint(value * TIME_SCALE))


def quantize_time_array(values: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D timestamps for {label}, got shape {arr.shape}")
    bad = ~np.isfinite(arr)
    if np.any(bad):
        examples = np.flatnonzero(bad)[:10].tolist()
        raise ValueError(f"Non-finite timestamps in {label}, row examples={examples}")
    return np.rint(arr * TIME_SCALE).astype(np.int64)


def validate_horizons(values: Iterable[int]) -> tuple[int, ...]:
    horizons = tuple(int(h) for h in values)
    if not horizons:
        raise ValueError("At least one horizon is required")
    if len(set(horizons)) != len(horizons):
        raise ValueError(f"Horizons must be unique, got {horizons}")
    if any(h < 0 for h in horizons):
        raise ValueError(f"Horizons must be non-negative, got {horizons}")
    return horizons


def validate_min_source_match_rate(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"--min-source-match-rate must be in [0, 1], got {value}")
    return value


def horizon_offset_seconds(spec: FeatureRegressionTaskSpec, horizon_min: int) -> float:
    if spec.horizon_mode == "center":
        return float(horizon_min * 60.0)
    if spec.horizon_mode == "gap":
        return float(spec.input_window_minutes * 60.0 + horizon_min * 60.0)
    raise ValueError(f"Unsupported feature horizon mode: {spec.horizon_mode}")


def aggregate_x_stats(x_stats: np.ndarray, expected_rows: int, expected_features: int) -> np.ndarray:
    if x_stats.ndim not in {2, 3}:
        raise ValueError(f"Expected 2D or 3D X_stats, got {x_stats.shape}")
    if x_stats.shape[0] != expected_rows:
        raise ValueError(f"X_stats rows {x_stats.shape[0]} != expected rows {expected_rows}")
    if x_stats.ndim == 2:
        if x_stats.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} X_stats features, got shape {x_stats.shape}")
        return np.asarray(x_stats, dtype=np.float32)

    matching_axes = [axis for axis in (1, 2) if x_stats.shape[axis] == expected_features]
    if len(matching_axes) != 1:
        raise ValueError(
            "Could not uniquely identify the feature axis in "
            f"X_stats shape {x_stats.shape}; expected feature count={expected_features}"
        )
    feature_axis = matching_axes[0]
    aggregation_axis = 2 if feature_axis == 1 else 1
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return np.nanmean(x_stats, axis=aggregation_axis, dtype=np.float64).astype(np.float32)


def _require_unique_int(values: np.ndarray, label: str) -> None:
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    duplicate_values = unique[counts > 1]
    if duplicate_values.size:
        raise ValueError(f"{label} contains duplicate values, examples={duplicate_values[:10].tolist()}")


def validate_feature_cache(cache_dir: Path) -> dict[str, object]:
    required_files = ["values.npy", "mask.npy", "patient_ids.npy", "anchor_times.npy", "anchor_ids.npy", "split_labels.npy"]
    for name in required_files:
        path = cache_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing feature-cache file: {path}")
    cache_values = np.load(cache_dir / "values.npy", mmap_mode="r")
    cache_mask = np.load(cache_dir / "mask.npy", mmap_mode="r")
    if cache_values.ndim != 3:
        raise ValueError(f"Feature-cache values.npy must have shape (N, T, F), got {cache_values.shape}")
    if cache_mask.shape != cache_values.shape:
        raise ValueError(f"Feature-cache values/mask shape mismatch: {cache_values.shape} vs {cache_mask.shape}")

    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing feature-cache metadata: {metadata_path}")
    cache_metadata = json.loads(metadata_path.read_text())
    if "n_samples" in cache_metadata and int(cache_metadata["n_samples"]) != cache_values.shape[0]:
        raise ValueError("Feature-cache metadata n_samples does not match values.npy")
    if "n_feature_windows" in cache_metadata and int(cache_metadata["n_feature_windows"]) != cache_values.shape[1]:
        raise ValueError("Feature-cache metadata n_feature_windows does not match values.npy")
    feature_names = cache_metadata.get("feature_names")
    if feature_names is not None and len(feature_names) != cache_values.shape[2]:
        raise ValueError("Feature-cache metadata feature_names length does not match values.npy")
    if cache_metadata.get("is_merged_cache", False) and not (cache_dir / "_SUCCESS").exists():
        raise FileNotFoundError(f"Merged feature cache is missing completion marker: {cache_dir / '_SUCCESS'}")
    return {
        "cache_values_shape": list(cache_values.shape),
        "cache_mask_shape": list(cache_mask.shape),
        "cache_metadata_path": str(metadata_path),
        "cache_metadata_n_samples": cache_metadata.get("n_samples"),
        "cache_metadata_n_feature_windows": cache_metadata.get("n_feature_windows"),
        "cache_metadata_feature_names_count": None if feature_names is None else len(feature_names),
        "merged_cache_success_marker_required": bool(cache_metadata.get("is_merged_cache", False)),
    }


def _segment_start_lookup(full_data_root: Path) -> dict[tuple[str, str], float]:
    path = full_data_root / "segment_metadata.json"
    if not path.exists():
        return {}
    records = json.loads(path.read_text())
    lookup: dict[tuple[str, str], float] = {}
    for record in records:
        key = (str(record["patient_id"]), str(record["seg_name"]))
        value = float(record["seg_start_secs"])
        if key in lookup and not np.isclose(lookup[key], value, atol=1e-6, rtol=0.0):
            raise ValueError(f"Conflicting segment starts for {key} in {path}")
        lookup[key] = value
    return lookup


def _time_summary(values: np.ndarray, label: str) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    q = quantize_time_array(arr, label)
    unique_q = np.unique(q)
    if unique_q.size > 1:
        diffs = np.diff(np.sort(unique_q)).astype(np.float64) / TIME_SCALE
        spacing_values, spacing_counts = np.unique(np.round(diffs, TIME_DECIMALS), return_counts=True)
        order = np.argsort(spacing_counts)[::-1][:10]
        common_spacing = [
            {"seconds": float(spacing_values[idx]), "count": int(spacing_counts[idx])}
            for idx in order
        ]
    else:
        common_spacing = []
    return {
        "label": label,
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "unique_times": int(unique_q.size),
        "common_spacing": common_spacing,
    }


def _segment_time_summary(patient_ids: np.ndarray, seg_names: np.ndarray, values: np.ndarray, label: str) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    q = quantize_time_array(arr, label)
    frame = pd.DataFrame({"patient_id": patient_ids, "seg_name": seg_names, "qtime": q})
    spacing_counts: dict[float, int] = {}
    nonpositive = 0
    n_segments = 0
    for _, group in frame.groupby(["patient_id", "seg_name"], sort=False):
        n_segments += 1
        sorted_q = np.sort(group["qtime"].to_numpy(dtype=np.int64))
        if sorted_q.size < 2:
            continue
        diffs = np.diff(sorted_q)
        nonpositive += int(np.sum(diffs <= 0))
        positive_seconds = np.round(diffs[diffs > 0].astype(np.float64) / TIME_SCALE, TIME_DECIMALS)
        values_unique, counts = np.unique(positive_seconds, return_counts=True)
        for spacing, count in zip(values_unique.tolist(), counts.tolist()):
            spacing_counts[float(spacing)] = spacing_counts.get(float(spacing), 0) + int(count)
    common_spacing = [
        {"seconds": float(spacing), "count": int(count)}
        for spacing, count in sorted(spacing_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    ]
    return {
        "label": label,
        "number_of_segments": int(n_segments),
        "number_of_source_rows": int(arr.size),
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "unique_times": int(np.unique(q).size),
        "most_common_within_segment_spacing": common_spacing,
        "nonpositive_within_segment_differences": int(nonpositive),
    }


def load_cache_anchors(cache_dir: Path, full_data_root: Path, input_window_minutes: int) -> tuple[pd.DataFrame, dict[str, object]]:
    cache_validation = validate_feature_cache(cache_dir)
    anchors_path = cache_dir / "anchors.csv"
    if not anchors_path.exists():
        raise FileNotFoundError(f"Missing full-data feature-cache anchors: {anchors_path}")
    anchors = pd.read_csv(anchors_path)
    required = {"anchor_id", "patient_id", "segment_id", "seg_name", "window_time", "split_label"}
    missing = required.difference(anchors.columns)
    if missing:
        raise ValueError(f"Feature-cache anchors missing columns: {sorted(missing)}")
    anchors = anchors.copy()
    anchors["patient_id"] = validate_identifier_array(anchors["patient_id"], "anchors.csv patient_id")
    anchors["segment_id"] = validate_identifier_array(anchors["segment_id"], "anchors.csv segment_id")
    anchors["seg_name"] = validate_identifier_array(anchors["seg_name"], "anchors.csv seg_name")
    anchors["split_label"] = validate_identifier_array(anchors["split_label"], "anchors.csv split_label")
    anchors["anchor_id"] = parse_integral_ids(anchors["anchor_id"], "anchors.csv anchor_id")

    values = np.load(cache_dir / "values.npy", mmap_mode="r")
    n_cache_rows = int(values.shape[0])
    if len(anchors) != n_cache_rows:
        raise ValueError(f"anchors.csv rows {len(anchors)} != feature cache rows {n_cache_rows}")

    cache_patient_ids = load_identifier_vector(cache_dir / "patient_ids.npy", "cache patient_ids.npy")
    cache_anchor_times = np.asarray(np.load(cache_dir / "anchor_times.npy", mmap_mode="r"), dtype=np.float64)
    cache_anchor_ids = parse_integral_ids(np.load(cache_dir / "anchor_ids.npy", mmap_mode="r"), "cache anchor_ids.npy")
    cache_split_labels = load_identifier_vector(cache_dir / "split_labels.npy", "cache split_labels.npy")

    if not (len(cache_patient_ids) == len(cache_anchor_times) == len(cache_anchor_ids) == len(cache_split_labels) == n_cache_rows):
        raise ValueError("Feature-cache metadata arrays are not all row-aligned to values.npy")
    if not np.array_equal(cache_patient_ids, anchors["patient_id"].astype(str).to_numpy()):
        raise ValueError("cache patient_ids.npy does not match anchors.csv patient_id row by row")
    if not np.array_equal(cache_anchor_ids, anchors["anchor_id"].to_numpy()):
        raise ValueError("cache anchor_ids.npy does not match anchors.csv anchor_id row by row")
    if not np.array_equal(cache_split_labels, anchors["split_label"].astype(str).to_numpy()):
        raise ValueError("cache split_labels.npy does not match anchors.csv split_label row by row")

    _require_unique_int(cache_anchor_ids, "anchor_ids.npy")
    _require_unique_int(anchors["anchor_id"].to_numpy(), "anchors.csv anchor_id")
    csv_window_q = quantize_time_array(anchors["window_time"].to_numpy(dtype=np.float64), "anchors.csv window_time")
    cache_anchor_q = quantize_time_array(cache_anchor_times, "feature cache anchor_times.npy")

    segment_starts = _segment_start_lookup(full_data_root)
    if segment_starts:
        starts = np.asarray(
            [segment_starts.get((str(pid), str(seg)), np.nan) for pid, seg in zip(anchors["patient_id"], anchors["seg_name"])],
            dtype=np.float64,
        )
    else:
        starts = np.full(len(anchors), np.nan, dtype=np.float64)

    matches_window_time = bool(np.array_equal(cache_anchor_q, csv_window_q))
    matches_absolute_time = False
    missing_segment_starts = int(np.isnan(starts).sum())
    if missing_segment_starts == 0:
        absolute_times = starts + anchors["window_time"].to_numpy(dtype=np.float64)
        matches_absolute_time = bool(
            np.array_equal(cache_anchor_q, quantize_time_array(absolute_times, "absolute anchor times from segment metadata"))
        )

    if not (matches_window_time or matches_absolute_time):
        examples = anchors[["anchor_id", "patient_id", "seg_name", "window_time"]].head(10).to_dict("records")
        raise ValueError(
            "feature-cache anchor_times.npy do not match anchors.csv window_time or segment_metadata absolute times "
            f"after {TIME_DECIMALS}-decimal quantization; examples={examples}"
        )

    anchors["anchor_time"] = cache_anchor_times
    anchors["source_lookup_time"] = anchors["window_time"].to_numpy(dtype=np.float64)
    half_window = input_window_minutes * 60.0 / 2.0
    anchors["input_start_time"] = anchors["anchor_time"].to_numpy(dtype=np.float64) - half_window
    anchors["input_end_time"] = anchors["anchor_time"].to_numpy(dtype=np.float64) + half_window

    composite_keys = pd.MultiIndex.from_arrays(
        [
            anchors["patient_id"].astype(str).to_numpy(),
            anchors["seg_name"].astype(str).to_numpy(),
            cache_anchor_q,
        ]
    )
    if composite_keys.duplicated().any():
        examples = anchors.loc[composite_keys.duplicated(), ["anchor_id", "patient_id", "seg_name", "anchor_time"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate anchor composite identities by (patient_id, seg_name, anchor_time), examples={examples}")

    diagnostics = {
        **cache_validation,
        "n_cache_rows": n_cache_rows,
        "anchor_id_unique": True,
        "row_alignment": {
            "patient_ids": "match",
            "anchor_ids": "match",
            "split_labels": "match",
        },
        "canonical_anchor_time_relation": (
            "equals_anchors_csv_window_time"
            if matches_window_time
            else "equals_segment_start_plus_anchors_csv_window_time"
        ),
        "canonical_anchor_time_basis_verified": False,
        "anchor_times_match_window_time": matches_window_time,
        "anchor_times_match_segment_metadata_absolute_time": matches_absolute_time,
        "missing_segment_starts": missing_segment_starts,
        "source_lookup_time_field": "anchors.csv:window_time",
        "source_lookup_time_relation": "checked_against_full_data_root/window_times.npy_by_current_source_identity_audit",
    }
    return anchors, diagnostics


def _load_json_list(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    values = json.loads(path.read_text())
    if isinstance(values, dict) and "features" in values:
        values = values["features"]
    if not isinstance(values, list):
        raise ValueError(f"Expected list metadata in {path}")
    return [str(value) for value in values]


def _load_source_name_metadata(full_data_root: Path, candidates: list[tuple[str, tuple[str, ...]]]) -> tuple[list[str] | None, str | None]:
    for filename, keys in candidates:
        path = full_data_root / filename
        if not path.exists():
            continue
        values = json.loads(path.read_text())
        for key in keys:
            candidate = values.get(key) if isinstance(values, dict) else values
            if isinstance(candidate, list):
                return [str(value) for value in candidate], f"{filename}:{key}" if isinstance(values, dict) else filename
    return None, None


def validate_feature_order(full_data_root: Path, spec: FeatureRegressionTaskSpec) -> dict[str, object]:
    if spec.aggregation != "mean":
        raise ValueError(
            "This builder currently implements only mean aggregation; "
            f"received aggregation={spec.aggregation!r}"
        )
    if list(spec.feature_names) != list(WAVEFORM_FEATURE_NAMES):
        raise ValueError("spec.feature_names does not match repository WAVEFORM_FEATURE_NAMES order")
    if list(spec.correlation_names) != list(FOCUSED_CORRELATION_NAMES):
        raise ValueError("spec.correlation_names does not match repository FOCUSED_CORRELATION_NAMES order")
    x_names, x_source = _load_source_name_metadata(
        full_data_root,
        [
            ("X_stats_feature_names.json", ("features", "feature_names", "X_stats_feature_names")),
            ("waveform_feature_names.json", ("features", "feature_names", "X_stats_feature_names")),
            ("feature_names.json", ("features", "feature_names", "X_stats_feature_names")),
            ("metadata.json", ("X_stats_feature_names", "x_stats_feature_names", "feature_names")),
        ],
    )
    corr_metadata, corr_source = _load_source_name_metadata(
        full_data_root,
        [
            ("corr_features_focused_names.json", ("features", "correlation_names", "corr_features_focused_names")),
            ("correlation_names.json", ("features", "correlation_names", "corr_features_focused_names")),
            ("metadata.json", ("corr_features_focused_names", "correlation_names")),
        ],
    )
    if x_names is not None and x_names != list(spec.feature_names):
        raise ValueError(
            "Source X_stats feature-name order does not match spec.feature_names: "
            f"metadata={x_names}, spec={list(spec.feature_names)}"
        )
    if corr_metadata is not None and corr_metadata != list(spec.correlation_names):
        raise ValueError(
            "corr_features_focused_names.json order does not match spec.correlation_names: "
            f"metadata={corr_metadata}, spec={list(spec.correlation_names)}"
        )
    return {
        "x_stats_feature_order_reference": x_source or "historical extraction implementation",
        "correlation_order_reference": corr_source or "waveform_baselines.task_specs.FOCUSED_CORRELATION_NAMES",
        "x_stats_feature_order_verified": x_names is not None,
        "x_stats_feature_order_status": (
            f"Verified against source metadata {x_source}"
            if x_names is not None
            else "Assumed from the historical extraction implementation; no source-side X_stats feature-name metadata was found"
        ),
        "correlation_order_verified": corr_metadata is not None,
        "correlation_order_status": (
            f"Verified against source metadata {corr_source}"
            if corr_metadata is not None
            else "Assumed from repository constants; no source-side correlation feature-name metadata was found"
        ),
    }


def load_source_values(full_data_root: Path, spec: FeatureRegressionTaskSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    order_diagnostics = validate_feature_order(full_data_root, spec)
    patient_ids = load_identifier_vector(full_data_root / "patient_ids.npy", "source patient_ids.npy")
    seg_names = load_identifier_vector(full_data_root / "seg_names.npy", "source seg_names.npy")
    window_times = np.asarray(np.load(full_data_root / "window_times.npy", mmap_mode="r"), dtype=np.float64)
    corr = np.load(full_data_root / "corr_features_focused.npy", mmap_mode="r")
    x_stats = np.load(full_data_root / "X_stats.npy", mmap_mode="r")

    expected_rows = len(patient_ids)
    if corr.ndim != 2:
        raise ValueError(f"Expected 2D corr_features_focused, got shape {corr.shape}")
    if not (len(seg_names) == len(window_times) == corr.shape[0] == x_stats.shape[0] == expected_rows):
        raise ValueError("Full-data source arrays are not row-aligned")
    if corr.shape != (expected_rows, len(spec.correlation_names)):
        raise ValueError(f"Expected corr_features_focused shape {(expected_rows, len(spec.correlation_names))}, got {corr.shape}")

    x_values = aggregate_x_stats(x_stats, expected_rows, len(spec.feature_names))
    values = np.concatenate([x_values, np.asarray(corr, dtype=np.float32)], axis=1)
    if len(spec.base_target_names) != len(spec.feature_names) + len(spec.correlation_names):
        raise ValueError("FeatureRegressionTaskSpec base_target_names invariant failed")
    if values.ndim != 2 or values.shape != (expected_rows, len(spec.base_target_names)):
        raise ValueError(f"Final source target matrix shape {values.shape} != {(expected_rows, len(spec.base_target_names))}")

    diagnostics = {
        **order_diagnostics,
        "source_array_paths": {
            "patient_ids": str(full_data_root / "patient_ids.npy"),
            "seg_names": str(full_data_root / "seg_names.npy"),
            "window_times": str(full_data_root / "window_times.npy"),
            "X_stats": str(full_data_root / "X_stats.npy"),
            "corr_features_focused": str(full_data_root / "corr_features_focused.npy"),
        },
        "source_array_shapes": {
            "patient_ids": [int(expected_rows)],
            "seg_names": [int(len(seg_names))],
            "window_times": list(window_times.shape),
            "X_stats": list(x_stats.shape),
            "corr_features_focused": list(corr.shape),
            "source_target_values": list(values.shape),
        },
        "source_time_summary": _segment_time_summary(patient_ids, seg_names, window_times, "source window_times.npy"),
    }
    return patient_ids, seg_names, window_times, values, diagnostics


def _duplicate_source_error(patient_ids: np.ndarray, seg_names: np.ndarray, window_times: np.ndarray, duplicate_groups: list[np.ndarray]) -> ValueError:
    examples = []
    for rows in duplicate_groups[:10]:
        idx0 = int(rows[0])
        examples.append(
            {
                "patient_id": str(patient_ids[idx0]),
                "seg_name": str(seg_names[idx0]),
                "window_time": float(window_times[idx0]),
                "duplicate_row_indices": [int(idx) for idx in rows.tolist()],
            }
        )
    return ValueError(f"Duplicate source identities by (patient_id, seg_name, quantized_window_time), examples={examples}")


def _build_source_segment_index(
    patient_ids: np.ndarray,
    seg_names: np.ndarray,
    window_times: np.ndarray,
) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    source_q = quantize_time_array(window_times, "source window_times.npy")
    frame = pd.DataFrame({"patient_id": patient_ids, "seg_name": seg_names, "qtime": source_q})
    grouped: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    duplicate_groups: list[np.ndarray] = []
    for (patient_id, seg_name), group in frame.groupby(["patient_id", "seg_name"], sort=False):
        rows = group.index.to_numpy(dtype=np.int64)
        order = np.argsort(source_q[rows], kind="stable")
        sorted_rows = rows[order]
        sorted_q = source_q[sorted_rows]
        duplicate_q = np.unique(sorted_q[:-1][sorted_q[1:] == sorted_q[:-1]])
        for qtime in duplicate_q:
            duplicate_groups.append(sorted_rows[sorted_q == qtime])
        grouped[(str(patient_id), str(seg_name))] = (sorted_q, sorted_rows)
    if duplicate_groups:
        raise _duplicate_source_error(patient_ids, seg_names, window_times, duplicate_groups)
    return grouped, {
        "source_identity": "(patient_id, seg_name, quantized_window_time)",
        "source_rows_unique": True,
        "n_source_segments": int(len(grouped)),
        "time_quantization_decimals": TIME_DECIMALS,
    }


def validate_anchor_source_identities(
    anchors: pd.DataFrame,
    source_segments: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    anchor_q = quantize_time_array(anchors["source_lookup_time"].to_numpy(dtype=np.float64), "anchor source_lookup_time")
    matched = np.zeros(len(anchors), dtype=bool)
    anchor_frame = pd.DataFrame(
        {
            "row_idx": np.arange(len(anchors), dtype=np.int64),
            "patient_id": anchors["patient_id"].astype(str).to_numpy(),
            "seg_name": anchors["seg_name"].astype(str).to_numpy(),
        }
    )
    grouped = anchor_frame.groupby(["patient_id", "seg_name"], sort=False)
    for segment_key, group in grouped:
        source_entry = source_segments.get((str(segment_key[0]), str(segment_key[1])))
        if source_entry is None:
            continue
        source_q, _ = source_entry
        rows = group["row_idx"].to_numpy(dtype=np.int64)
        target_q = anchor_q[rows]
        positions = np.searchsorted(source_q, target_q)
        in_bounds = positions < source_q.size
        exact = np.zeros(rows.size, dtype=bool)
        exact[in_bounds] = source_q[positions[in_bounds]] == target_q[in_bounds]
        matched[rows] = exact
    if not matched.all():
        bad_rows = np.flatnonzero(~matched)[:10]
        examples = anchors.iloc[bad_rows][
            ["anchor_id", "patient_id", "segment_id", "seg_name", "source_lookup_time"]
        ].to_dict("records")
        raise ValueError(
            "Some cache anchors do not have an exact current source row; "
            f"matched={int(matched.sum())}/{len(matched)}, examples={examples}"
        )
    return {
        "current_source_identity_matches": int(matched.sum()),
        "current_source_identity_total": int(len(matched)),
        "current_source_identity_match_rate": float(matched.mean()) if len(matched) else 0.0,
    }


def build_targets(
    anchors: pd.DataFrame,
    full_data_root: Path,
    spec: FeatureRegressionTaskSpec,
    min_source_match_rate: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    patient_ids, seg_names, window_times, values, source_diagnostics = load_source_values(full_data_root, spec)
    source_segments, duplicate_diagnostics = _build_source_segment_index(patient_ids, seg_names, window_times)
    current_identity_diagnostics = validate_anchor_source_identities(anchors, source_segments)

    n_rows = len(anchors)
    base_dim = len(spec.base_target_names)
    targets = np.full((n_rows, base_dim * len(spec.horizons_min)), np.nan, dtype=np.float32)
    mask = np.zeros_like(targets, dtype=bool)
    anchor_lookup_times = anchors["source_lookup_time"].to_numpy(dtype=np.float64)
    anchor_lookup_q = quantize_time_array(anchor_lookup_times, "anchor source_lookup_time")
    diagnostics = {
        "mode": "segment_aware_full_data_regression",
        "horizon_mode": spec.horizon_mode,
        "number_of_anchors": int(n_rows),
        "number_of_source_rows": int(values.shape[0]),
        "duplicate_key_checks": duplicate_diagnostics,
        "current_source_identity_audit": current_identity_diagnostics,
        "time_summaries": [
            source_diagnostics["source_time_summary"],
            _time_summary(anchor_lookup_times, "anchor source_lookup_time"),
            _time_summary(anchors["anchor_time"].to_numpy(dtype=np.float64), "canonical cache anchor_time"),
        ],
        "per_horizon": {},
    }
    diagnostics.update(source_diagnostics)

    anchor_frame = pd.DataFrame(
        {
            "row_idx": np.arange(n_rows, dtype=np.int64),
            "patient_id": anchors["patient_id"].astype(str).to_numpy(),
            "seg_name": anchors["seg_name"].astype(str).to_numpy(),
        }
    )
    anchor_groups = anchor_frame.groupby(["patient_id", "seg_name"], sort=False)

    for horizon_idx, horizon_min in enumerate(spec.horizons_min):
        offset_seconds = horizon_offset_seconds(spec, horizon_min)
        offset_q = quantize_time_seconds(offset_seconds)
        col_start = horizon_idx * base_dim
        col_end = col_start + base_dim
        matched_rows = np.zeros(n_rows, dtype=bool)

        for segment_key, group in anchor_groups:
            source_entry = source_segments.get((str(segment_key[0]), str(segment_key[1])))
            if source_entry is None:
                continue
            source_q, source_rows = source_entry
            anchor_rows = group["row_idx"].to_numpy(dtype=np.int64)
            target_q = anchor_lookup_q[anchor_rows] + offset_q
            positions = np.searchsorted(source_q, target_q)
            in_bounds = positions < source_q.size
            exact = np.zeros(anchor_rows.size, dtype=bool)
            exact[in_bounds] = source_q[positions[in_bounds]] == target_q[in_bounds]
            if not np.any(exact):
                continue
            matched_anchor_rows = anchor_rows[exact]
            matched_source_rows = source_rows[positions[exact]]
            target_values = values[matched_source_rows]
            targets[matched_anchor_rows, col_start:col_end] = target_values
            mask[matched_anchor_rows, col_start:col_end] = np.isfinite(target_values)
            matched_rows[matched_anchor_rows] = True

        horizon_mask = mask[:, col_start:col_end]
        rows_with_any_valid = horizon_mask.any(axis=1)
        rows_entirely_nonfinite = matched_rows & ~rows_with_any_valid
        valid_counts = horizon_mask.sum(axis=0).astype(np.int64)
        summary = {
            "horizon_min": int(horizon_min),
            "offset_seconds": float(offset_seconds),
            "number_of_anchor_rows": int(n_rows),
            "rows_with_future_source": int(matched_rows.sum()),
            "rows_without_future_source": int(n_rows - matched_rows.sum()),
            "source_match_rate": float(matched_rows.mean()) if n_rows else 0.0,
            "rows_with_any_valid_target": int(rows_with_any_valid.sum()),
            "rows_whose_matched_source_is_entirely_non_finite": int(rows_entirely_nonfinite.sum()),
            "valid_target_values": int(horizon_mask.sum()),
            "total_target_values": int(horizon_mask.size),
            "valid_value_fraction": float(horizon_mask.mean()) if horizon_mask.size else 0.0,
            "per_feature_valid_counts": {name: int(count) for name, count in zip(spec.base_target_names, valid_counts.tolist())},
            "per_feature_valid_fractions": {
                name: float(count / max(n_rows, 1)) for name, count in zip(spec.base_target_names, valid_counts.tolist())
            },
        }
        print(json.dumps({
            "horizon_min": summary["horizon_min"],
            "offset_seconds": summary["offset_seconds"],
            "source_match_rate": summary["source_match_rate"],
            "rows_with_any_valid_target": summary["rows_with_any_valid_target"],
            "valid_value_fraction": summary["valid_value_fraction"],
        }, sort_keys=True), flush=True)
        if summary["rows_with_future_source"] == 0:
            raise ValueError(f"Horizon {horizon_min} has zero matched source rows")
        if min_source_match_rate is not None and summary["source_match_rate"] < min_source_match_rate:
            raise ValueError(
                f"Horizon {horizon_min} source match rate {summary['source_match_rate']:.6f} "
                f"is below the required minimum {min_source_match_rate:.6f}"
            )
        diagnostics["per_horizon"][str(horizon_min)] = summary
    return targets, mask, diagnostics


def validate_saved_target_bundle(output_path: Path, expected_target_names: list[str]) -> dict[str, object]:
    metadata_path = output_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing target-bundle metadata: {metadata_path}")
    with np.load(output_path, allow_pickle=True) as bundle:
        required = {
            "anchor_ids",
            "anchor_patient_ids",
            "anchor_times",
            "segment_ids",
            "segment_names",
            "split_labels",
            "input_start_times",
            "input_end_times",
            "feature_targets",
            "feature_mask",
        }
        missing = required.difference(bundle.files)
        if missing:
            raise AssertionError(f"Target bundle is missing required arrays: {sorted(missing)}")
        targets = bundle["feature_targets"]
        mask = bundle["feature_mask"]
        if targets.shape != mask.shape:
            raise AssertionError("feature_targets and feature_mask shapes differ")
        if not np.array_equal(mask, np.isfinite(targets)):
            raise AssertionError("feature_mask is not equal to target finiteness")
        n_rows = targets.shape[0]
        for name in required - {"feature_targets", "feature_mask"}:
            if len(bundle[name]) != n_rows:
                raise AssertionError(f"{name} length {len(bundle[name])} != target rows {n_rows}")
        anchor_ids = parse_integral_ids(bundle["anchor_ids"], "saved target bundle anchor_ids")
        if len(np.unique(anchor_ids)) != len(anchor_ids):
            raise AssertionError("Saved target bundle contains duplicate anchor IDs")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("feature_target_names") != expected_target_names:
        raise AssertionError("metadata feature_target_names do not match expected target column order")
    return {"saved_bundle_audit": "passed", "required_arrays": sorted(required), "n_rows": int(n_rows)}


def main() -> None:
    args = parse_args()
    horizons = validate_horizons(args.horizons)
    min_source_match_rate = validate_min_source_match_rate(args.min_source_match_rate)
    spec = FeatureRegressionTaskSpec(
        horizons_min=horizons,
        horizon_mode=args.feature_horizon_mode,
        input_window_minutes=DEFAULT_FEATURE_TASK.input_window_minutes,
        feature_names=DEFAULT_FEATURE_TASK.feature_names,
        correlation_names=DEFAULT_FEATURE_TASK.correlation_names,
        aggregation=DEFAULT_FEATURE_TASK.aggregation,
    )
    anchors, cache_diagnostics = load_cache_anchors(args.feature_cache_dir, args.full_data_root, spec.input_window_minutes)
    targets, mask, diagnostics = build_targets(anchors, args.full_data_root, spec, min_source_match_rate=min_source_match_rate)
    save_target_bundle(
        output_path=args.output,
        anchors=anchors,
        feature_targets=targets,
        feature_mask=mask,
        event_targets=None,
        event_mask=None,
        feature_spec=spec,
    )
    metadata_path = args.output.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["source"] = {
        "full_data_root": str(args.full_data_root),
        "feature_cache_dir": str(args.feature_cache_dir),
    }
    metadata["timing_semantics"] = {
        "anchor_reference": "window_center",
        "canonical_anchor_time_relation": cache_diagnostics["canonical_anchor_time_relation"],
        "canonical_anchor_time_basis_verified": cache_diagnostics["canonical_anchor_time_basis_verified"],
        "source_lookup_time_field": cache_diagnostics["source_lookup_time_field"],
        "source_lookup_time_relation": cache_diagnostics["source_lookup_time_relation"],
        "input_window_minutes": int(spec.input_window_minutes),
        "target_window_minutes": int(spec.input_window_minutes),
        "horizon_mode": spec.horizon_mode,
        "gap_mode_definition": "target_center = anchor_center + input_window_minutes + horizon_minutes",
        "center_mode_definition": "target_center = anchor_center + horizon_minutes",
        "time_quantization_decimals": TIME_DECIMALS,
    }
    metadata["feature_names"] = list(spec.feature_names)
    metadata["correlation_names"] = list(spec.correlation_names)
    metadata["target_names"] = list(spec.target_names)
    metadata["aggregation"] = spec.aggregation
    metadata["cache_identity_validation"] = cache_diagnostics
    metadata["diagnostics"] = diagnostics
    metadata_path.write_text(json.dumps(metadata, indent=2))
    audit = validate_saved_target_bundle(args.output, list(spec.target_names))
    metadata = json.loads(metadata_path.read_text())
    metadata["post_write_bundle_audit"] = audit
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"output": str(args.output), "shape": list(targets.shape), "valid_values": int(mask.sum())}, indent=2))


if __name__ == "__main__":
    main()
