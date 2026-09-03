from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from waveform_baselines.wf_features.cache import FeatureCache, load_feature_cache

from .definitions import FEATURE_DEFINITIONS_V8, feature_names


@dataclass
class V8FeatureCache:
    values: np.ndarray
    mask: np.ndarray
    patient_ids: np.ndarray
    anchor_times: np.ndarray
    anchor_ids: np.ndarray
    split_labels: np.ndarray
    feature_names: list[str]
    metadata: dict[str, object]
    cache_dir: Path
    segment_ids: np.ndarray | None = None
    segment_names: np.ndarray | None = None


@dataclass
class CombinedFeatureCache:
    values: np.ndarray
    mask: np.ndarray
    patient_ids: np.ndarray
    anchor_times: np.ndarray
    anchor_ids: np.ndarray
    split_labels: np.ndarray
    feature_names: list[str]
    metadata: dict[str, object]
    v7_feature_count: int
    v8_feature_count: int


DERIVED_PHYSIOLOGICAL_FEATURES = [
    ("shock_index", "ratio", "ecg_hr_bpm / abp_sbp_median_mmhg"),
    ("modified_shock_index", "ratio", "ecg_hr_bpm / abp_map_median_mmhg"),
    ("diastolic_shock_index", "ratio", "ecg_hr_bpm / abp_dbp_median_mmhg"),
    ("map_margin_65", "mmHg", "abp_map_median_mmhg - 65"),
    ("sbp_margin_90", "mmHg", "abp_sbp_median_mmhg - 90"),
]

CURRENT_V8_SCHEMA_REVISION = 7


def current_v8_schema_hash() -> str:
    from .config import DEFAULT_V8_EXTRACTION_CONFIG

    payload = {
        "feature_version": "v8",
        "feature_schema_revision": CURRENT_V8_SCHEMA_REVISION,
        "features": [
            {
                "name": feature.name,
                "unit": feature.unit,
                "enabled_by_default": feature.enabled_by_default,
                "feature_role": feature.feature_role,
                "synchronization_required": feature.synchronization_required,
            }
            for feature in FEATURE_DEFINITIONS_V8
        ],
        "schema_config": DEFAULT_V8_EXTRACTION_CONFIG.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _success_marker(cache_dir: Path) -> Path:
    return Path(cache_dir) / "_SUCCESS"


def feature_stats(values: np.ndarray, mask: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        valid = mask[:, :, idx] & np.isfinite(values[:, :, idx])
        arr = values[:, :, idx][valid].astype(np.float64, copy=False)
        if arr.size == 0:
            report[name] = {"count": 0, "valid_fraction": 0.0, "missing_fraction": 1.0, "non_finite_fraction": 1.0, "unique_finite_count": 0}
            continue
        report[name] = {
            "count": int(arr.size),
            "valid_fraction": float(np.mean(valid)),
            "missing_fraction": float(1.0 - np.mean(valid)),
            "non_finite_fraction": float(np.mean(~np.isfinite(values[:, :, idx]))),
            "unique_finite_count": int(np.unique(arr).size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "iqr": float(np.percentile(arr, 75.0) - np.percentile(arr, 25.0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p01": float(np.percentile(arr, 1.0)),
            "p05": float(np.percentile(arr, 5.0)),
            "p95": float(np.percentile(arr, 95.0)),
            "p99": float(np.percentile(arr, 99.0)),
        }
    return report


def load_v8_feature_cache(cache_dir: Path, require_success: bool = True) -> V8FeatureCache:
    cache_dir = Path(cache_dir)
    if require_success and not _success_marker(cache_dir).exists():
        raise FileNotFoundError(f"V8 feature cache {cache_dir} is missing _SUCCESS")
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    values = np.load(cache_dir / "values.npy", mmap_mode="r")
    mask = np.load(cache_dir / "mask.npy", mmap_mode="r")
    patient_ids = np.load(cache_dir / "patient_ids.npy", allow_pickle=True)
    anchor_times = np.load(cache_dir / "anchor_times.npy")
    anchor_ids = np.load(cache_dir / "anchor_ids.npy")
    split_labels = np.load(cache_dir / "split_labels.npy", allow_pickle=True)
    segment_ids = np.load(cache_dir / "segment_ids.npy", allow_pickle=True).astype(str) if (cache_dir / "segment_ids.npy").exists() else None
    segment_names = np.load(cache_dir / "segment_names.npy", allow_pickle=True).astype(str) if (cache_dir / "segment_names.npy").exists() else None
    names = list(metadata["feature_names"])
    if metadata.get("feature_version") != "v8":
        raise ValueError(f"V8 cache metadata feature_version must be 'v8', got {metadata.get('feature_version')!r}")
    if len(names) != len(set(names)):
        raise ValueError("V8 cache metadata contains duplicate feature names")
    if values.shape != mask.shape or values.ndim != 3:
        raise ValueError(f"Invalid v8 values/mask shapes: {values.shape} vs {mask.shape}")
    if values.shape[2] != len(names):
        raise ValueError(f"V8 feature dimension {values.shape[2]} does not match metadata names {len(names)}")
    if names != feature_names():
        raise ValueError("V8 cache feature_names do not match current v8 definitions")
    expected_hash = current_v8_schema_hash()
    if metadata.get("feature_schema_revision") != CURRENT_V8_SCHEMA_REVISION:
        raise ValueError(
            f"Stale V8 cache schema revision {metadata.get('feature_schema_revision')!r}; "
            f"expected {CURRENT_V8_SCHEMA_REVISION}"
        )
    if metadata.get("feature_schema_hash") != expected_hash:
        raise ValueError("V8 cache schema hash does not match current v8 definitions/configuration")
    n = values.shape[0]
    if not (len(patient_ids) == len(anchor_times) == len(anchor_ids) == len(split_labels) == n):
        raise ValueError(f"V8 metadata arrays do not all have length N={n}")
    return V8FeatureCache(values, mask, patient_ids, anchor_times, anchor_ids, split_labels, names, metadata, cache_dir, segment_ids, segment_names)


def subset_feature_cache_by_anchor_ids(cache: FeatureCache, anchor_ids: np.ndarray) -> FeatureCache:
    requested = np.asarray(anchor_ids, dtype=np.int64)
    source_ids = np.asarray(cache.anchor_ids, dtype=np.int64)
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError("Cannot subset feature cache with duplicate source anchor_ids")
    if len(np.unique(requested)) != len(requested):
        raise ValueError("Cannot subset feature cache with duplicate requested anchor_ids")
    order = np.argsort(source_ids, kind="stable")
    sorted_ids = source_ids[order]
    positions = np.searchsorted(sorted_ids, requested)
    missing = (positions >= len(sorted_ids)) | (sorted_ids[np.minimum(positions, len(sorted_ids) - 1)] != requested)
    if np.any(missing):
        examples = ", ".join(str(int(value)) for value in requested[missing][:10])
        raise ValueError(f"Requested anchor_ids not found in source feature cache: {examples}")
    indices = order[positions]
    return FeatureCache(
        values=cache.values[indices],
        mask=cache.mask[indices],
        patient_ids=cache.patient_ids[indices],
        anchor_times=cache.anchor_times[indices],
        anchor_ids=cache.anchor_ids[indices],
        split_labels=cache.split_labels[indices],
        feature_names=cache.feature_names,
        metadata=cache.metadata,
        cache_dir=cache.cache_dir,
        segment_ids=cache.segment_ids[indices] if cache.segment_ids is not None else None,
        segment_names=cache.segment_names[indices] if cache.segment_names is not None else None,
    )


def validate_v7_v8_alignment(v7: FeatureCache, v8: V8FeatureCache, anchor_time_atol: float = 1e-6) -> dict[str, object]:
    if v7.values.shape[:2] != v8.values.shape[:2]:
        raise ValueError(f"v7/v8 tensor sample/time shapes differ: {v7.values.shape[:2]} vs {v8.values.shape[:2]}")
    if v7.values.shape[0] != v8.values.shape[0]:
        raise ValueError(f"v7.N={v7.values.shape[0]} does not match v8.N={v8.values.shape[0]}")
    if not np.array_equal(np.asarray(v7.anchor_ids, dtype=np.int64), np.asarray(v8.anchor_ids, dtype=np.int64)):
        raise ValueError("v7/v8 anchor_ids differ or are not in the same order")
    if not np.array_equal(np.asarray(v7.patient_ids, dtype=str), np.asarray(v8.patient_ids, dtype=str)):
        raise ValueError("v7/v8 patient_ids differ or are not in the same order")
    if not np.allclose(np.asarray(v7.anchor_times, dtype=np.float64), np.asarray(v8.anchor_times, dtype=np.float64), atol=anchor_time_atol, rtol=0.0):
        max_diff = float(np.max(np.abs(np.asarray(v7.anchor_times, dtype=np.float64) - np.asarray(v8.anchor_times, dtype=np.float64))))
        raise ValueError(f"v7/v8 anchor_times differ; max absolute difference={max_diff}")
    if not np.array_equal(np.asarray(v7.split_labels, dtype=str), np.asarray(v8.split_labels, dtype=str)):
        raise ValueError("v7/v8 split_labels differ or are not in the same order")
    overlap = sorted(set(v7.feature_names).intersection(v8.feature_names))
    if overlap:
        raise ValueError(f"v7/v8 feature name overlap is not allowed: {overlap[:10]}")
    return {
        "n_samples": int(v8.values.shape[0]),
        "n_feature_windows": int(v8.values.shape[1]),
        "v7_feature_count": int(v7.values.shape[2]),
        "v8_feature_count": int(v8.values.shape[2]),
        "anchor_time_atol": anchor_time_atol,
        "feature_name_overlap_count": 0,
    }


def load_combined_feature_cache(v7_path: Path, v8_path: Path, require_success: bool = True) -> CombinedFeatureCache:
    v7 = load_feature_cache(Path(v7_path), require_success=require_success)
    v8 = load_v8_feature_cache(Path(v8_path), require_success=require_success)
    alignment = validate_v7_v8_alignment(v7, v8)
    values = np.concatenate([np.asarray(v7.values), np.asarray(v8.values)], axis=-1)
    mask = np.concatenate([np.asarray(v7.mask), np.asarray(v8.mask)], axis=-1)
    names = list(v7.feature_names) + list(v8.feature_names)
    metadata = {
        "source_caches": {"v7": str(v7_path), "v8": str(v8_path)},
        "feature_names": names,
        "feature_sources": {name: "v7" for name in v7.feature_names} | {name: "v8" for name in v8.feature_names},
        "v7_feature_names": list(v7.feature_names),
        "v8_feature_names": list(v8.feature_names),
        "alignment": alignment,
    }
    return CombinedFeatureCache(values, mask, v7.patient_ids, v7.anchor_times, v7.anchor_ids, v7.split_labels, names, metadata, len(v7.feature_names), len(v8.feature_names))


def add_derived_physiological_features(cache: CombinedFeatureCache) -> CombinedFeatureCache:
    """Append downstream-derived causal features without modifying raw v7/v8 caches."""
    names = list(cache.feature_names)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    required = ["ecg_hr_bpm", "abp_sbp_median_mmhg", "abp_dbp_median_mmhg", "abp_map_median_mmhg"]
    missing = [name for name in required if name not in name_to_idx]
    if missing:
        raise ValueError(f"Cannot derive physiological features; missing required columns: {missing}")
    derived_values = np.full((*cache.values.shape[:2], len(DERIVED_PHYSIOLOGICAL_FEATURES)), np.nan, dtype=cache.values.dtype)
    derived_mask = np.zeros(derived_values.shape, dtype=bool)

    def column(name: str) -> tuple[np.ndarray, np.ndarray]:
        idx = name_to_idx[name]
        vals = np.asarray(cache.values[:, :, idx], dtype=np.float64)
        valid = np.asarray(cache.mask[:, :, idx], dtype=bool) & np.isfinite(vals)
        return vals, valid

    hr, hr_m = column("ecg_hr_bpm")
    sbp, sbp_m = column("abp_sbp_median_mmhg")
    dbp, dbp_m = column("abp_dbp_median_mmhg")
    mapv, map_m = column("abp_map_median_mmhg")
    calculations = [
        (hr / sbp, hr_m & sbp_m & (np.abs(sbp) > 1e-8)),
        (hr / mapv, hr_m & map_m & (np.abs(mapv) > 1e-8)),
        (hr / dbp, hr_m & dbp_m & (np.abs(dbp) > 1e-8)),
        (mapv - 65.0, map_m),
        (sbp - 90.0, sbp_m),
    ]
    for idx, (vals, valid) in enumerate(calculations):
        ok = valid & np.isfinite(vals)
        derived_values[:, :, idx][ok] = vals[ok].astype(cache.values.dtype, copy=False)
        derived_mask[:, :, idx] = ok
    new_names = names + [name for name, _, _ in DERIVED_PHYSIOLOGICAL_FEATURES]
    metadata = dict(cache.metadata)
    metadata["feature_names"] = new_names
    metadata["downstream_derived_features"] = [
        {"name": name, "units": unit, "formula": formula, "source": "derived_after_v7_v8_join", "stored_in_raw_v8_cache": False}
        for name, unit, formula in DERIVED_PHYSIOLOGICAL_FEATURES
    ]
    return CombinedFeatureCache(
        np.concatenate([np.asarray(cache.values), derived_values], axis=-1),
        np.concatenate([np.asarray(cache.mask), derived_mask], axis=-1),
        cache.patient_ids,
        cache.anchor_times,
        cache.anchor_ids,
        cache.split_labels,
        new_names,
        metadata,
        cache.v7_feature_count,
        cache.v8_feature_count,
    )


def metadata_payload(base: dict[str, object]) -> dict[str, object]:
    names = feature_names()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate v8 feature names are not allowed")
    feature_rows = [
        {
            "name": feature.name,
            "description": feature.description,
            "units": feature.unit,
            "source_channels": list(feature.channels),
            "window_duration": "5 minutes" if feature.name.endswith("_5m") else "1 minute",
            "minimum_observations": feature.minimum_required,
            "validity_rules": feature.minimum_required,
            "missingness_behavior": "NaN with mask=false when insufficient, unreliable, non-finite, disabled, or synchronization-gated.",
            "feature_role": feature.feature_role,
            "requires_channel_synchronization": feature.synchronization_required,
            "enabled_by_default": feature.enabled_by_default,
        }
        for feature in FEATURE_DEFINITIONS_V8
    ]
    out = dict(base)
    out.update({
        "feature_version": "v8",
        "feature_schema_revision": CURRENT_V8_SCHEMA_REVISION,
        "feature_schema_hash": current_v8_schema_hash(),
        "feature_names": names,
        "features": feature_rows,
        "feature_units": {feature.name: feature.unit for feature in FEATURE_DEFINITIONS_V8},
        "feature_roles": {feature.name: feature.feature_role for feature in FEATURE_DEFINITIONS_V8},
        "feature_descriptions": {feature.name: feature.description for feature in FEATURE_DEFINITIONS_V8},
        "feature_channels": {feature.name: list(feature.channels) for feature in FEATURE_DEFINITIONS_V8},
        "feature_minimum_required": {feature.name: feature.minimum_required for feature in FEATURE_DEFINITIONS_V8},
        "feature_valid_ranges": {feature.name: feature.valid_range for feature in FEATURE_DEFINITIONS_V8},
        "feature_synchronization_required": {feature.name: feature.synchronization_required for feature in FEATURE_DEFINITIONS_V8},
        "feature_enabled_by_default": {feature.name: feature.enabled_by_default for feature in FEATURE_DEFINITIONS_V8},
        "missing_data_behavior": "Unreliable or insufficient measurements are stored as NaN with mask=false.",
        "causal_interval": "Each token uses only samples from the corresponding minute ending at the token endpoint; rolling 5-minute features use only the preceding 5 minutes ending at that endpoint.",
        "task_independent": True,
    })
    return out
