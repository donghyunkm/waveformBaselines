#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waveform_baselines.data_index import build_aligned_20m_anchor_table
from waveform_baselines.wf_features.cache import (
    _canonicalize_anchor_table,
    _count_labels,
    _ensure_unique_keys,
    _feature_stats,
    _split_patient_lookup,
    _success_marker,
    _validate_anchor_table_keys,
    load_feature_cache,
)
from waveform_baselines.wf_features.config import CACHE_ROOT, DEFAULT_EXTRACTION_CONFIG
from waveform_baselines.wf_features.definitions import FEATURE_DEFINITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded waveform-feature caches into one cache.")
    parser.add_argument("--shard-name-prefix", type=str, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _canonical_expected_anchors(splits_path: Path):
    split_lookup = _split_patient_lookup(splits_path)
    anchors = build_aligned_20m_anchor_table()
    anchors = anchors.loc[anchors["patient_id"].astype(str).isin(split_lookup)].reset_index(drop=True)
    if anchors.empty:
        raise ValueError(f"No expected anchors matched patients in split file {splits_path}")
    anchors = _canonicalize_anchor_table(anchors)
    _validate_anchor_table_keys(anchors, "expected global")
    return anchors


def _key_set(patient_ids: np.ndarray, anchor_times: np.ndarray) -> set[tuple[str, float]]:
    return set(zip(np.asarray(patient_ids, dtype=str).tolist(), np.asarray(anchor_times, dtype=np.float64).tolist()))


def _validate_diagnostics(diagnostics: dict[str, int]) -> None:
    for key, value in diagnostics.items():
        if int(value) < 0:
            raise ValueError(f"Negative extraction diagnostic {key}={value}")
    attempted = int(diagnostics.get("ecg_xqrs_runs_attempted", 0))
    used = int(diagnostics.get("ecg_xqrs_runs_used", 0))
    failed = int(diagnostics.get("ecg_xqrs_runs_failed", 0))
    zero = int(diagnostics.get("ecg_xqrs_runs_zero_peaks", 0))
    if used > attempted:
        raise ValueError("Merged ECG diagnostics invalid: xqrs_runs_used > xqrs_runs_attempted")
    if failed > attempted:
        raise ValueError("Merged ECG diagnostics invalid: xqrs_runs_failed > xqrs_runs_attempted")
    if zero > attempted:
        raise ValueError("Merged ECG diagnostics invalid: xqrs_runs_zero_peaks > xqrs_runs_attempted")


def _require_same_metadata(shards) -> None:
    first = shards[0].metadata
    required_equal = [
        "feature_version",
        "feature_names",
        "feature_units",
        "feature_descriptions",
        "sampling_rate_hz",
        "n_feature_windows",
        "channel_order",
        "extraction_config",
        "splits_path",
        "target_bundle_path",
    ]
    for shard in shards[1:]:
        for key in required_equal:
            if shard.metadata.get(key) != first.get(key):
                raise ValueError(f"Shard metadata mismatch for {key}")


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0:
        raise ValueError("shard_count must be positive")
    version_root = args.cache_root / DEFAULT_EXTRACTION_CONFIG.feature_version
    output_dir = version_root / args.output_name
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output cache {output_dir} already exists and is nonempty; pass --overwrite to replace it")

    shards = []
    shard_dirs = []
    seen_indices: set[int] = set()
    for expected_index in range(args.shard_count):
        shard_dir = version_root / f"{args.shard_name_prefix}_{expected_index:03d}"
        if not shard_dir.exists():
            legacy_dir = version_root / f"{args.shard_name_prefix}_{expected_index}"
            if legacy_dir.exists():
                raise ValueError(f"Found non-padded legacy shard directory {legacy_dir}; expected {shard_dir}")
            raise FileNotFoundError(f"Missing expected shard directory {shard_dir}")
        shard = load_feature_cache(shard_dir, require_success=True)
        meta = shard.metadata
        shard_index = int(meta.get("shard_index", -1))
        shard_count = int(meta.get("shard_count", -1))
        if shard_index != expected_index or shard_count != args.shard_count:
            raise ValueError(f"Shard metadata mismatch in {shard_dir}: index={shard_index} count={shard_count}")
        if shard_index in seen_indices:
            raise ValueError(f"Duplicate shard index {shard_index}")
        seen_indices.add(shard_index)
        shards.append(shard)
        shard_dirs.append(shard_dir)
    if seen_indices != set(range(args.shard_count)):
        raise ValueError(f"Shard index set mismatch: {sorted(seen_indices)}")

    _require_same_metadata(shards)
    feature_order = shards[0].feature_names
    metadata0 = shards[0].metadata
    expected_windows = int(metadata0["n_feature_windows"])
    expected_features = len(feature_order)
    for shard in shards:
        if shard.values.shape != shard.mask.shape:
            raise ValueError(f"Shard {shard.cache_dir} values/mask shape mismatch")
        if shard.values.ndim != 3 or shard.values.shape[1] != expected_windows or shard.values.shape[2] != expected_features:
            raise ValueError(f"Shard {shard.cache_dir} has wrong array shape {shard.values.shape}")
        n = shard.values.shape[0]
        if not (len(shard.patient_ids) == len(shard.anchor_times) == len(shard.anchor_ids) == len(shard.split_labels) == n):
            raise ValueError(f"Shard {shard.cache_dir} metadata arrays do not match values N={n}")

    values = np.concatenate([np.asarray(shard.values) for shard in shards], axis=0)
    mask = np.concatenate([np.asarray(shard.mask) for shard in shards], axis=0)
    patient_ids = np.concatenate([np.asarray(shard.patient_ids).astype(str) for shard in shards], axis=0)
    anchor_times = np.concatenate([np.asarray(shard.anchor_times, dtype=np.float64) for shard in shards], axis=0)
    anchor_ids = np.concatenate([np.asarray(shard.anchor_ids, dtype=np.int64) for shard in shards], axis=0)
    split_labels = np.concatenate([np.asarray(shard.split_labels, dtype=object) for shard in shards], axis=0)

    n_samples = int(values.shape[0])
    if not (len(patient_ids) == len(anchor_times) == len(anchor_ids) == len(split_labels) == n_samples):
        raise ValueError("Merged metadata arrays do not all have length N")
    if np.any(split_labels == "unknown"):
        raise ValueError("Merged cache contains unknown split labels")
    _ensure_unique_keys(list(zip(patient_ids.tolist(), anchor_times.tolist())), "merged feature (patient_id, anchor_time)")
    _ensure_unique_keys([(int(anchor_id),) for anchor_id in anchor_ids.tolist()], "merged feature anchor_id")

    expected = _canonical_expected_anchors(Path(str(metadata0["splits_path"])))
    expected_patient_ids = expected["patient_id"].astype(str).to_numpy()
    expected_anchor_times = expected["anchor_time"].to_numpy(dtype=np.float64)
    expected_anchor_ids = expected["anchor_id"].to_numpy(dtype=np.int64)
    if n_samples != len(expected):
        raise ValueError(f"Merged sample count {n_samples} does not match expected full anchor count {len(expected)}")
    if set(anchor_ids.tolist()) != set(expected_anchor_ids.tolist()):
        raise ValueError("Merged anchor_id set does not exactly match expected full anchor_id set")
    if _key_set(patient_ids, anchor_times) != _key_set(expected_patient_ids, expected_anchor_times):
        raise ValueError("Merged (patient_id, anchor_time) set does not exactly match expected full anchor set")

    order = np.argsort(anchor_ids, kind="stable")
    values = values[order]
    mask = mask[order]
    patient_ids = patient_ids[order]
    anchor_times = anchor_times[order]
    anchor_ids = anchor_ids[order]
    split_labels = split_labels[order]
    if not np.array_equal(anchor_ids, expected_anchor_ids):
        raise ValueError("Merged canonical anchor_id order does not match expected order")

    split_sample_counts = _count_labels(split_labels)
    unique_patient_ids, unique_indices = np.unique(patient_ids, return_index=True)
    split_patient_counts = _count_labels(split_labels[unique_indices])

    extraction_diagnostics: dict[str, int] = {}
    for shard in shards:
        for key, value in shard.metadata.get("extraction_diagnostics", {}).items():
            extraction_diagnostics[str(key)] = extraction_diagnostics.get(str(key), 0) + int(value)
    _validate_diagnostics(extraction_diagnostics)

    quality_report = _feature_stats(values, mask, feature_order)
    metadata = dict(metadata0)
    metadata.update({
        "n_samples": n_samples,
        "expected_full_n_samples": int(len(expected)),
        "is_merged_cache": True,
        "source_shard_count": args.shard_count,
        "shard_count": None,
        "shard_index": None,
        "shard_assignment": "canonical_anchor_id_sort_then_positional_stride",
        "shard_dirs": [str(path) for path in shard_dirs],
        "split_sample_counts": split_sample_counts,
        "split_patient_counts": split_patient_counts,
        "extraction_diagnostics": extraction_diagnostics,
        "feature_units": {feature.name: feature.unit for feature in FEATURE_DEFINITIONS},
        "feature_descriptions": {feature.name: feature.description for feature in FEATURE_DEFINITIONS},
    })

    tmp_dir = version_root / f".{args.output_name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    np.save(tmp_dir / "values.npy", values.astype(np.float32, copy=False))
    np.save(tmp_dir / "mask.npy", mask.astype(bool, copy=False))
    np.save(tmp_dir / "patient_ids.npy", patient_ids)
    np.save(tmp_dir / "anchor_times.npy", anchor_times)
    np.save(tmp_dir / "anchor_ids.npy", anchor_ids)
    np.save(tmp_dir / "split_labels.npy", split_labels)
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (tmp_dir / "feature_quality_report.json").write_text(json.dumps(quality_report, indent=2))
    _success_marker(tmp_dir).write_text("merge_validated=true\n")
    load_feature_cache(tmp_dir, require_success=True)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output cache {output_dir} appeared during merge")
        shutil.rmtree(output_dir)
    tmp_dir.rename(output_dir)
    print(json.dumps({"cache_dir": str(output_dir), "n_samples": n_samples, "shape": list(values.shape)}, indent=2))


if __name__ == "__main__":
    main()
