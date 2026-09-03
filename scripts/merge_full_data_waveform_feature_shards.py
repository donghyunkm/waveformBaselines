#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waveform_baselines.wf_features.cache import _feature_stats, _success_marker, load_feature_cache  # noqa: E402
from waveform_baselines.wf_features.config import CACHE_ROOT, DEFAULT_EXTRACTION_CONFIG  # noqa: E402
from waveform_baselines.wf_features.definitions import FEATURE_DEFINITIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded full-data segment-aware waveform-feature caches.")
    parser.add_argument("--shard-name-prefix", type=str, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Allow merging intentionally truncated smoke-test shards.")
    return parser.parse_args()


def require_same_metadata(shards) -> None:
    first = shards[0].metadata
    required_equal = [
        "feature_version",
        "feature_names",
        "feature_units",
        "feature_descriptions",
        "channel_order",
        "extraction_config",
        "full_data_root",
        "vasopressor_free_manifest",
        "waveform_root",
        "expected_full_n_samples",
        "is_full_data_segment_window_cache",
    ]
    for shard in shards[1:]:
        for key in required_equal:
            if shard.metadata.get(key) != first.get(key):
                raise ValueError(f"Shard metadata mismatch for {key}")


def count_labels(values: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(np.asarray(values, dtype=object), return_counts=True)
    return {str(label): int(count) for label, count in zip(labels.tolist(), counts.tolist())}


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    version_root = args.cache_root / DEFAULT_EXTRACTION_CONFIG.feature_version
    output_dir = version_root / args.output_name
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is nonempty; pass --overwrite")

    shards = []
    shard_dirs = []
    for idx in range(args.shard_count):
        shard_dir = version_root / f"{args.shard_name_prefix}_{idx:03d}"
        shard = load_feature_cache(shard_dir, require_success=True)
        if not shard.metadata.get("is_full_data_segment_window_cache"):
            raise ValueError(f"{shard_dir} is not a full-data segment-window cache")
        if int(shard.metadata.get("shard_index", -1)) != idx or int(shard.metadata.get("shard_count", -1)) != args.shard_count:
            raise ValueError(f"Shard metadata mismatch in {shard_dir}")
        shards.append(shard)
        shard_dirs.append(shard_dir)

    require_same_metadata(shards)
    feature_order = shards[0].feature_names
    values = np.concatenate([np.asarray(shard.values) for shard in shards], axis=0)
    mask = np.concatenate([np.asarray(shard.mask) for shard in shards], axis=0)
    patient_ids = np.concatenate([np.asarray(shard.patient_ids).astype(str) for shard in shards], axis=0)
    anchor_times = np.concatenate([np.asarray(shard.anchor_times, dtype=np.float64) for shard in shards], axis=0)
    anchor_ids = np.concatenate([np.asarray(shard.anchor_ids, dtype=np.int64) for shard in shards], axis=0)
    split_labels = np.concatenate([np.asarray(shard.split_labels, dtype=object) for shard in shards], axis=0)
    segment_ids = np.concatenate([np.load(shard.cache_dir / "segment_ids.npy", allow_pickle=True).astype(str) for shard in shards], axis=0)
    segment_names = np.concatenate([np.load(shard.cache_dir / "segment_names.npy", allow_pickle=True).astype(str) for shard in shards], axis=0)
    anchors = pd.concat([pd.read_csv(shard.cache_dir / "anchors.csv") for shard in shards], ignore_index=True)

    expected_n = int(shards[0].metadata["expected_full_n_samples"])
    if len(anchor_ids) != expected_n and not args.allow_partial:
        raise ValueError(f"Merged shard rows {len(anchor_ids)} != expected_full_n_samples {expected_n}")
    if len(set(anchor_ids.tolist())) != len(anchor_ids):
        raise ValueError("Duplicate anchor_id values across shards")

    order = np.argsort(anchor_ids, kind="stable")
    values = values[order]
    mask = mask[order]
    patient_ids = patient_ids[order]
    anchor_times = anchor_times[order]
    anchor_ids = anchor_ids[order]
    split_labels = split_labels[order]
    segment_ids = segment_ids[order]
    segment_names = segment_names[order]
    anchors = anchors.sort_values("anchor_id", kind="stable").reset_index(drop=True)

    diagnostics: dict[str, int] = {}
    for shard in shards:
        for key, value in shard.metadata.get("extraction_diagnostics", {}).items():
            diagnostics[str(key)] = diagnostics.get(str(key), 0) + int(value)

    metadata = dict(shards[0].metadata)
    metadata.update(
        {
            "n_samples": int(values.shape[0]),
            "is_partial_merge": bool(args.allow_partial and len(anchor_ids) != expected_n),
            "is_merged_cache": True,
            "source_shard_count": args.shard_count,
            "shard_index": None,
            "shard_count": None,
            "shard_dirs": [str(path) for path in shard_dirs],
            "split_sample_counts": count_labels(split_labels),
            "split_patient_counts": count_labels(split_labels[np.unique(patient_ids, return_index=True)[1]]),
            "extraction_diagnostics": diagnostics,
            "feature_units": {feature.name: feature.unit for feature in FEATURE_DEFINITIONS},
            "feature_descriptions": {feature.name: feature.description for feature in FEATURE_DEFINITIONS},
        }
    )

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
    np.save(tmp_dir / "segment_ids.npy", segment_ids)
    np.save(tmp_dir / "segment_names.npy", segment_names)
    anchors.to_csv(tmp_dir / "anchors.csv", index=False)
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (tmp_dir / "feature_quality_report.json").write_text(json.dumps(_feature_stats(values, mask, feature_order), indent=2))
    _success_marker(tmp_dir).write_text("merge_validated=true\n")
    load_feature_cache(tmp_dir, require_success=True)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} appeared during merge")
        shutil.rmtree(output_dir)
    tmp_dir.rename(output_dir)
    print(json.dumps({"cache_dir": str(output_dir), "shape": list(values.shape), "n_samples": int(values.shape[0])}, indent=2))


if __name__ == "__main__":
    main()
