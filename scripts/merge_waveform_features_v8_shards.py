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

from waveform_baselines.wf_features_v8.cache import feature_stats, load_v8_feature_cache  # noqa: E402
from waveform_baselines.wf_features_v8.config import DEFAULT_V8_EXTRACTION_CONFIG, V8_CACHE_ROOT  # noqa: E402
from waveform_baselines.wf_features_v8.definitions import FEATURE_DEFINITIONS_V8  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded full-data v8 waveform-feature caches.")
    parser.add_argument("--shard-name-prefix", type=str, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--cache-root", type=Path, default=V8_CACHE_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Allow merging intentionally truncated smoke-test shards.")
    parser.add_argument("--quality-mode", choices=("aggregate-shards", "exact"), default="aggregate-shards")
    return parser.parse_args()


def require_same_metadata(shards) -> None:
    first = shards[0].metadata
    required_equal = [
        "feature_version",
        "feature_schema_revision",
        "feature_schema_hash",
        "feature_names",
        "feature_units",
        "feature_descriptions",
        "feature_roles",
        "feature_synchronization_required",
        "feature_enabled_by_default",
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


def aggregate_shard_feature_stats(shards, names: list[str]) -> dict[str, dict[str, float | int | bool | None | str]]:
    total_positions = int(sum(int(shard.values.shape[0]) * int(shard.values.shape[1]) for shard in shards))
    reports = [json.loads((shard.cache_dir / "feature_quality_report.json").read_text()) for shard in shards]
    shard_positions = [int(shard.values.shape[0]) * int(shard.values.shape[1]) for shard in shards]
    merged: dict[str, dict[str, float | int | bool | None | str]] = {}

    def weighted(rows: list[tuple[float, float]]) -> float | None:
        weight_sum = float(sum(weight for weight, _ in rows))
        if weight_sum <= 0.0:
            return None
        return float(sum(weight * value for weight, value in rows) / weight_sum)

    for name in names:
        rows = [report[name] for report in reports]
        count = int(sum(int(row.get("count", 0)) for row in rows))
        non_finite_total = float(sum(float(row.get("non_finite_fraction", 0.0)) * positions for row, positions in zip(rows, shard_positions)))
        if count == 0:
            merged[name] = {
                "count": 0,
                "valid_fraction": 0.0,
                "missing_fraction": 1.0,
                "non_finite_fraction": float(non_finite_total / total_positions) if total_positions else 0.0,
                "unique_finite_count": 0,
                "aggregation": "merged_from_shard_quality_reports",
            }
            continue
        weighted_rows = [(float(row.get("count", 0)), row) for row in rows if int(row.get("count", 0)) > 0]
        mean = weighted([(weight, float(row["mean"])) for weight, row in weighted_rows])
        second_moment = weighted([(weight, float(row["std"]) ** 2 + float(row["mean"]) ** 2) for weight, row in weighted_rows])
        variance = max(0.0, float(second_moment) - float(mean) ** 2) if mean is not None and second_moment is not None else None
        out: dict[str, float | int | bool | None | str] = {
            "count": count,
            "valid_fraction": float(count / total_positions),
            "missing_fraction": float(1.0 - count / total_positions),
            "non_finite_fraction": float(non_finite_total / total_positions) if total_positions else 0.0,
            "unique_finite_count": int(max(int(row.get("unique_finite_count", 0)) for row in rows)),
            "mean": mean,
            "std": float(np.sqrt(variance)) if variance is not None else None,
            "min": float(min(float(row["min"]) for _, row in weighted_rows)),
            "max": float(max(float(row["max"]) for _, row in weighted_rows)),
            "aggregation": "merged_from_shard_quality_reports",
            "quantiles_are_weighted_shard_summaries": True,
        }
        for key in ("median", "iqr", "p01", "p05", "p95", "p99"):
            out[key] = weighted([(weight, float(row[key])) for weight, row in weighted_rows if key in row])
        merged[name] = out
    return merged


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    version_root = args.cache_root / DEFAULT_V8_EXTRACTION_CONFIG.feature_version
    output_dir = version_root / args.output_name
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is nonempty; pass --overwrite")

    shards = []
    shard_dirs = []
    for idx in range(args.shard_count):
        shard_dir = version_root / f"{args.shard_name_prefix}_{idx:04d}"
        shard = load_v8_feature_cache(shard_dir, require_success=True)
        if not shard.metadata.get("is_full_data_segment_window_cache"):
            raise ValueError(f"{shard_dir} is not a full-data segment-window cache")
        if int(shard.metadata.get("shard_index", -1)) != idx or int(shard.metadata.get("shard_count", -1)) != args.shard_count:
            raise ValueError(f"Shard metadata mismatch in {shard_dir}")
        shards.append(shard)
        shard_dirs.append(shard_dir)

    require_same_metadata(shards)
    feature_order = shards[0].feature_names
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
    patient_ids = patient_ids[order]
    anchor_times = anchor_times[order]
    anchor_ids = anchor_ids[order]
    split_labels = split_labels[order]
    segment_ids = segment_ids[order]
    segment_names = segment_names[order]
    anchors = anchors.sort_values("anchor_id", kind="stable").reset_index(drop=True)

    metadata = dict(shards[0].metadata)
    metadata.update({
        "n_samples": int(len(anchor_ids)),
        "is_partial_merge": bool(args.allow_partial and len(anchor_ids) != expected_n),
        "is_merged_cache": True,
        "source_shard_count": args.shard_count,
        "shard_index": None,
        "shard_count": None,
        "shard_dirs": [str(path) for path in shard_dirs],
        "split_sample_counts": count_labels(split_labels),
        "split_patient_counts": count_labels(split_labels[np.unique(patient_ids, return_index=True)[1]]),
        "feature_units": {feature.name: feature.unit for feature in FEATURE_DEFINITIONS_V8},
        "feature_descriptions": {feature.name: feature.description for feature in FEATURE_DEFINITIONS_V8},
    })

    tmp_dir = version_root / f".{args.output_name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    values_shape = (len(anchor_ids), int(shards[0].values.shape[1]), int(shards[0].values.shape[2]))
    values = np.lib.format.open_memmap(tmp_dir / "values.npy", mode="w+", dtype=np.float32, shape=values_shape)
    mask = np.lib.format.open_memmap(tmp_dir / "mask.npy", mode="w+", dtype=bool, shape=values_shape)
    for shard in shards:
        shard_ids = np.asarray(shard.anchor_ids, dtype=np.int64)
        positions = np.searchsorted(anchor_ids, shard_ids)
        if np.any(positions >= len(anchor_ids)) or not np.array_equal(anchor_ids[positions], shard_ids):
            raise ValueError(f"Shard {shard.cache_dir} anchor_ids are not present in merged anchor order")
        values[positions] = np.asarray(shard.values, dtype=np.float32)
        mask[positions] = np.asarray(shard.mask, dtype=bool)
    values.flush()
    mask.flush()
    np.save(tmp_dir / "patient_ids.npy", patient_ids)
    np.save(tmp_dir / "anchor_times.npy", anchor_times)
    np.save(tmp_dir / "anchor_ids.npy", anchor_ids)
    np.save(tmp_dir / "split_labels.npy", split_labels)
    np.save(tmp_dir / "segment_ids.npy", segment_ids)
    np.save(tmp_dir / "segment_names.npy", segment_names)
    anchors.to_csv(tmp_dir / "anchors.csv", index=False)
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if args.quality_mode == "exact":
        quality = feature_stats(values, mask, feature_order)
    else:
        quality = aggregate_shard_feature_stats(shards, feature_order)
        metadata["feature_quality_report_mode"] = "aggregate-shards"
        (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (tmp_dir / "feature_quality_report.json").write_text(json.dumps(quality, indent=2))
    (tmp_dir / "_SUCCESS").write_text("merge_validated=true\n")
    load_v8_feature_cache(tmp_dir, require_success=True)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} appeared during merge")
        shutil.rmtree(output_dir)
    tmp_dir.rename(output_dir)
    print(json.dumps({"cache_dir": str(output_dir), "shape": list(values.shape), "n_samples": int(values.shape[0])}, indent=2))


if __name__ == "__main__":
    main()
