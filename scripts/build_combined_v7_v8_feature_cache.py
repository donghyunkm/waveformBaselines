#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waveform_baselines.wf_features.cache import load_feature_cache
from waveform_baselines.wf_features_v8.cache import load_v8_feature_cache, validate_v7_v8_alignment


DEFAULT_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full")
DEFAULT_V7 = DEFAULT_ROOT / "v7" / "full_data_vasopressor_free_waveform_features_v7"
DEFAULT_V8 = DEFAULT_ROOT / "v8" / "full_data_vasopressor_free_waveform_features_v8_segment_plan"
DEFAULT_OUTPUT = DEFAULT_ROOT / "combined_v7_v8" / "full_data_vasopressor_free_waveform_features_v7_v8_segment_plan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a row-aligned combined v7+v8 extracted-feature cache.")
    parser.add_argument("--v7-cache", type=Path, default=DEFAULT_V7)
    parser.add_argument("--v8-cache", type=Path, default=DEFAULT_V8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is nonempty; pass --overwrite")

    v7 = load_feature_cache(args.v7_cache, require_success=True)
    v8 = load_v8_feature_cache(args.v8_cache, require_success=True)
    alignment = validate_v7_v8_alignment(v7, v8)
    if v7.values.shape[:2] != v8.values.shape[:2]:
        raise ValueError(f"v7/v8 shape mismatch: {v7.values.shape[:2]} vs {v8.values.shape[:2]}")

    tmp_dir = output_dir.parent / f".{output_dir.name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    n_samples, n_windows = v7.values.shape[:2]
    n_features = int(v7.values.shape[2] + v8.values.shape[2])
    values = np.lib.format.open_memmap(tmp_dir / "values.npy", mode="w+", dtype=np.float32, shape=(n_samples, n_windows, n_features))
    mask = np.lib.format.open_memmap(tmp_dir / "mask.npy", mode="w+", dtype=bool, shape=(n_samples, n_windows, n_features))
    v7_width = int(v7.values.shape[2])
    values[:, :, :v7_width] = np.asarray(v7.values, dtype=np.float32)
    values[:, :, v7_width:] = np.asarray(v8.values, dtype=np.float32)
    mask[:, :, :v7_width] = np.asarray(v7.mask, dtype=bool)
    mask[:, :, v7_width:] = np.asarray(v8.mask, dtype=bool)
    values.flush()
    mask.flush()

    np.save(tmp_dir / "patient_ids.npy", np.asarray(v7.patient_ids).astype(str))
    np.save(tmp_dir / "anchor_times.npy", np.asarray(v7.anchor_times, dtype=np.float64))
    np.save(tmp_dir / "anchor_ids.npy", np.asarray(v7.anchor_ids, dtype=np.int64))
    np.save(tmp_dir / "split_labels.npy", np.asarray(v7.split_labels).astype(str))
    if v7.segment_ids is not None:
        np.save(tmp_dir / "segment_ids.npy", np.asarray(v7.segment_ids).astype(str))
    if v7.segment_names is not None:
        np.save(tmp_dir / "segment_names.npy", np.asarray(v7.segment_names).astype(str))

    anchors_path = args.v8_cache / "anchors.csv"
    if anchors_path.exists():
        shutil.copy2(anchors_path, tmp_dir / "anchors.csv")

    feature_names = list(v7.feature_names) + list(v8.feature_names)
    metadata = {
        "feature_version": "combined_v7_v8",
        "n_samples": int(n_samples),
        "n_feature_windows": int(n_windows),
        "feature_names": feature_names,
        "is_merged_cache": True,
        "source_caches": {"v7": str(args.v7_cache), "v8": str(args.v8_cache)},
        "v7_feature_count": int(v7.values.shape[2]),
        "v8_feature_count": int(v8.values.shape[2]),
        "alignment": alignment,
        "v7_metadata": {
            "feature_version": v7.metadata.get("feature_version"),
            "feature_schema_revision": v7.metadata.get("feature_schema_revision"),
            "feature_schema_hash": v7.metadata.get("feature_schema_hash"),
        },
        "v8_metadata": {
            "feature_version": v8.metadata.get("feature_version"),
            "feature_schema_revision": v8.metadata.get("feature_schema_revision"),
            "feature_schema_hash": v8.metadata.get("feature_schema_hash"),
            "feature_quality_report_mode": v8.metadata.get("feature_quality_report_mode"),
        },
    }
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (tmp_dir / "_SUCCESS").write_text(f"completed_at_unix={time.time():.6f}\n")
    load_feature_cache(tmp_dir, require_success=True)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} appeared during write")
        shutil.rmtree(output_dir)
    tmp_dir.rename(output_dir)
    print(json.dumps({"cache_dir": str(output_dir), "shape": [int(n_samples), int(n_windows), int(n_features)]}, indent=2))


if __name__ == "__main__":
    main()
