#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_FULL_DATA_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/physionet_restricted/mimic_derived_data/data_m3_120s_prediction")
DEFAULT_MANIFEST = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/"
    "full_data_segment_level_vasopressor_free_waveform_manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/raw_anchors/"
    "full_data_vasopressor_present_waveform_anchors"
)
CHANNEL_ORDER = ("II", "ABP", "PLETH", "RESP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a full-data raw-waveform anchor cache for confirmed vasopressor-overlap segments."
    )
    parser.add_argument("--full-data-root", type=Path, default=DEFAULT_FULL_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_lookup(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    if all(isinstance(v, str) for v in payload.values()):
        return {str(pid): str(label) for pid, label in payload.items()}
    out: dict[str, str] = {}
    for label, patients in payload.items():
        if isinstance(patients, list):
            for patient in patients:
                out[str(patient)] = str(label)
    return out


def _true_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def load_confirmed_overlap_segments(path: Path) -> pd.DataFrame:
    required = [
        "segment_id",
        "segment_name",
        "segment_path",
        "has_vasopressor_overlap",
        "vasopressor_free",
        "overlapping_vasopressor_count",
        "first_overlapping_vasopressor_time",
        "last_overlapping_vasopressor_time",
        "first_overlapping_vasopressor_name",
        "vasopressor_source",
        "vasopressor_match_method",
        "classification_status",
    ]
    manifest = pd.read_csv(path, usecols=required, low_memory=False)
    overlap = manifest[_true_mask(manifest["has_vasopressor_overlap"])].copy()
    if overlap.empty:
        raise ValueError(f"No confirmed vasopressor-overlap segments found in {path}")
    contradictory = overlap[overlap["vasopressor_free"].astype(str).str.lower().eq("true")]
    if not contradictory.empty:
        examples = contradictory["segment_id"].head(10).tolist()
        raise ValueError(f"Confirmed-overlap rows also marked vasopressor_free=True, examples={examples}")
    if overlap["segment_id"].duplicated().any():
        dupes = overlap.loc[overlap["segment_id"].duplicated(), "segment_id"].head(10).tolist()
        raise ValueError(f"Duplicate confirmed-overlap segment_id values in {path}: {dupes}")
    return overlap


def build_anchor_table(full_data_root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    patient_ids = np.load(full_data_root / "patient_ids.npy", mmap_mode="r", allow_pickle=True)
    seg_names = np.load(full_data_root / "seg_names.npy", mmap_mode="r", allow_pickle=True)
    window_times = np.load(full_data_root / "window_times.npy", mmap_mode="r")
    if not (len(patient_ids) == len(seg_names) == len(window_times)):
        raise ValueError("Full-data patient_ids, seg_names, and window_times arrays must have equal length")

    anchors = pd.DataFrame(
        {
            "patient_id": np.asarray(patient_ids, dtype=str),
            "seg_name": np.asarray(seg_names, dtype=str),
            "window_time": np.asarray(window_times, dtype=np.float64),
            "anchor_id": np.arange(len(patient_ids), dtype=np.int64),
        }
    )
    anchors["segment_id"] = anchors["patient_id"] + "/" + anchors["seg_name"]
    anchors = anchors.merge(manifest, on="segment_id", how="inner", validate="many_to_one")
    if anchors.empty:
        raise ValueError("No full-data windows matched confirmed vasopressor-overlap segments")
    if "segment_name" in anchors.columns:
        mismatch = anchors["seg_name"].astype(str) != anchors["segment_name"].astype(str)
        if mismatch.any():
            examples = anchors.loc[mismatch, ["segment_id", "seg_name", "segment_name"]].head(10).to_dict("records")
            raise ValueError(f"seg_names.npy does not match manifest segment_name, examples={examples}")
    return anchors.sort_values("anchor_id", kind="stable").reset_index(drop=True)


def apply_splits(anchors: pd.DataFrame, split_path: Path) -> pd.DataFrame:
    lookup = split_lookup(split_path)
    anchors = anchors.copy()
    anchors["split_label"] = anchors["patient_id"].map(lookup)
    missing = anchors["split_label"].isna()
    if missing.any():
        examples = sorted(anchors.loc[missing, "patient_id"].unique().tolist())[:10]
        raise ValueError(f"Patients missing from split file {split_path}: {examples}")
    return anchors


def count_labels(values: pd.Series | np.ndarray) -> dict[str, int]:
    counts = Counter(np.asarray(values, dtype=object).tolist())
    return {str(k): int(v) for k, v in sorted(counts.items())}


def write_cache(anchors: pd.DataFrame, args: argparse.Namespace, full_count: int) -> None:
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} exists and is nonempty; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    n = len(anchors)
    np.save(out / "values.npy", np.empty((n, 0, 0), dtype=np.float32))
    np.save(out / "mask.npy", np.empty((n, 0, 0), dtype=bool))
    np.save(out / "patient_ids.npy", anchors["patient_id"].astype(str).to_numpy())
    np.save(out / "anchor_times.npy", anchors["window_time"].to_numpy(dtype=np.float64))
    np.save(out / "anchor_ids.npy", anchors["anchor_id"].to_numpy(dtype=np.int64))
    np.save(out / "split_labels.npy", anchors["split_label"].astype(str).to_numpy())
    np.save(out / "segment_ids.npy", anchors["segment_id"].astype(str).to_numpy())
    np.save(out / "segment_names.npy", anchors["seg_name"].astype(str).to_numpy())

    anchor_cols = [
        "anchor_id",
        "patient_id",
        "segment_id",
        "seg_name",
        "window_time",
        "split_label",
        "segment_path",
        "overlapping_vasopressor_count",
        "first_overlapping_vasopressor_time",
        "last_overlapping_vasopressor_time",
        "first_overlapping_vasopressor_name",
        "vasopressor_source",
        "vasopressor_match_method",
        "classification_status",
    ]
    anchors[anchor_cols].to_csv(out / "anchors.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    metadata = {
        "cache_kind": "full_data_raw_waveform_anchor_cache",
        "cohort": "confirmed_vasopressor_overlap_segments",
        "selection_rule": "has_vasopressor_overlap == True; uncertain/NA rows excluded",
        "is_full_data_segment_window_cache": True,
        "is_raw_waveform_anchor_only_cache": True,
        "full_data_root": str(args.full_data_root),
        "manifest": str(args.manifest),
        "n_samples": int(n),
        "expected_full_n_samples": int(full_count),
        "n_segments": int(anchors["segment_id"].nunique()),
        "n_patients": int(anchors["patient_id"].nunique()),
        "split_sample_counts": count_labels(anchors["split_label"]),
        "split_patient_counts": count_labels(anchors.drop_duplicates("patient_id")["split_label"]),
        "channel_order": list(CHANNEL_ORDER),
        "feature_names": [],
        "n_feature_windows": 0,
        "feature_count": 0,
        "anchor_time_basis": "segment-relative window_time from full_data_root/window_times.npy",
        "segment_path_source": "manifest segment_path",
        "created_at_unix": time.time(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    (out / "feature_quality_report.json").write_text(json.dumps({}, indent=2))
    (out / "_SUCCESS").write_text(f"completed_at_unix={time.time():.6f}\n")
    print(json.dumps({"output_dir": str(out), "n_samples": int(n), "n_segments": int(anchors["segment_id"].nunique()), "n_patients": int(anchors["patient_id"].nunique()), "split_sample_counts": metadata["split_sample_counts"]}, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    manifest = load_confirmed_overlap_segments(args.manifest)
    anchors = build_anchor_table(args.full_data_root, manifest)
    full_count = len(anchors)
    anchors = apply_splits(anchors, args.full_data_root / "patient_splits.json")
    if args.max_samples is not None:
        anchors = anchors.iloc[: args.max_samples].copy()
    write_cache(anchors, args, full_count)


if __name__ == "__main__":
    main()
