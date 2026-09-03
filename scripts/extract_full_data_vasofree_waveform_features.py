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

try:
    import wfdb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wfdb required: pip install wfdb") from exc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waveform_baselines.wf_features import CACHE_ROOT, DEFAULT_EXTRACTION_CONFIG  # noqa: E402
from waveform_baselines.wf_features.definitions import FEATURE_DEFINITIONS, feature_names  # noqa: E402
from waveform_baselines.wf_features.pipeline import extract_feature_sequence  # noqa: E402


DEFAULT_FULL_DATA_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/physionet_restricted/mimic_derived_data/data_m3_120s_prediction")
DEFAULT_MANIFEST = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/"
    "full_data_segment_level_vasopressor_free_waveform_manifest.csv"
)
DEFAULT_WAVEFORM_ROOT = Path("/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched")
DEFAULT_OUTPUT_NAME = "full_data_vasopressor_free_waveform_features_v7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract v7 waveform features for full-data windows filtered by the segment-level vasopressor-free manifest.")
    parser.add_argument("--full-data-root", type=Path, default=DEFAULT_FULL_DATA_ROOT)
    parser.add_argument("--vasopressor-free-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--waveform-root", type=Path, default=DEFAULT_WAVEFORM_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
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


def load_free_segment_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, usecols=["segment_id", "segment_path", "vasopressor_free"])
    free = manifest[manifest["vasopressor_free"].astype(str).str.lower().eq("true")].copy()
    if free.empty:
        raise ValueError(f"No vasopressor_free=True segments found in {path}")
    if free["segment_id"].duplicated().any():
        dupes = free.loc[free["segment_id"].duplicated(), "segment_id"].head(5).tolist()
        raise ValueError(f"Duplicate free segment_id values in {path}: {dupes}")
    return free


def build_full_data_anchor_table(full_data_root: Path, manifest_path: Path) -> pd.DataFrame:
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
    free_segments = load_free_segment_manifest(manifest_path)
    anchors = anchors.merge(free_segments, on="segment_id", how="inner", validate="many_to_one")
    if anchors.empty:
        raise ValueError("No full-data windows matched vasopressor-free segment manifest")
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


def shard_anchor_table(anchors: pd.DataFrame, shard_index: int | None, shard_count: int | None) -> pd.DataFrame:
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard-index and shard-count must be provided together")
    if shard_count is None:
        return anchors
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard {shard_index}/{shard_count}")
    boundaries = np.linspace(0, len(anchors), int(shard_count) + 1, dtype=int)
    start = int(boundaries[int(shard_index)])
    end = int(boundaries[int(shard_index) + 1])
    shard = anchors.iloc[start:end].reset_index(drop=True)
    if shard.empty:
        raise ValueError(f"Shard {shard_index}/{shard_count} has no anchors")
    return shard


class SegmentReader:
    def __init__(self, channel_order: tuple[str, ...]) -> None:
        self.channel_order = channel_order
        self.current_segment_id: str | None = None
        self.current_data: np.ndarray | None = None
        self.current_fs: float | None = None

    def load(self, row: pd.Series) -> tuple[np.ndarray, float]:
        segment_id = str(row["segment_id"])
        if segment_id == self.current_segment_id and self.current_data is not None and self.current_fs is not None:
            return self.current_data, self.current_fs
        rec = wfdb.rdrecord(str(row["segment_path"]))
        missing = [channel for channel in self.channel_order if channel not in rec.sig_name]
        if missing:
            raise ValueError(f"Segment {segment_id} missing required channels {missing}; available={rec.sig_name}")
        indices = [rec.sig_name.index(channel) for channel in self.channel_order]
        data = np.asarray(rec.p_signal[:, indices].T, dtype=np.float32)
        self.current_segment_id = segment_id
        self.current_data = data
        self.current_fs = float(rec.fs)
        return data, float(rec.fs)


def extract_window(row: pd.Series, reader: SegmentReader, input_samples: int, fs: int) -> np.ndarray:
    segment, segment_fs = reader.load(row)
    if int(round(segment_fs)) != fs:
        raise ValueError(f"Segment {row['segment_id']} fs={segment_fs} does not match expected {fs}")
    anchor_sample = int(round(float(row["window_time"]) * fs))
    start = anchor_sample - input_samples // 2
    end = start + input_samples
    if start < 0 or end > segment.shape[1]:
        raise IndexError(f"Window anchor_id={row['anchor_id']} segment={row['segment_id']} sample [{start}, {end}) outside length {segment.shape[1]}")
    return np.asarray(segment[:, start:end], dtype=np.float32)


def count_labels(values: np.ndarray) -> dict[str, int]:
    counts = Counter(np.asarray(values, dtype=object).tolist())
    return {str(k): int(v) for k, v in sorted(counts.items())}


def feature_stats(values: np.ndarray, mask: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        valid = mask[:, :, idx] & np.isfinite(values[:, :, idx])
        arr = values[:, :, idx][valid].astype(np.float64, copy=False)
        if arr.size == 0:
            out[name] = {"count": 0, "missing_fraction": 1.0, "non_finite_fraction": 1.0}
            continue
        out[name] = {
            "count": int(arr.size),
            "missing_fraction": float(1.0 - np.mean(valid)),
            "non_finite_fraction": float(np.mean(~np.isfinite(values[:, :, idx]))),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p05": float(np.percentile(arr, 5.0)),
            "p95": float(np.percentile(arr, 95.0)),
        }
    return out


def main() -> None:
    args = parse_args()
    config = DEFAULT_EXTRACTION_CONFIG
    cache_dir = args.cache_root / config.feature_version / args.output_name
    if cache_dir.exists() and any(cache_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{cache_dir} exists and is nonempty; pass --overwrite")

    anchors = build_full_data_anchor_table(args.full_data_root, args.vasopressor_free_manifest)
    expected_full_n_samples = int(len(anchors))
    anchors = apply_splits(anchors, args.full_data_root / "patient_splits.json")
    if args.max_samples is not None:
        anchors = anchors.iloc[: args.max_samples].copy()
    anchors = shard_anchor_table(anchors, args.shard_index, args.shard_count)

    feature_order = feature_names()
    n_samples = int(len(anchors))
    values = np.full((n_samples, config.n_feature_windows, len(feature_order)), np.nan, dtype=np.float32)
    mask = np.zeros((n_samples, config.n_feature_windows, len(feature_order)), dtype=bool)
    diagnostics: dict[str, int] = {}
    reader = SegmentReader(config.channel_order)

    print(json.dumps({
        "event": "full_data_vasofree_feature_extraction_start",
        "cache_dir": str(cache_dir),
        "n_samples": n_samples,
        "expected_full_n_samples": expected_full_n_samples,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }, sort_keys=True), flush=True)

    for out_idx, row in enumerate(anchors.itertuples(index=False)):
        row_series = pd.Series(row._asdict())
        waveform = extract_window(row_series, reader, config.input_samples, config.sampling_rate_hz)
        seq_values, seq_mask, _ = extract_feature_sequence(waveform, config=config, diagnostics=diagnostics)
        values[out_idx] = seq_values
        mask[out_idx] = seq_mask
        if (out_idx + 1) % 1000 == 0:
            print(json.dumps({"event": "progress", "processed": out_idx + 1, "n_samples": n_samples}, sort_keys=True), flush=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "values.npy", values)
    np.save(cache_dir / "mask.npy", mask)
    np.save(cache_dir / "patient_ids.npy", anchors["patient_id"].astype(str).to_numpy())
    np.save(cache_dir / "anchor_times.npy", anchors["window_time"].to_numpy(dtype=np.float64))
    np.save(cache_dir / "anchor_ids.npy", anchors["anchor_id"].to_numpy(dtype=np.int64))
    np.save(cache_dir / "split_labels.npy", anchors["split_label"].astype(str).to_numpy())
    np.save(cache_dir / "segment_ids.npy", anchors["segment_id"].astype(str).to_numpy())
    np.save(cache_dir / "segment_names.npy", anchors["seg_name"].astype(str).to_numpy())
    anchors[["anchor_id", "patient_id", "segment_id", "seg_name", "window_time", "split_label", "segment_path"]].to_csv(cache_dir / "anchors.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    metadata = {
        "feature_version": config.feature_version,
        "extraction_config": config.to_dict(),
        "channel_order": list(config.channel_order),
        "feature_names": feature_order,
        "feature_units": {feature.name: feature.unit for feature in FEATURE_DEFINITIONS},
        "feature_descriptions": {feature.name: feature.description for feature in FEATURE_DEFINITIONS},
        "full_data_root": str(args.full_data_root),
        "vasopressor_free_manifest": str(args.vasopressor_free_manifest),
        "waveform_root": str(args.waveform_root),
        "n_samples": n_samples,
        "expected_full_n_samples": expected_full_n_samples,
        "split_sample_counts": count_labels(anchors["split_label"].to_numpy()),
        "split_patient_counts": count_labels(anchors.drop_duplicates("patient_id")["split_label"].to_numpy()),
        "extraction_diagnostics": diagnostics,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_assignment": "contiguous_sorted_anchor_id_chunks",
        "is_full_data_segment_window_cache": True,
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (cache_dir / "feature_quality_report.json").write_text(json.dumps(feature_stats(values, mask, feature_order), indent=2))
    (cache_dir / "_SUCCESS").write_text(f"completed_at_unix={time.time():.6f}\n")
    print(json.dumps({"cache_dir": str(cache_dir), "shape": list(values.shape), "n_samples": n_samples}, indent=2), flush=True)


if __name__ == "__main__":
    main()
