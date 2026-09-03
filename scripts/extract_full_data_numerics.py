#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_full_data_event_targets import DEFAULT_CACHE_DIR, DEFAULT_FULL_DATA_ROOT, DEFAULT_WAVEFORM_ROOT, NumericsRecord, VITAL_ALIASES, VITAL_NAMES, build_numerics_index, find_overlapping_record, load_cache_anchors

DEFAULT_OUTPUT_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/numerics/full_data_v1")
NUMERICS_SAMPLES_PER_WINDOW = 1200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract full-data bedside-monitor numerics windows aligned to the full-data v7 feature cache.")
    parser.add_argument("--full-data-root", type=Path, default=DEFAULT_FULL_DATA_ROOT)
    parser.add_argument("--feature-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--waveform-root", type=Path, default=DEFAULT_WAVEFORM_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--job-idx", type=int, default=0)
    parser.add_argument("--num-jobs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def extract_window(record: NumericsRecord | None, anchor_time: float) -> np.ndarray:
    out = np.full((len(VITAL_NAMES), NUMERICS_SAMPLES_PER_WINDOW), np.nan, dtype=np.float32)
    if record is None:
        return out
    try:
        data = wfdb.rdrecord(str(record.path)).p_signal
    except Exception:
        return out
    ctx_start = anchor_time - NUMERICS_SAMPLES_PER_WINDOW / 2.0
    sample_start = int((ctx_start - record.start_epoch) * record.fs)
    sample_end = sample_start + NUMERICS_SAMPLES_PER_WINDOW
    valid_start = max(0, sample_start)
    valid_end = min(record.sig_len, sample_end)
    if valid_end <= valid_start:
        return out
    out_offset = valid_start - sample_start
    n_valid = valid_end - valid_start
    for vital_idx, col_idx in record.sig_indices.items():
        if col_idx is not None:
            out[vital_idx, out_offset : out_offset + n_valid] = data[valid_start:valid_end, col_idx]
    return out


def main() -> None:
    args = parse_args()
    if args.job_idx < 0 or args.job_idx >= args.num_jobs:
        raise ValueError("job-idx must be in [0, num-jobs)")
    anchors = load_cache_anchors(args.feature_cache_dir, args.full_data_root).sort_values("anchor_id").reset_index(drop=True)
    start = args.job_idx * len(anchors) // args.num_jobs
    end = (args.job_idx + 1) * len(anchors) // args.num_jobs
    shard = anchors.iloc[start:end].copy()
    part_dir = args.output_root / "parts" / f"numerics_part_{args.job_idx:03d}"
    part_dir.mkdir(parents=True, exist_ok=True)
    done = part_dir / "_SUCCESS"
    if done.exists():
        print(json.dumps({"event": "already_complete", "part_dir": str(part_dir)}))
        return

    numerics_index = build_numerics_index(args.waveform_root, set(shard["patient_id"].astype(str)))
    x = np.lib.format.open_memmap(part_dir / "X_numerics.npy", mode="w+", dtype="float32", shape=(len(shard), len(VITAL_NAMES), NUMERICS_SAMPLES_PER_WINDOW))
    patient_ids = np.lib.format.open_memmap(part_dir / "numerics_patient_ids.npy", mode="w+", dtype="<U12", shape=(len(shard),))
    seg_names = np.lib.format.open_memmap(part_dir / "numerics_seg_names.npy", mode="w+", dtype="<U64", shape=(len(shard),))
    window_times = np.lib.format.open_memmap(part_dir / "numerics_window_times.npy", mode="w+", dtype="float64", shape=(len(shard),))
    anchor_ids = np.lib.format.open_memmap(part_dir / "anchor_ids.npy", mode="w+", dtype="int64", shape=(len(shard),))

    record_cache: dict[str, NumericsRecord | None] = {}
    missing_records = 0
    for out_idx, row in enumerate(shard.itertuples(index=False)):
        segment_id = str(row.segment_id)
        record = record_cache.get(segment_id)
        if segment_id not in record_cache:
            records = numerics_index.get(str(row.patient_id), [])
            needed_start = float(row.input_start_time)
            needed_end = float(row.input_end_time) + 15 * 60.0
            record = find_overlapping_record(records, needed_start, needed_end)
            record_cache[segment_id] = record
            if record is None:
                missing_records += 1
        x[out_idx] = extract_window(record, float(row.absolute_anchor_time))
        patient_ids[out_idx] = str(row.patient_id)
        seg_names[out_idx] = str(row.seg_name)
        window_times[out_idx] = float(row.absolute_anchor_time)
        anchor_ids[out_idx] = int(row.anchor_id)
        if args.progress_every and (out_idx + 1) % args.progress_every == 0:
            print(json.dumps({"event": "progress", "rows_done": out_idx + 1, "rows_total": len(shard)}), flush=True)

    del x, patient_ids, seg_names, window_times, anchor_ids
    metadata = {
        "job_idx": args.job_idx,
        "num_jobs": args.num_jobs,
        "global_start": int(start),
        "global_end": int(end),
        "rows": int(len(shard)),
        "vital_names": VITAL_NAMES,
        "vital_aliases": VITAL_ALIASES,
        "source": "MIMIC-III matched waveform bedside monitor numerics records via RECORDS-numerics",
        "samples_per_window": NUMERICS_SAMPLES_PER_WINDOW,
        "sampling_rate_hz": 1.0,
        "window_time_reference": "center",
        "window_time_basis": "absolute",
        "missing_segment_records": int(missing_records),
    }
    (part_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    done.write_text(json.dumps({"rows": int(len(shard))}))
    print(json.dumps({"event": "complete", "part_dir": str(part_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
