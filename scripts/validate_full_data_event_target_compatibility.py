#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.extract_full_data_numerics import NUMERICS_SAMPLES_PER_WINDOW, VITAL_NAMES, extract_window
from scripts.build_full_data_event_targets import build_numerics_index, find_overlapping_record
from waveform_baselines.target_builders import build_event_targets
from waveform_baselines.task_specs import DEFAULT_EVENT_TASK, EventTaskSpec

DEFAULT_ANCHORS = Path("outputs/targets/vasopressor_free_overlap_anchors.csv")
DEFAULT_TARGETS = Path("outputs/targets/event_targets_vasopressor_free_anchor_horizon_filtered_5m_10m.npz")
DEFAULT_WAVEFORM_ROOT = Path("/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate direct waveform-numerics event labels against the prior saved classification target bundle.")
    parser.add_argument("--anchors-csv", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--target-bundle", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--waveform-root", type=Path, default=DEFAULT_WAVEFORM_ROOT)
    parser.add_argument("--max-patients", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/full_data_event_target_compat"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchors = pd.read_csv(args.anchors_csv)
    targets = np.load(args.target_bundle, allow_pickle=True)
    old_targets = targets["event_targets"]
    old_mask = targets["event_mask"]

    patients = anchors["patient_id"].astype(str).drop_duplicates().head(args.max_patients)
    subset = anchors[anchors["patient_id"].astype(str).isin(set(patients))].copy()
    subset["segment_id"] = subset["patient_id"].astype(str)
    subset["anchor_id"] = subset.index.astype(np.int64)
    subset["seg_name"] = subset["segment_id"]

    args.work_dir.mkdir(parents=True, exist_ok=True)
    rows = subset.index.to_numpy(dtype=np.int64)

    legacy_root = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/output_v2")
    legacy_pids = np.load(legacy_root / "numerics_patient_ids.npy", mmap_mode="r", allow_pickle=True).astype(str)
    legacy_times = np.asarray(np.load(legacy_root / "numerics_window_times.npy", mmap_mode="r"), dtype=np.float64)
    legacy_x = np.load(legacy_root / "X_numerics.npy", mmap_mode="r")
    legacy_index = {(pid, float(time)): idx for idx, (pid, time) in enumerate(zip(legacy_pids.tolist(), legacy_times.tolist()))}

    numerics_index = build_numerics_index(args.waveform_root, set(subset["patient_id"].astype(str)))
    x = np.lib.format.open_memmap(args.work_dir / "X_numerics.npy", mode="w+", dtype="float32", shape=(len(subset), len(VITAL_NAMES), NUMERICS_SAMPLES_PER_WINDOW))
    pids = np.lib.format.open_memmap(args.work_dir / "numerics_patient_ids.npy", mode="w+", dtype="<U12", shape=(len(subset),))
    times = np.lib.format.open_memmap(args.work_dir / "numerics_window_times.npy", mode="w+", dtype="float64", shape=(len(subset),))

    numerics_mismatch_rows = 0
    max_abs_diff = 0.0
    missing_legacy_rows = 0
    for out_idx, row in enumerate(subset.itertuples(index=False)):
        anchor_time = float(row.anchor_time)
        records = numerics_index.get(str(row.patient_id), [])
        record = find_overlapping_record(records, float(row.input_start_time), float(row.input_end_time) + 15 * 60.0)
        window = extract_window(record, anchor_time)
        x[out_idx] = window
        pids[out_idx] = str(row.patient_id)
        times[out_idx] = anchor_time
        legacy_idx = legacy_index.get((str(row.patient_id), anchor_time))
        if legacy_idx is None:
            missing_legacy_rows += 1
        else:
            ref = np.asarray(legacy_x[legacy_idx], dtype=np.float32)
            equal = np.allclose(window, ref, equal_nan=True, atol=0.0, rtol=0.0)
            if not equal:
                numerics_mismatch_rows += 1
                finite = np.isfinite(window) & np.isfinite(ref)
                if np.any(finite):
                    max_abs_diff = max(max_abs_diff, float(np.max(np.abs(window[finite] - ref[finite]))))
        if args.progress_every and (out_idx + 1) % args.progress_every == 0:
            print(json.dumps({"event": "extract_progress", "rows_done": out_idx + 1, "rows_total": len(subset)}), flush=True)
    del x, pids, times

    spec = EventTaskSpec(
        horizons_min=(5, 10),
        target_generation_mode="anchor_horizon_filtered",
        hypotension_threshold=DEFAULT_EVENT_TASK.hypotension_threshold,
        tachycardia_threshold=DEFAULT_EVENT_TASK.tachycardia_threshold,
        sustain_minutes=DEFAULT_EVENT_TASK.sustain_minutes,
        hypotension_channel=DEFAULT_EVENT_TASK.hypotension_channel,
        tachycardia_channel=DEFAULT_EVENT_TASK.tachycardia_channel,
        event_names=DEFAULT_EVENT_TASK.event_names,
    )
    result = build_event_targets(subset, args.work_dir, task_spec=spec)
    cmp_targets = result.targets
    cmp_mask = result.mask
    ref_targets = old_targets[rows]
    ref_mask = old_mask[rows]
    mask_mismatch = cmp_mask != ref_mask
    value_mismatch = (cmp_targets != ref_targets) & cmp_mask & ref_mask
    summary = {
        "patients_checked": int(len(patients)),
        "rows_checked": int(len(rows)),
        "numerics_missing_legacy_rows": int(missing_legacy_rows),
        "numerics_mismatch_rows": int(numerics_mismatch_rows),
        "numerics_max_abs_diff": float(max_abs_diff),
        "mask_mismatches": int(mask_mismatch.sum()),
        "value_mismatches_on_joint_valid": int(value_mismatch.sum()),
        "old_valid_values": int(ref_mask.sum()),
        "new_valid_values": int(cmp_mask.sum()),
        "diagnostics": result.diagnostics,
    }
    print(json.dumps(summary, indent=2))
    if summary["mask_mismatches"] or summary["value_mismatches_on_joint_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
