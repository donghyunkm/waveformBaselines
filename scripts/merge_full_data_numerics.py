#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scripts.extract_full_data_numerics import DEFAULT_OUTPUT_ROOT, NUMERICS_SAMPLES_PER_WINDOW, VITAL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge full-data numerics shards into row-aligned arrays for event target generation.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parts = []
    for part in sorted((args.output_root / "parts").glob("numerics_part_*")):
        if not (part / "_SUCCESS").exists():
            continue
        meta = json.loads((part / "metadata.json").read_text())
        parts.append((part, meta))
    if not parts:
        raise FileNotFoundError(f"No completed numerics parts found under {args.output_root / 'parts'}")
    parts.sort(key=lambda item: int(item[1]["global_start"]))
    total = sum(int(meta["rows"]) for _, meta in parts)
    args.output_root.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(args.output_root / "X_numerics.npy", mode="w+", dtype="float32", shape=(total, len(VITAL_NAMES), NUMERICS_SAMPLES_PER_WINDOW))
    pids = np.lib.format.open_memmap(args.output_root / "numerics_patient_ids.npy", mode="w+", dtype="<U12", shape=(total,))
    segs = np.lib.format.open_memmap(args.output_root / "numerics_seg_names.npy", mode="w+", dtype="<U64", shape=(total,))
    times = np.lib.format.open_memmap(args.output_root / "numerics_window_times.npy", mode="w+", dtype="float64", shape=(total,))
    anchor_ids = np.lib.format.open_memmap(args.output_root / "anchor_ids.npy", mode="w+", dtype="int64", shape=(total,))
    cursor = 0
    for part, meta in parts:
        n = int(meta["rows"])
        x[cursor:cursor+n] = np.load(part / "X_numerics.npy", mmap_mode="r")[:n]
        pids[cursor:cursor+n] = np.load(part / "numerics_patient_ids.npy", mmap_mode="r", allow_pickle=True)[:n]
        segs[cursor:cursor+n] = np.load(part / "numerics_seg_names.npy", mmap_mode="r", allow_pickle=True)[:n]
        times[cursor:cursor+n] = np.load(part / "numerics_window_times.npy", mmap_mode="r")[:n]
        anchor_ids[cursor:cursor+n] = np.load(part / "anchor_ids.npy", mmap_mode="r")[:n]
        cursor += n
    del x, pids, segs, times, anchor_ids
    metadata = {
        "rows": int(total),
        "n_parts": int(len(parts)),
        "vital_names": VITAL_NAMES,
        "samples_per_window": NUMERICS_SAMPLES_PER_WINDOW,
        "sampling_rate_hz": 1.0,
        "window_time_reference": "center",
        "window_time_basis": "absolute",
        "source": "Merged full-data bedside monitor numerics shards aligned to full-data feature-cache anchor order.",
    }
    (args.output_root / "numerics_metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.output_root / "n_windows.txt").write_text(str(total))
    (args.output_root / "_SUCCESS").write_text(json.dumps({"rows": int(total)}))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
