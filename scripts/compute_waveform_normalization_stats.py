#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from waveform_baselines.normalization import (
    compute_training_channel_stats,
    save_training_channel_stats,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute train-split waveform normalization stats shared across patients."
    )
    parser.add_argument(
        "--waveform-dir",
        type=Path,
        default=Path("/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms"),
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=Path("outputs/splits/splits.json"),
    )
    parser.add_argument(
        "--clip-lower-percentile",
        type=float,
        default=None,
        help="Optional lower percentile for training-set clipping before mean/std.",
    )
    parser.add_argument(
        "--clip-upper-percentile",
        type=float,
        default=None,
        help="Optional upper percentile for training-set clipping before mean/std.",
    )
    parser.add_argument(
        "--clip-sample-size",
        type=int,
        default=1_000_000,
        help="Reservoir size per channel for percentile estimation when clipping is enabled.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = compute_training_channel_stats(
        waveform_dir=args.waveform_dir,
        splits_path=args.splits_path,
        clip_lower_percentile=args.clip_lower_percentile,
        clip_upper_percentile=args.clip_upper_percentile,
        clip_sample_size=args.clip_sample_size,
        seed=args.seed,
    )
    out_path = save_training_channel_stats(
        waveform_dir=args.waveform_dir,
        splits_path=args.splits_path,
        stats=stats,
    )

    print(f"Saved normalization stats to {out_path}", flush=True)
    print(f"Method: {stats['method']}", flush=True)
    print(f"Training split source: {stats['splits_path']}", flush=True)
    for channel, channel_stats in stats["channels"].items():
        print(
            f"  {channel}: mean={channel_stats['mean']:.6f} "
            f"std={channel_stats['std']:.6f} count={channel_stats['count']:,} "
            f"excluded_nonfinite={channel_stats['n_excluded_nonfinite']:,} "
            f"excluded_frac={channel_stats['fraction_excluded_nonfinite']:.6e}",
            flush=True,
        )
    if stats["clipping"] is not None:
        print("Clipping:", stats["clipping"], flush=True)


if __name__ == "__main__":
    main()
