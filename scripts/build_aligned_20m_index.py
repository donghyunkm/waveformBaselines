#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from waveform_baselines.data_index import build_aligned_20m_anchor_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical aligned 20-minute waveform anchor table.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Destination CSV for aligned anchors.")
    parser.add_argument("--raw-root", type=Path, default=None, help="Override raw waveform root.")
    parser.add_argument("--icu-output-dir", type=Path, default=None, help="Override icuDataExtraction/output_v2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {}
    if args.raw_root is not None:
        kwargs["raw_root"] = args.raw_root
    if args.icu_output_dir is not None:
        kwargs["icu_output_dir"] = args.icu_output_dir

    anchors = build_aligned_20m_anchor_table(**kwargs)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(args.output_csv, index=False)
    print(f"Saved {len(anchors):,} aligned 20-minute anchors to {args.output_csv}")


if __name__ == "__main__":
    main()
