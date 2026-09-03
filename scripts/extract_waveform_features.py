#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waveform_baselines.wf_features import CACHE_ROOT, DEFAULT_EXTRACTION_CONFIG, FeatureCacheBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline waveform-derived feature cache.")
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--splits-path", type=Path, required=True)
    parser.add_argument("--target-bundle-path", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = FeatureCacheBuilder(config=DEFAULT_EXTRACTION_CONFIG, cache_root=args.cache_root)
    cache = builder.build(
        output_name=args.output_name,
        splits_path=args.splits_path,
        target_bundle_path=args.target_bundle_path,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(json.dumps({
        "cache_dir": str(cache.cache_dir),
        "n_samples": int(cache.values.shape[0]),
        "shape": list(cache.values.shape),
        "feature_dim": int(cache.values.shape[2]),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }, indent=2))


if __name__ == "__main__":
    main()
