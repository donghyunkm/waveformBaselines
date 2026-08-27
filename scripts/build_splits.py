"""
Generate patient-level train/val/test splits for the waveform baselines.

Uses the selected_segments.json index from icuDataExtraction to determine
the patient population, then splits 70/15/15 at the patient level.

Output: outputs/splits/splits.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ── Defaults ──────────────────────────────────────────────────────────────────

SEGMENTS_PATH = Path("/gpfs/home/dk5565/icuDataExtraction/parts_v2/selected_segments.json")
OUTPUT_DIR = Path("/gpfs/home/dk5565/waveformBaselines/outputs/splits")

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42


def build_splits(
    segments_path: Path = SEGMENTS_PATH,
    output_dir: Path = OUTPUT_DIR,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = SEED,
) -> dict:
    """Build patient-level splits and save to JSON."""
    index = json.loads(segments_path.read_text())
    patients = sorted(index["patients"])
    n = len(patients)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    splits = {
        "train": sorted([patients[i] for i in train_idx]),
        "val": sorted([patients[i] for i in val_idx]),
        "test": sorted([patients[i] for i in test_idx]),
    }

    # Compute window counts per patient for diagnostics
    segs = index["segs"]
    half_ctx = 75000
    anchor_stride = 18750
    min_windows = 100

    def count_windows(patient_id: str) -> int:
        best = 0
        for _, _, sig_len, _ in segs[patient_id]:
            first = half_ctx
            last = sig_len - half_ctx
            if last < first:
                continue
            n_anchors = len(range(first, last + 1, anchor_stride))
            if n_anchors >= min_windows and n_anchors > best:
                best = n_anchors
        return best

    window_counts = {pid: count_windows(pid) for pid in patients}

    stats = {}
    for split_name, split_pids in splits.items():
        windows = [window_counts[pid] for pid in split_pids]
        stats[split_name] = {
            "n_patients": len(split_pids),
            "n_windows": sum(windows),
            "mean_windows_per_patient": float(np.mean(windows)) if windows else 0,
        }

    output = {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": 1.0 - train_frac - val_frac,
        "source": str(segments_path),
        "stats": stats,
        **splits,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "splits.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Splits saved to {out_path}")
    for split_name, split_stats in stats.items():
        print(f"  {split_name}: {split_stats['n_patients']} patients, "
              f"{split_stats['n_windows']:,} windows")

    return output


def main():
    parser = argparse.ArgumentParser(description="Build patient-level splits")
    parser.add_argument("--segments-path", type=Path, default=SEGMENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    build_splits(
        segments_path=args.segments_path,
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
