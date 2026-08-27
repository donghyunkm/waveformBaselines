#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SEGMENTS_PATH = Path("/gpfs/home/dk5565/icuDataExtraction/parts_v2/selected_segments.json")
WAVEFORM_DIR = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms")
VASOPRESSOR_FREE_MANIFEST = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/manifests/"
    "hypotension_subject_split_vasopressor_free_stays_v1.csv"
)
OUTPUT_PATH = Path("/gpfs/home/dk5565/waveformBaselines/outputs/splits/vasopressor_free_splits.json")


def load_available_patients(segments_path: Path, waveform_dir: Path) -> set[str]:
    index = json.loads(segments_path.read_text())
    indexed = set(index["patients"])
    meta = json.loads((waveform_dir / "metadata.json").read_text())
    extracted = set(meta["patients"])
    return indexed & extracted


def build_splits(
    manifest_path: Path = VASOPRESSOR_FREE_MANIFEST,
    segments_path: Path = SEGMENTS_PATH,
    waveform_dir: Path = WAVEFORM_DIR,
    output_path: Path = OUTPUT_PATH,
) -> dict:
    available_patients = load_available_patients(segments_path, waveform_dir)

    splits = {"train": [], "val": [], "test": []}
    manifest_counts = {"train": 0, "val": 0, "test": 0}
    dropped_patients: list[str] = []

    with manifest_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = row["subject_id"]
            split = row["split"]
            if split not in splits:
                continue
            manifest_counts[split] += 1
            if patient_id in available_patients:
                splits[split].append(patient_id)
            else:
                dropped_patients.append(patient_id)

    meta = json.loads((waveform_dir / "metadata.json").read_text())
    patient_meta = meta["patients"]

    stats = {}
    for split_name, patient_ids in splits.items():
        unique_ids = sorted(set(patient_ids))
        splits[split_name] = unique_ids
        n_windows = sum(int(patient_meta[pid]["n_anchors"]) for pid in unique_ids)
        stats[split_name] = {
            "n_patients": len(unique_ids),
            "n_windows": n_windows,
            "manifest_patients": manifest_counts[split_name],
            "dropped_missing_waveforms": manifest_counts[split_name] - len(unique_ids),
        }

    output = {
        "source": str(manifest_path),
        "segments_source": str(segments_path),
        "waveform_dir": str(waveform_dir),
        "cohort": "vasopressor_free_physiojepa_overlap",
        "cohort_note": (
            "Subset of the PhysioJEPA vasopressor-free cohort that overlaps this repo's "
            "20-minute waveform baseline extraction."
        ),
        "stats": stats,
        "n_available_patients": len(available_patients),
        "n_manifest_patients_total": sum(manifest_counts.values()),
        "n_overlap_patients_total": sum(len(v) for v in splits.values()),
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    dropped_path = output_path.with_name(output_path.stem + "_dropped_patients.json")
    dropped_path.write_text(json.dumps(sorted(set(dropped_patients)), indent=2))

    print(f"Saved splits to {output_path}")
    for split_name, split_stats in stats.items():
        print(
            f"  {split_name}: {split_stats['n_patients']} patients, "
            f"{split_stats['n_windows']:,} windows "
            f"(dropped {split_stats['dropped_missing_waveforms']} manifest patients)"
        )
    print(f"Dropped manifest patients list: {dropped_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vasopressor-free patient splits for waveform baselines")
    parser.add_argument("--manifest-path", type=Path, default=VASOPRESSOR_FREE_MANIFEST)
    parser.add_argument("--segments-path", type=Path, default=SEGMENTS_PATH)
    parser.add_argument("--waveform-dir", type=Path, default=WAVEFORM_DIR)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    build_splits(
        manifest_path=args.manifest_path,
        segments_path=args.segments_path,
        waveform_dir=args.waveform_dir,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
