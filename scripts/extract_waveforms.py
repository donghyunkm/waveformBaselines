#!/usr/bin/env python3
"""
Pre-extract WFDB waveform segments to per-patient numpy files.

Converts each patient's best WFDB segment to a .npy file with shape
`(n_channels, sig_len)` for a configurable subset of waveform channels in float32.

Also saves a metadata JSON with per-patient stats (for normalization at training time)
and anchor positions.

Output structure:
    <output_dir>/
        <patient_id>.npy          # shape (3, sig_len), float32
    <output_dir>/metadata.json    # patient metadata

Usage:
    python scripts/extract_waveforms.py [--workers 8] [--output-dir ...]
    python scripts/extract_waveforms.py --channels II,PLETH,ABP,RESP --output-dir ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import wfdb
except ImportError as e:
    raise ImportError("wfdb required: pip install wfdb") from e

# ── Constants ─────────────────────────────────────────────────────────────────

FS = 125
CTX_SAMPLES = 150_000  # 20 min at 125 Hz; used for anchor eligibility
HALF_CTX = CTX_SAMPLES // 2
ANCHOR_STRIDE = int(2.5 * 60 * FS)  # 18750 samples = 150 s
MIN_WINDOWS_PER_SEGMENT = 100

# Signal order in WFDB records
RAW_SIGNAL_ORDER = ("II", "ABP", "PLETH", "RESP")
# Default extracted channels for backward compatibility
DEFAULT_MODEL_CHANNELS = ("ABP", "II", "PLETH")

SEGMENTS_PATH = Path("/gpfs/home/dk5565/icuDataExtraction/parts_v2/selected_segments.json")
DEFAULT_OUTPUT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms")


def compute_anchors(sig_len: int) -> list[int]:
    """Compute anchor center positions for a segment."""
    first = HALF_CTX
    last = sig_len - HALF_CTX
    if last < first:
        return []
    anchors = list(range(first, last + 1, ANCHOR_STRIDE))
    if len(anchors) < MIN_WINDOWS_PER_SEGMENT:
        return []
    return anchors


def find_best_segment(entries: list) -> tuple | None:
    """Find the segment with the most valid windows."""
    best = None
    best_n_anchors = 0
    for entry in entries:
        stay_dir, seg_name, sig_len, seg_start_secs = entry
        anchors = compute_anchors(sig_len)
        if len(anchors) > best_n_anchors:
            best = (stay_dir, seg_name, sig_len, seg_start_secs)
            best_n_anchors = len(anchors)
    return best


def parse_channels(channel_text: str) -> tuple[str, ...]:
    channels = tuple(part.strip() for part in channel_text.split(",") if part.strip())
    if not channels:
        raise ValueError("At least one channel must be specified.")
    invalid = [ch for ch in channels if ch not in RAW_SIGNAL_ORDER]
    if invalid:
        raise ValueError(
            f"Unknown channels {invalid}. Valid channels: {RAW_SIGNAL_ORDER}"
        )
    if len(set(channels)) != len(channels):
        raise ValueError("Duplicate channels are not allowed.")
    return channels


def extract_patient(
    pid: str,
    seg_info: tuple,
    output_dir: Path,
    model_channels: tuple[str, ...],
) -> dict:
    """
    Extract one patient's segment to a .npy file.
    
    Returns metadata dict or None on failure.
    """
    stay_dir, seg_name, sig_len, seg_start_secs = seg_info
    out_path = output_dir / f"{pid}.npy"
    
    n_channels = len(model_channels)

    # Skip if already extracted
    if out_path.exists():
        # Verify shape
        try:
            arr = np.load(out_path, mmap_mode='r')
            if arr.shape == (n_channels, sig_len):
                anchors = compute_anchors(sig_len)
                # Compute per-channel stats from the file
                stats = []
                for ch in range(n_channels):
                    ch_data = np.array(arr[ch])
                    valid = ch_data[np.isfinite(ch_data)]
                    if valid.size > 0:
                        stats.append({"mean": float(valid.mean()), "std": float(valid.std())})
                    else:
                        stats.append({"mean": 0.0, "std": 1.0})
                return {
                    "patient_id": pid,
                    "sig_len": sig_len,
                    "seg_start_secs": seg_start_secs,
                    "n_anchors": len(anchors),
                    "anchors": anchors,
                    "channels": list(model_channels),
                    "channel_stats": stats,
                    "file": str(out_path),
                    "status": "skipped_exists",
                }
        except Exception:
            pass  # Re-extract on any error

    # Read WFDB record
    try:
        rec_path = str(Path(stay_dir) / seg_name)
        rec = wfdb.rdrecord(rec_path)
    except Exception as e:
        return {
            "patient_id": pid,
            "status": "error",
            "error": f"wfdb read failed: {e}",
        }

    # Extract model channels
    try:
        sig_indices = [rec.sig_name.index(sname) for sname in model_channels]

        # Full signal: (sig_len, n_signals) -> select channels and transpose to
        # (n_channels, sig_len)
        data = rec.p_signal[:, sig_indices]
        data = data.T.astype(np.float32)
    except Exception as e:
        return {
            "patient_id": pid,
            "status": "error",
            "error": f"channel extraction failed: {e}",
        }

    # Compute per-channel stats (for normalization at training time)
    stats = []
    for ch in range(n_channels):
        ch_data = data[ch]
        valid = ch_data[np.isfinite(ch_data)]
        if valid.size > 0:
            stats.append({"mean": float(valid.mean()), "std": float(valid.std())})
        else:
            stats.append({"mean": 0.0, "std": 1.0})

    # Save
    np.save(out_path, data)
    
    anchors = compute_anchors(sig_len)
    
    return {
        "patient_id": pid,
        "sig_len": sig_len,
        "seg_start_secs": seg_start_secs,
        "n_anchors": len(anchors),
        "anchors": anchors,
        "channels": list(model_channels),
        "channel_stats": stats,
        "file": str(out_path),
        "status": "extracted",
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-extract WFDB waveforms to numpy")
    parser.add_argument("--segments-path", type=str, default=str(SEGMENTS_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--channels",
        type=str,
        default=",".join(DEFAULT_MODEL_CHANNELS),
        help=f"Comma-separated channel order to extract. Valid: {RAW_SIGNAL_ORDER}",
    )
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel extraction workers")
    args = parser.parse_args()

    model_channels = parse_channels(args.channels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segment index
    seg_index = json.loads(Path(args.segments_path).read_text())
    segs = seg_index["segs"]
    all_patients = sorted(seg_index["patients"])

    print(f"Patients in index: {len(all_patients)}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Channels: {model_channels}", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print(flush=True)

    # Find best segment for each patient
    tasks = []
    for pid in all_patients:
        if pid not in segs:
            continue
        best = find_best_segment(segs[pid])
        if best is not None:
            tasks.append((pid, best))

    print(f"Patients to extract: {len(tasks)}", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    results = []
    errors = []
    extracted = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(extract_patient, pid, seg_info, output_dir, model_channels): pid
            for pid, seg_info in tasks
        }

        for i, future in enumerate(as_completed(futures)):
            pid = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"patient_id": pid, "status": "error", "error": str(e)}

            if result.get("status") == "extracted":
                extracted += 1
            elif result.get("status") == "skipped_exists":
                skipped += 1
            elif result.get("status") == "error":
                errors.append(result)
            
            results.append(result)

            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(tasks)}] "
                      f"extracted={extracted} skipped={skipped} errors={len(errors)} "
                      f"({rate:.1f} patients/s, ETA {eta/60:.0f}m)",
                      flush=True)

    elapsed = time.time() - t0
    print(flush=True)
    print(f"Done in {elapsed/60:.1f} minutes", flush=True)
    print(f"  Extracted: {extracted}", flush=True)
    print(f"  Skipped (exists): {skipped}", flush=True)
    print(f"  Errors: {len(errors)}", flush=True)

    if errors:
        print(f"\nErrors:", flush=True)
        for e in errors[:20]:
            print(f"  {e['patient_id']}: {e.get('error', 'unknown')}", flush=True)

    # Save metadata
    metadata = {
        "channels": list(model_channels),
        "raw_signal_order": list(RAW_SIGNAL_ORDER),
        "fs": FS,
        "ctx_samples": CTX_SAMPLES,
        "anchor_stride": ANCHOR_STRIDE,
        "min_windows": MIN_WINDOWS_PER_SEGMENT,
        "patients": {},
    }
    
    for r in results:
        if r.get("status") in ("extracted", "skipped_exists"):
            metadata["patients"][r["patient_id"]] = {
                "sig_len": r["sig_len"],
                "seg_start_secs": r["seg_start_secs"],
                "n_anchors": r["n_anchors"],
                "anchors": r["anchors"],
                "channels": r.get("channels", list(model_channels)),
                "channel_stats": r["channel_stats"],
            }

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata saved to {meta_path}", flush=True)
    print(f"Total patients in metadata: {len(metadata['patients'])}", flush=True)


if __name__ == "__main__":
    main()
