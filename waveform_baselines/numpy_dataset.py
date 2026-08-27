"""
Fast NumPy-based Dataset for pre-extracted waveform segments.

Reads per-patient .npy files (produced by scripts/extract_waveforms.py)
and slices configurable waveform windows from them. No WFDB parsing at
training time.

Key performance features:
- np.load with mmap_mode='r' — OS handles caching/paging, no manual LRU
- Patient-grouped sampler ensures consecutive batches hit the same file
- Per-channel normalization using train-split statistics shared across patients
- Zero-copy window slicing from memory-mapped files
- Minimal per-sample overhead
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization import load_training_channel_stats

# ── Constants ─────────────────────────────────────────────────────────────────

FS = 125
DEFAULT_CTX_SAMPLES = 150_000  # 20 min at 125 Hz
DEFAULT_HALF_CTX = DEFAULT_CTX_SAMPLES // 2
ANCHOR_STRIDE = int(2.5 * 60 * FS)  # 18750 samples

MODEL_CHANNELS = ("ABP", "II", "PLETH")
RAW_SIGNAL_ORDER = ("II", "ABP", "PLETH", "RESP")

DEFAULT_WAVEFORM_DIR = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms")
DEFAULT_SPLITS_PATH = Path("/gpfs/home/dk5565/waveformBaselines/outputs/splits/splits.json")


class NumpyWaveformDataset(Dataset):
    """
    Fast dataset reading pre-extracted per-patient numpy waveform files.

    Parameters
    ----------
    split : str
        One of "train", "val", "test", or "all".
    waveform_dir : Path
        Directory with <patient_id>.npy files and metadata.json.
    splits_path : Path
        Path to splits.json with patient lists per split.
    normalize : bool
        If True, apply per-channel z-normalization using training-split stats.
    """

    def __init__(
        self,
        split: str = "train",
        waveform_dir: Path = DEFAULT_WAVEFORM_DIR,
        splits_path: Path = DEFAULT_SPLITS_PATH,
        normalize: bool = True,
        channels: tuple[str, ...] | None = None,
        seq_len: int | None = None,
    ):
        super().__init__()
        self.split = split
        self.waveform_dir = Path(waveform_dir)
        self.normalize = normalize
        self.splits_path = Path(splits_path)
        self.seq_len = seq_len if seq_len is not None else DEFAULT_CTX_SAMPLES
        self.half_ctx = self.seq_len // 2

        # Load metadata (has anchors, stats, sig_len per patient)
        meta_path = self.waveform_dir / "metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)

        self.fs = int(metadata.get("fs", FS))
        self.available_channels = tuple(metadata.get("channels", MODEL_CHANNELS))
        if channels is None:
            self.channels = self.available_channels
        else:
            self.channels = tuple(channels)
        if len(set(self.channels)) != len(self.channels):
            raise ValueError(f"Duplicate channels requested: {self.channels}")
        missing = [ch for ch in self.channels if ch not in self.available_channels]
        if missing:
            raise ValueError(
                f"Requested channels {missing} not present in {self.waveform_dir}/metadata.json. "
                f"Available: {self.available_channels}"
            )
        self.channel_indices = tuple(self.available_channels.index(ch) for ch in self.channels)
        self.n_channels = len(self.channels)
        self._patient_meta = metadata["patients"]
        self._normalization_stats = None
        if self.normalize:
            raw_stats = load_training_channel_stats(self.waveform_dir, self.splits_path)
            all_stats = raw_stats.get("channels", {})
            missing_stats = [ch for ch in self.channels if ch not in all_stats]
            if missing_stats:
                raise ValueError(
                    f"Normalization stats missing channels {missing_stats} in "
                    f"{self.waveform_dir} for splits {self.splits_path}"
                )
            self._normalization_stats = {}
            for ch in self.channels:
                stats = dict(all_stats[ch])
                mu = float(stats["mean"])
                std = float(stats["std"])
                if not np.isfinite(mu):
                    warnings.warn(
                        f"Non-finite normalization mean for channel {ch} in "
                        f"{self.waveform_dir}; using 0.0"
                    )
                    mu = 0.0
                if not np.isfinite(std) or std < 1e-8:
                    warnings.warn(
                        f"Degenerate normalization std for channel {ch} in "
                        f"{self.waveform_dir}; using 1.0"
                    )
                    std = 1.0
                stats["mean"] = mu
                stats["std"] = std
                self._normalization_stats[ch] = stats

        # Determine patient list for this split
        if split == "all":
            patient_list = sorted(self._patient_meta.keys())
        else:
            with open(self.splits_path) as f:
                splits = json.load(f)
            if split not in splits:
                raise ValueError(f"Split '{split}' not in {self.splits_path}")
            patient_list = sorted(splits[split])

        # Build flat window index: list of (patient_id, anchor_center_sample, patient_idx_in_list)
        # Also store patient boundaries for the grouped sampler
        self._windows: list[tuple[str, int]] = []
        self._patient_ids: list[str] = []  # ordered list of patients that have data
        self._patient_boundaries: list[tuple[int, int]] = []  # (start_idx, end_idx) per patient
        self._patient_seg_start: dict[str, float] = {}  # pid -> seg_start_secs

        # Memory-mapped file handles (lazy-loaded)
        self._mmap_cache: dict[str, np.ndarray] = {}

        for pid in patient_list:
            if pid not in self._patient_meta:
                continue
            
            pmeta = self._patient_meta[pid]
            npy_path = self.waveform_dir / f"{pid}.npy"
            if not npy_path.exists():
                continue

            anchors = pmeta["anchors"]
            if not anchors:
                continue

            start_idx = len(self._windows)
            for anchor in anchors:
                self._windows.append((pid, anchor))
            end_idx = len(self._windows)

            self._patient_ids.append(pid)
            self._patient_boundaries.append((start_idx, end_idx))
            self._patient_seg_start[pid] = pmeta["seg_start_secs"]

    def __len__(self) -> int:
        return len(self._windows)

    @property
    def patient_ids(self) -> list[str]:
        return self._patient_ids

    @property
    def patient_boundaries(self) -> list[tuple[int, int]]:
        """(start_idx, end_idx) in the window list for each patient."""
        return self._patient_boundaries

    def _get_mmap(self, patient_id: str) -> np.ndarray:
        """Get memory-mapped array for a patient (lazy load)."""
        if patient_id not in self._mmap_cache:
            npy_path = self.waveform_dir / f"{patient_id}.npy"
            self._mmap_cache[patient_id] = np.load(npy_path, mmap_mode='r')
        return self._mmap_cache[patient_id]

    def __getitem__(self, index: int) -> dict:
        patient_id, anchor_center = self._windows[index]

        # Get memory-mapped segment
        segment = self._get_mmap(patient_id)

        # Slice window: (n_channels, seq_len)
        start = anchor_center - self.half_ctx
        end = start + self.seq_len
        if start < 0 or end > segment.shape[1]:
            raise IndexError(
                f"Requested window [{start}, {end}) outside patient segment {patient_id} "
                f"with length {segment.shape[1]}"
            )
        window = segment[self.channel_indices, start:end].copy()  # copy to detach from mmap

        # Replace any non-finite raw waveform values before normalization.
        np.nan_to_num(window, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize using training-split per-channel stats shared across patients.
        if self.normalize:
            for ch in range(self.n_channels):
                stats = self._normalization_stats[self.channels[ch]]
                window[ch] = (window[ch] - float(stats["mean"])) / float(stats["std"])

        # Final safeguard: data loader must only emit finite waveforms.
        np.nan_to_num(window, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute anchor time in seconds
        seg_start = self._patient_seg_start[patient_id]
        anchor_time = seg_start + anchor_center / self.fs

        return {
            "waveform": torch.from_numpy(window),
            "patient_id": patient_id,
            "anchor_time": anchor_time,
        }

    def window_counts_by_patient(self) -> dict[str, int]:
        """Number of windows per patient."""
        counts: dict[str, int] = {}
        for pid, _ in self._windows:
            counts[pid] = counts.get(pid, 0) + 1
        return counts
