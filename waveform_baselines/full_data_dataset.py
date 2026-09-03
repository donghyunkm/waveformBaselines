from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import wfdb
except ImportError:  # pragma: no cover
    wfdb = None

from .normalization import load_training_channel_stats


FS = 125
DEFAULT_CTX_SAMPLES = 150_000
INPUT_WINDOW_POSITION_CHOICES = ("center", "input_end")


def load_full_data_anchor_rows(anchor_cache_dir: Path, split: str) -> list[dict[str, str]]:
    anchors_path = Path(anchor_cache_dir) / "anchors.csv"
    if not anchors_path.exists():
        raise FileNotFoundError(f"Missing full-data anchors file: {anchors_path}")

    rows: list[dict[str, str]] = []
    with open(anchors_path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"anchor_id", "patient_id", "segment_id", "seg_name", "window_time", "split_label", "segment_path"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{anchors_path} missing required columns: {sorted(missing)}")
        for row in reader:
            if split == "all" or str(row["split_label"]) == split:
                rows.append(row)
    if not rows:
        raise ValueError(f"No full-data anchors found for split={split!r} in {anchors_path}")
    return rows


class FullDataSegmentWaveformDataset(Dataset):
    """Segment-aware full-data raw waveform dataset.

    ``waveform_dir`` points at the completed full-data feature cache directory
    that contains ``anchors.csv``. The CSV supplies row-level ``anchor_id`` and
    WFDB ``segment_path`` values, so target alignment can use ``anchor_id``
    instead of the non-unique full-data ``(patient_id, anchor_time)`` pair.
    """

    def __init__(
        self,
        split: str = "train",
        waveform_dir: Path | str = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v7/full_data_vasopressor_free_waveform_features_v7"),
        splits_path: Path | str | None = None,
        normalize: bool = True,
        channels: tuple[str, ...] | None = None,
        seq_len: int | None = None,
        input_window_position: str = "center",
    ) -> None:
        super().__init__()
        if wfdb is None:
            raise ImportError("wfdb is required for FullDataSegmentWaveformDataset")

        self.split = split
        self.waveform_dir = Path(waveform_dir)
        self.splits_path = Path(splits_path) if splits_path is not None else None
        self.normalize = normalize
        self.seq_len = int(seq_len if seq_len is not None else DEFAULT_CTX_SAMPLES)
        self.half_ctx = self.seq_len // 2
        self.fs = FS
        self.input_window_position = str(input_window_position)
        if self.input_window_position not in INPUT_WINDOW_POSITION_CHOICES:
            raise ValueError(
                f"input_window_position must be one of {INPUT_WINDOW_POSITION_CHOICES}, "
                f"got {self.input_window_position!r}"
            )

        metadata_path = self.waveform_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        available = metadata.get("channel_order") or metadata.get("extraction_config", {}).get("channel_order")
        self.available_channels = tuple(available or ("II", "ABP", "PLETH", "RESP"))
        self.channels = tuple(channels) if channels is not None else self.available_channels
        if len(set(self.channels)) != len(self.channels):
            raise ValueError(f"Duplicate channels requested: {self.channels}")
        missing = [ch for ch in self.channels if ch not in self.available_channels]
        if missing:
            raise ValueError(
                f"Requested channels {missing} not available in full-data cache metadata. "
                f"Available: {self.available_channels}"
            )
        self.n_channels = len(self.channels)

        raw_rows = load_full_data_anchor_rows(self.waveform_dir, split)
        raw_rows.sort(key=lambda r: (str(r["segment_id"]), float(r["window_time"]), int(r["anchor_id"])))
        self._windows: list[dict[str, object]] = []
        self._patient_ids_set: set[str] = set()
        self._segment_boundaries: list[tuple[int, int]] = []
        self._segment_ids: list[str] = []

        current_segment: str | None = None
        segment_start_idx = 0
        for row in raw_rows:
            segment_id = str(row["segment_id"])
            if current_segment is None:
                current_segment = segment_id
                segment_start_idx = len(self._windows)
            elif segment_id != current_segment:
                self._segment_ids.append(current_segment)
                self._segment_boundaries.append((segment_start_idx, len(self._windows)))
                current_segment = segment_id
                segment_start_idx = len(self._windows)

            patient_id = str(row["patient_id"])
            self._patient_ids_set.add(patient_id)
            self._windows.append(
                {
                    "anchor_id": int(row["anchor_id"]),
                    "patient_id": patient_id,
                    "segment_id": segment_id,
                    "segment_name": str(row["seg_name"]),
                    "segment_path": str(row["segment_path"]),
                    "anchor_time": float(row["window_time"]),
                    "split_label": str(row["split_label"]),
                }
            )
        if current_segment is not None:
            self._segment_ids.append(current_segment)
            self._segment_boundaries.append((segment_start_idx, len(self._windows)))

        self._normalization_stats = None
        if self.normalize:
            raw_stats = load_training_channel_stats(self.waveform_dir, self.splits_path or Path("patient_splits.json"))
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
                    warnings.warn(f"Non-finite normalization mean for channel {ch}; using 0.0")
                    mu = 0.0
                if not np.isfinite(std) or std < 1e-8:
                    warnings.warn(f"Degenerate normalization std for channel {ch}; using 1.0")
                    std = 1.0
                stats["mean"] = mu
                stats["std"] = std
                self._normalization_stats[ch] = stats

        self._current_segment_path: str | None = None
        self._current_segment: np.ndarray | None = None
        self._current_signal_names: tuple[str, ...] | None = None
        self._current_fs: float | None = None

    def __len__(self) -> int:
        return len(self._windows)

    @property
    def patient_ids(self) -> list[str]:
        return sorted(self._patient_ids_set)

    @property
    def patient_boundaries(self) -> list[tuple[int, int]]:
        return self._segment_boundaries

    def target_identity(self, index: int) -> dict[str, object]:
        row = self._windows[index]
        return {
            "anchor_id": int(row["anchor_id"]),
            "patient_id": str(row["patient_id"]),
            "anchor_time": float(row["anchor_time"]),
            "segment_id": str(row["segment_id"]),
            "segment_name": str(row["segment_name"]),
        }

    def _load_segment(self, segment_path: str) -> tuple[np.ndarray, tuple[str, ...]]:
        if segment_path != self._current_segment_path:
            rec = wfdb.rdrecord(segment_path)
            self._current_segment = np.asarray(rec.p_signal.T, dtype=np.float32)
            self._current_signal_names = tuple(str(name) for name in rec.sig_name)
            self._current_fs = float(rec.fs)
            self._current_segment_path = segment_path
        if self._current_segment is None or self._current_signal_names is None or self._current_fs is None:
            raise RuntimeError(f"Failed to load segment {segment_path}")
        if int(round(self._current_fs)) != self.fs:
            raise ValueError(f"Segment {segment_path} fs={self._current_fs} does not match expected {self.fs}")
        return self._current_segment, self._current_signal_names

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self._windows[index]
        segment, signal_names = self._load_segment(str(row["segment_path"]))
        missing = [ch for ch in self.channels if ch not in signal_names]
        if missing:
            raise ValueError(f"Segment {row['segment_id']} missing channels {missing}; available={signal_names}")
        channel_indices = [signal_names.index(ch) for ch in self.channels]

        anchor_sample = int(round(float(row["anchor_time"]) * self.fs))
        if self.input_window_position == "center":
            start = anchor_sample - self.half_ctx
        else:
            input_end = anchor_sample + DEFAULT_CTX_SAMPLES // 2
            start = input_end - self.seq_len
        end = start + self.seq_len
        if start < 0 or end > segment.shape[1]:
            raise IndexError(
                f"Requested window [{start}, {end}) outside segment {row['segment_id']} "
                f"with length {segment.shape[1]}"
            )
        window = segment[channel_indices, start:end].copy()
        np.nan_to_num(window, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize:
            for ch_idx, ch in enumerate(self.channels):
                stats = self._normalization_stats[ch]
                window[ch_idx] = (window[ch_idx] - float(stats["mean"])) / float(stats["std"])
        np.nan_to_num(window, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "waveform": torch.from_numpy(window),
            "patient_id": str(row["patient_id"]),
            "anchor_time": float(row["anchor_time"]),
            "anchor_id": int(row["anchor_id"]),
            "segment_id": str(row["segment_id"]),
            "segment_name": str(row["segment_name"]),
        }
