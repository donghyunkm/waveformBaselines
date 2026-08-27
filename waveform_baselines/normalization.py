from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def normalization_stats_path(waveform_dir: Path, splits_path: Path) -> Path:
    stem = Path(splits_path).stem
    return Path(waveform_dir) / f"normalization_stats_{stem}.json"


@dataclass
class RunningMoments:
    count: int = 0
    sum: float = 0.0
    sumsq: float = 0.0
    n_excluded_nonfinite: int = 0

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(arr)
        self.n_excluded_nonfinite += int(arr.size - valid.sum())
        if not valid.any():
            return
        arr = arr[valid]
        self.count += int(arr.size)
        self.sum += float(arr.sum())
        self.sumsq += float(np.square(arr).sum())

    def finalize(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "mean": 0.0,
                "std": 1.0,
                "count": 0,
                "n_excluded_nonfinite": self.n_excluded_nonfinite,
                "fraction_excluded_nonfinite": 0.0,
            }
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean * mean, 0.0)
        std = float(np.sqrt(var))
        total_seen = self.count + self.n_excluded_nonfinite
        frac_excluded = (
            self.n_excluded_nonfinite / total_seen if total_seen > 0 else 0.0
        )
        return {
            "mean": float(mean),
            "std": std,
            "count": int(self.count),
            "n_excluded_nonfinite": int(self.n_excluded_nonfinite),
            "fraction_excluded_nonfinite": float(frac_excluded),
        }


class ReservoirSampler:
    def __init__(self, max_size: int, seed: int):
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)
        self.values = np.empty(max_size, dtype=np.float64)
        self.n_seen = 0
        self.size = 0

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        for value in arr:
            self.n_seen += 1
            if self.size < self.max_size:
                self.values[self.size] = value
                self.size += 1
            else:
                j = self.rng.integers(0, self.n_seen)
                if j < self.max_size:
                    self.values[j] = value

    def sample(self) -> np.ndarray:
        return self.values[: self.size].copy()


def _load_patient_ids_for_split(splits_path: Path, split: str = "train") -> list[str]:
    splits = json.loads(Path(splits_path).read_text())
    if split not in splits:
        raise ValueError(f"Split '{split}' not found in {splits_path}")
    return sorted(str(pid) for pid in splits[split])


def _load_waveform_metadata(waveform_dir: Path) -> dict:
    return json.loads((Path(waveform_dir) / "metadata.json").read_text())


def compute_training_channel_stats(
    waveform_dir: Path,
    splits_path: Path,
    channels: Iterable[str] | None = None,
    split: str = "train",
    clip_lower_percentile: float | None = None,
    clip_upper_percentile: float | None = None,
    clip_sample_size: int = 1_000_000,
    seed: int = 42,
) -> dict:
    waveform_dir = Path(waveform_dir)
    splits_path = Path(splits_path)
    metadata = _load_waveform_metadata(waveform_dir)
    available_channels = tuple(metadata["channels"])
    if channels is None:
        selected_channels = available_channels
    else:
        selected_channels = tuple(channels)
    missing = [ch for ch in selected_channels if ch not in available_channels]
    if missing:
        raise ValueError(
            f"Requested channels {missing} not available in {waveform_dir}. "
            f"Available: {available_channels}"
        )

    patient_ids = _load_patient_ids_for_split(splits_path, split=split)
    channel_indices = {ch: available_channels.index(ch) for ch in selected_channels}
    moments = {ch: RunningMoments() for ch in selected_channels}

    clip_bounds = None
    if clip_lower_percentile is not None or clip_upper_percentile is not None:
        if clip_lower_percentile is None or clip_upper_percentile is None:
            raise ValueError("Specify both clip percentiles or neither.")
        samplers = {
            ch: ReservoirSampler(max_size=clip_sample_size, seed=seed + idx)
            for idx, ch in enumerate(selected_channels)
        }
        for pid in patient_ids:
            if pid not in metadata["patients"]:
                continue
            npy_path = waveform_dir / f"{pid}.npy"
            if not npy_path.exists():
                continue
            arr = np.load(npy_path, mmap_mode="r")
            for ch in selected_channels:
                samplers[ch].update(arr[channel_indices[ch]])
        clip_bounds = {}
        for ch in selected_channels:
            sample = samplers[ch].sample()
            if sample.size == 0:
                clip_bounds[ch] = None
                continue
            clip_bounds[ch] = {
                "lower": float(np.percentile(sample, clip_lower_percentile)),
                "upper": float(np.percentile(sample, clip_upper_percentile)),
            }

    for pid in patient_ids:
        if pid not in metadata["patients"]:
            continue
        npy_path = waveform_dir / f"{pid}.npy"
        if not npy_path.exists():
            continue
        arr = np.load(npy_path, mmap_mode="r")
        for ch in selected_channels:
            values = np.asarray(arr[channel_indices[ch]], dtype=np.float64)
            if clip_bounds and clip_bounds[ch] is not None:
                valid = np.isfinite(values)
                clipped = values.copy()
                clipped[valid] = np.clip(
                    clipped[valid],
                    clip_bounds[ch]["lower"],
                    clip_bounds[ch]["upper"],
                )
                values = clipped
            moments[ch].update(values)

    stats = {
        "method": "training_set_per_channel_zscore",
        "waveform_dir": str(waveform_dir),
        "splits_path": str(splits_path),
        "split_used": split,
        "channels": {ch: moments[ch].finalize() for ch in selected_channels},
    }
    if clip_bounds is not None:
        stats["clipping"] = {
            "method": "training_set_percentile_clip",
            "lower_percentile": clip_lower_percentile,
            "upper_percentile": clip_upper_percentile,
            "sample_size_per_channel": clip_sample_size,
            "bounds": clip_bounds,
        }
    else:
        stats["clipping"] = None
    return stats


def save_training_channel_stats(
    waveform_dir: Path,
    splits_path: Path,
    stats: dict,
) -> Path:
    out_path = normalization_stats_path(Path(waveform_dir), Path(splits_path))
    out_path.write_text(json.dumps(stats, indent=2))
    return out_path


def load_training_channel_stats(
    waveform_dir: Path,
    splits_path: Path,
) -> dict:
    stats_path = normalization_stats_path(Path(waveform_dir), Path(splits_path))
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing normalization stats file: {stats_path}. "
            f"Build it with: PYTHONPATH=. "
            f"/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python "
            f"scripts/compute_waveform_normalization_stats.py "
            f"--waveform-dir {Path(waveform_dir)} --splits-path {Path(splits_path)}"
        )
    with open(stats_path) as f:
        return json.load(f)
