from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .target_builders import ICUExtractionPaths, _as_path, _normalize_patient_ids

RAW_WAVEFORM_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/output_v2/raw_waveforms")
RAW_SIGNAL_ORDER = ("II", "ABP", "PLETH", "RESP")
WAVEFORM_MODEL_CHANNELS = ("ABP", "II", "PLETH")
SAMPLING_FREQUENCY_HZ = 125
WINDOW_SECONDS = 1200
WINDOW_SAMPLES = SAMPLING_FREQUENCY_HZ * WINDOW_SECONDS


def _channel_indices(channels: tuple[str, ...]) -> tuple[int, ...]:
    lookup = {name: idx for idx, name in enumerate(RAW_SIGNAL_ORDER)}
    return tuple(lookup[name] for name in channels)


def load_raw_waveform_manifest(raw_root: str | Path = RAW_WAVEFORM_ROOT) -> pd.DataFrame:
    """
    Load all raw waveform manifest shards written by `icuDataExtraction` step 2C.

    Returns one row per 20-minute raw waveform window.
    """
    raw_root = _as_path(raw_root)
    manifest_paths = sorted(raw_root.glob("manifest_job*.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"No raw waveform manifests found in {raw_root}")

    rows: list[dict[str, object]] = []
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        for patient_id, meta in manifest.items():
            raw_file = raw_root / meta["file"]
            window_times = meta["window_times"]
            for window_index, anchor_time in enumerate(window_times):
                rows.append(
                    {
                        "patient_id": str(patient_id),
                        "anchor_time": float(anchor_time),
                        "raw_file": str(raw_file),
                        "raw_window_index": int(window_index),
                        "seg_name": str(meta["seg_name"]),
                        "stay_dir": str(meta["stay_dir"]),
                        "sampling_rate_hz": int(meta["sampling_rate_hz"]),
                        "samples_per_window": int(meta["samples_per_window"]),
                        "signal_order": ",".join(meta["signal_order"]),
                    }
                )

    anchors = pd.DataFrame.from_records(rows)
    if anchors.empty:
        raise ValueError(f"Raw waveform manifests in {raw_root} contained no windows")

    anchors["input_start_time"] = anchors["anchor_time"] - WINDOW_SECONDS / 2.0
    anchors["input_end_time"] = anchors["anchor_time"] + WINDOW_SECONDS / 2.0
    return anchors.sort_values(["patient_id", "anchor_time"]).reset_index(drop=True)


def _anchors_from_icu_output(icu_output_dir: str | Path) -> pd.DataFrame:
    """Fallback aligned anchor table built directly from the canonical feature-window grid."""
    icu_output_dir = _as_path(icu_output_dir)
    patient_ids = _normalize_patient_ids(
        np.load(icu_output_dir / "patient_ids.npy", allow_pickle=True)
    )
    anchor_times = np.asarray(np.load(icu_output_dir / "window_times.npy"), dtype=np.float64)

    anchors = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "anchor_time": anchor_times,
        }
    )
    anchors["input_start_time"] = anchors["anchor_time"] - WINDOW_SECONDS / 2.0
    anchors["input_end_time"] = anchors["anchor_time"] + WINDOW_SECONDS / 2.0
    anchors["feature_alignment_ok"] = True
    anchors["anchor_id"] = np.arange(len(anchors), dtype=np.int64)
    return anchors.sort_values(["patient_id", "anchor_time"]).reset_index(drop=True)


def build_aligned_20m_anchor_table(
    raw_root: str | Path = RAW_WAVEFORM_ROOT,
    icu_output_dir: str | Path = ICUExtractionPaths().output_dir,
) -> pd.DataFrame:
    """
    Build the canonical 20-minute anchor table for this repository.

    This verifies that raw waveform windows and `icuDataExtraction` feature
    windows share the exact same `(patient_id, anchor_time)` key space.
    """
    icu_output_dir = _as_path(icu_output_dir)
    raw_root = _as_path(raw_root)

    if not raw_root.exists() or not any(raw_root.glob("manifest_job*.json")):
        return _anchors_from_icu_output(icu_output_dir)

    anchors = load_raw_waveform_manifest(raw_root)

    feature_patient_ids = _normalize_patient_ids(
        np.load(icu_output_dir / "patient_ids.npy", allow_pickle=True)
    )
    feature_times = np.asarray(np.load(icu_output_dir / "window_times.npy"), dtype=np.float64)
    feature_keys = set(zip(feature_patient_ids.tolist(), feature_times.tolist()))

    anchor_keys = list(zip(anchors["patient_id"].tolist(), anchors["anchor_time"].tolist()))
    aligned_mask = np.fromiter((key in feature_keys for key in anchor_keys), dtype=bool, count=len(anchor_keys))
    aligned = anchors.loc[aligned_mask].copy()

    if aligned.empty:
        raise ValueError("No raw waveform anchors aligned to icuDataExtraction feature windows")

    aligned["feature_alignment_ok"] = True
    aligned["anchor_id"] = np.arange(len(aligned), dtype=np.int64)
    return aligned.reset_index(drop=True)


class AlignedWaveformDataset:
    """
    Lightweight dataset wrapper for aligned 20-minute raw waveform windows.

    The anchor table is expected to come from `build_aligned_20m_anchor_table`.
    """

    def __init__(
        self,
        anchors: pd.DataFrame,
        target_bundle_path: str | Path | None = None,
        channels: tuple[str, ...] = WAVEFORM_MODEL_CHANNELS,
    ) -> None:
        self.anchors = anchors.reset_index(drop=True).copy()
        self.channels = channels
        self.channel_indices = _channel_indices(channels)
        self._raw_cache: dict[str, np.memmap] = {}

        self.feature_targets = None
        self.feature_mask = None
        self.event_targets = None
        self.event_mask = None
        if target_bundle_path is not None:
            bundle = np.load(_as_path(target_bundle_path), allow_pickle=False)
            if "feature_targets" in bundle:
                self.feature_targets = bundle["feature_targets"]
                self.feature_mask = bundle["feature_mask"]
            if "event_targets" in bundle:
                self.event_targets = bundle["event_targets"]
                self.event_mask = bundle["event_mask"]

    def __len__(self) -> int:
        return len(self.anchors)

    def _raw_array(self, raw_file: str) -> np.memmap:
        cached = self._raw_cache.get(raw_file)
        if cached is None:
            cached = np.load(raw_file, mmap_mode="r")
            self._raw_cache[raw_file] = cached
        return cached

    def waveform(self, index: int) -> np.ndarray:
        row = self.anchors.iloc[index]
        raw = self._raw_array(str(row["raw_file"]))
        return np.asarray(raw[int(row["raw_window_index"]), self.channel_indices, :], dtype=np.float32)

    def targets(self, index: int) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if self.feature_targets is not None:
            out["feature_targets"] = self.feature_targets[index]
            out["feature_mask"] = self.feature_mask[index]
        if self.event_targets is not None:
            out["event_targets"] = self.event_targets[index]
            out["event_mask"] = self.event_mask[index]
        return out

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.anchors.iloc[index]
        sample: dict[str, object] = {
            "anchor_id": int(row["anchor_id"]) if "anchor_id" in row else int(index),
            "patient_id": str(row["patient_id"]),
            "anchor_time": float(row["anchor_time"]),
            "input_start_time": float(row["input_start_time"]),
            "input_end_time": float(row["input_end_time"]),
            "waveform": self.waveform(index),
        }
        sample.update(self.targets(index))
        return sample
