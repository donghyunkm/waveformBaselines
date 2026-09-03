#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waveform_baselines.wf_features.config import DEFAULT_EXTRACTION_CONFIG
from waveform_baselines.wf_features.pipeline import extract_feature_sequence


def synthetic_segment() -> np.ndarray:
    config = DEFAULT_EXTRACTION_CONFIG
    t = np.arange(config.input_samples, dtype=np.float64) / config.sampling_rate_hz
    ecg = 0.1 * np.sin(2 * np.pi * 1.2 * t)
    for beat_time in np.arange(0.4, config.input_window_seconds, 1.0):
        beat_idx = int(round(beat_time * config.sampling_rate_hz))
        if beat_idx + 3 < ecg.size:
            ecg[beat_idx : beat_idx + 3] += np.array([0.8, 1.5, 0.6])
    abp = 80.0 + 20.0 * np.sin(2 * np.pi * 1.2 * t - 0.2) + 10.0 * np.maximum(np.sin(2 * np.pi * 1.2 * t), 0)
    pleth = 1.0 + 0.4 * np.sin(2 * np.pi * 1.2 * t - 0.4) + 0.15 * np.maximum(np.sin(2 * np.pi * 1.2 * t), 0)
    resp = 0.5 * np.sin(2 * np.pi * 0.25 * t)
    return np.stack([ecg, abp, pleth, resp], axis=0).astype(np.float32)


def main() -> None:
    config = DEFAULT_EXTRACTION_CONFIG
    waveform = synthetic_segment()
    values, mask, names = extract_feature_sequence(waveform, config)
    minute_axis = np.arange(config.n_feature_windows)
    out_dir = Path("docs/figures/extracted_features_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(14, 10), constrained_layout=True)
    time_axis = np.arange(5 * config.sampling_rate_hz) / config.sampling_rate_hz
    channel_names = ("II", "ABP", "PLETH", "RESP")
    for row_idx, channel_name in enumerate(channel_names):
        axes[row_idx, 0].plot(time_axis, waveform[row_idx, : time_axis.size], linewidth=1.0)
        axes[row_idx, 0].set_title(f"{channel_name} first 5 s")
        axes[row_idx, 0].set_xlabel("Seconds")
    selected_features = [
        "ecg_hr_bpm",
        "abp_map_median_mmhg",
        "pleth_amplitude_median",
        "resp_rate_bpm",
    ]
    for row_idx, feature_name in enumerate(selected_features):
        feat_idx = names.index(feature_name)
        feature_values = values[:, feat_idx].copy()
        feature_values[~mask[:, feat_idx]] = np.nan
        axes[row_idx, 1].plot(minute_axis, feature_values, marker="o")
        axes[row_idx, 1].set_title(f"{feature_name} by minute")
        axes[row_idx, 1].set_xlabel("Minute token")
    output_path = out_dir / "synthetic_feature_smoke.png"
    fig.savefig(output_path, dpi=160)
    print(output_path)


if __name__ == "__main__":
    main()
