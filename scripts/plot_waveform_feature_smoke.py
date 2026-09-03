#!/usr/bin/env python3
"""Plot detector overlays and minute-level features for one cached smoke sample."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from waveform_baselines.wf_features.cache import FeatureCacheBuilder, load_feature_cache
from waveform_baselines.wf_features.config import DEFAULT_EXTRACTION_CONFIG
from waveform_baselines.wf_features.ecg import _detect_r_peaks_by_finite_run
from waveform_baselines.wf_features.pulsatile import _detect_peaks_by_finite_run, _segment_between_troughs
from waveform_baselines.wf_features.resp import _detect_resp_extrema_by_finite_run
from waveform_baselines.wf_features.utils import butter_filter, interpolate_short_gaps, robust_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--minute-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = load_feature_cache(args.cache_dir)
    builder = FeatureCacheBuilder(config=DEFAULT_EXTRACTION_CONFIG)
    # Rebuild the aligned table and use the same private loader as extraction.
    from waveform_baselines.data_index import build_aligned_20m_anchor_table

    table = build_aligned_20m_anchor_table()
    patient_id = str(cache.patient_ids[args.sample_index])
    anchor_time = float(cache.anchor_times[args.sample_index])
    table_anchor_times = table["anchor_time"].astype(float).round(6)
    matches = table[
        (table["patient_id"].astype(str) == patient_id)
        & (table_anchor_times == round(anchor_time, 6))
    ]
    if matches.empty:
        raise ValueError(f"Cached sample key {(patient_id, anchor_time)} not found in aligned anchor table")
    waveform = builder._load_waveform_sample(matches.reset_index(drop=True), 0, None)
    fs = DEFAULT_EXTRACTION_CONFIG.sampling_rate_hz
    start = args.minute_index * fs * 60
    stop = start + fs * 60
    sec = np.arange(fs * 60) / fs
    fig, axes = plt.subplots(5, 1, figsize=(15, 13), constrained_layout=True)

    ecg = waveform[0, start:stop]
    max_gap = int(round(DEFAULT_EXTRACTION_CONFIG.max_interpolated_gap_seconds * fs))
    ecg_xqrs, ecg_usable = interpolate_short_gaps(ecg, max_gap)
    ecg_xqrs[~ecg_usable] = np.nan
    ecg_detector = butter_filter(ecg, fs, low_hz=5.0, high_hz=20.0, order=3, max_interp_gap_samples=max_gap)
    ecg_peaks, _ = _detect_r_peaks_by_finite_run(
        ecg_xqrs,
        fs,
        prominence=max(0.2 * robust_scale(ecg_detector), 1e-3),
        fallback_signal=ecg_detector,
        detector=DEFAULT_EXTRACTION_CONFIG.ecg_detector,
        allow_energy_fallback=DEFAULT_EXTRACTION_CONFIG.ecg_allow_energy_fallback,
    )
    axes[0].plot(sec, ecg, label="raw II", linewidth=0.7)
    axes[0].plot(sec, ecg_detector, label="5-20 Hz detector", linewidth=0.7)
    axes[0].scatter(ecg_peaks / fs, ecg[ecg_peaks], s=10, label="R peaks")
    axes[0].set_title("ECG: raw, detector, and R peaks")

    for ax, channel, title, low, high, morphology in [
        (axes[1], 1, "ABP: raw, detector, peaks, trough-to-trough beats", 0.5, 12.0, 20.0),
        (axes[2], 2, "PLETH: raw, detector, peaks, trough-to-trough beats", 0.5, 8.0, 8.0),
    ]:
        raw = waveform[channel, start:stop]
        detector = butter_filter(raw, fs, low_hz=low, high_hz=high, order=3)
        min_distance_s = 60.0 / DEFAULT_EXTRACTION_CONFIG.abp_max_pulse_bpm if channel == 1 else 0.3
        peaks = _detect_peaks_by_finite_run(detector, fs, min_distance_s=min_distance_s, prominence=max(0.2 * robust_scale(detector), 1e-3))
        onsets, offsets, _ = _segment_between_troughs(detector, peaks)
        ax.plot(sec, raw, label="calibrated/raw", linewidth=0.7)
        ax.plot(sec, detector, label="detector", linewidth=0.7)
        ax.scatter(peaks / fs, detector[peaks], s=10, label="peaks")
        for onset, offset in zip(onsets, offsets):
            ax.axvspan(onset / fs, offset / fs, color="tab:green", alpha=0.08)
        ax.set_title(title)
        ax.legend(loc="upper right", ncol=3)

    resp = waveform[3, start:stop]
    resp_filtered = butter_filter(resp, fs, low_hz=0.05, high_hz=1.5, order=2)
    resp_extrema = _detect_resp_extrema_by_finite_run(
        resp_filtered,
        fs,
        prominence=max(0.2 * robust_scale(resp_filtered), 1e-4),
        min_cycle_s=DEFAULT_EXTRACTION_CONFIG.resp_min_cycle_s,
    )
    resp_peaks = np.asarray([idx for idx, sign, _ in resp_extrema if sign > 0], dtype=int)
    resp_troughs = np.asarray([idx for idx, sign, _ in resp_extrema if sign < 0], dtype=int)
    axes[3].plot(sec, resp, label="raw RESP", linewidth=0.7)
    axes[3].plot(sec, resp_filtered, label="0.05-1.5 Hz", linewidth=0.7)
    axes[3].scatter(resp_peaks / fs, resp_filtered[resp_peaks], s=10, label="peaks")
    axes[3].scatter(resp_troughs / fs, resp_filtered[resp_troughs], s=10, label="troughs")
    axes[3].set_title("RESP: raw, filtered, and alternating extrema")
    axes[3].legend(loc="upper right", ncol=3)

    names = cache.feature_names
    trajectories = [("ecg_hr_bpm", "ECG HR"), ("abp_map_median_mmhg", "ABP MAP"), ("pleth_amplitude_median", "PLETH amplitude"), ("resp_rate_bpm", "RESP rate")]
    for name, label in trajectories:
        idx = names.index(name)
        axes[4].plot(np.arange(20), cache.values[args.sample_index, :, idx], marker="o", label=label)
    axes[4].set_title("Minute-level feature trajectories")
    axes[4].set_xlabel("Input minute token")
    axes[4].legend(loc="best", ncol=4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
