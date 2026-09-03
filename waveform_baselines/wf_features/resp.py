from __future__ import annotations

import numpy as np
from scipy import signal

from .quality import assess_signal_quality
from .utils import butter_filter, finite_runs, median_template_correlation, nan_iqr, robust_scale, safe_nanstat


def _detect_resp_extrema_by_finite_run(
    filtered: np.ndarray,
    fs: int,
    prominence: float,
    min_cycle_s: float,
) -> list[tuple[int, int, int]]:
    extrema: list[tuple[int, int, int]] = []
    min_distance = max(int(round(min_cycle_s * fs)), 1)
    min_run_samples = max(int(round(2.0 * min_cycle_s * fs)), min_distance + 1)
    for run_id, (start, end) in enumerate(finite_runs(np.isfinite(filtered))):
        segment = filtered[start:end]
        if segment.size < min_run_samples:
            continue
        peaks, _ = signal.find_peaks(segment, distance=min_distance, prominence=prominence)
        troughs, _ = signal.find_peaks(-segment, distance=min_distance, prominence=prominence)
        extrema.extend((start + int(idx), 1, run_id) for idx in peaks.tolist())
        extrema.extend((start + int(idx), -1, run_id) for idx in troughs.tolist())
    extrema.sort(key=lambda item: item[0])
    return extrema


def extract_resp_features(
    resp: np.ndarray,
    fs: int,
    micro_window_samples: int,
    template_points: int,
    detector_low_hz: float = 0.05,
    detector_high_hz: float = 1.5,
    min_cycle_s: float = 0.75,
    max_cycle_s: float = 15.0,
    quality_min_finite_fraction: float = 0.95,
    extreme_value_atol_fraction: float = 0.01,
    max_interp_gap_s: float = 0.2,
) -> dict[str, float]:
    quality = assess_signal_quality(
        resp,
        micro_window_samples=micro_window_samples,
        min_std=1e-4,
        min_finite_fraction=quality_min_finite_fraction,
        extreme_value_atol_fraction=extreme_value_atol_fraction,
    )
    max_gap_samples = max(int(round(max_interp_gap_s * fs)), 0)
    filtered = butter_filter(
        resp,
        fs=fs,
        low_hz=detector_low_hz,
        high_hz=detector_high_hz,
        order=2,
        max_interp_gap_samples=max_gap_samples,
    )
    prominence = max(0.2 * robust_scale(filtered), 1e-4)
    ordered = _detect_resp_extrema_by_finite_run(filtered, fs=fs, prominence=prominence, min_cycle_s=min_cycle_s)
    features: dict[str, float] = {
        "resp_valid_micro_fraction": quality.valid_micro_fraction,
        "resp_missing_micro_fraction": quality.missing_micro_fraction,
        "resp_flatline_fraction": quality.flatline_fraction,
    }
    if len(ordered) < 3:
        return features
    candidate_cycles = 0
    cycles = []
    segments: list[np.ndarray] = []
    for i in range(len(ordered) - 2):
        a_idx, a_sign, a_run = ordered[i]
        b_idx, b_sign, b_run = ordered[i + 1]
        c_idx, c_sign, c_run = ordered[i + 2]
        if a_run != b_run or a_run != c_run:
            continue
        if not (a_sign < 0 and b_sign > 0 and c_sign < 0):
            continue
        segment = filtered[a_idx:c_idx]
        if segment.size < 2 or not np.isfinite(segment).all():
            continue
        candidate_cycles += 1
        cycle_len = (c_idx - a_idx) / fs
        if cycle_len < min_cycle_s or cycle_len > max_cycle_s:
            continue
        amp = abs(filtered[b_idx] - filtered[a_idx])
        rise = (b_idx - a_idx) / fs
        fall = (c_idx - b_idx) / fs
        segments.append(segment)
        cycles.append({
            "length_s": cycle_len,
            "amplitude": amp,
            "rise_s": rise,
            "fall_s": fall,
            "rise_fall_ratio": rise / fall if fall > 1e-8 else np.nan,
            "rise_slope": amp / rise if rise > 1e-8 else np.nan,
            "fall_slope": amp / fall if fall > 1e-8 else np.nan,
            "area": float(np.trapezoid(np.abs(segment), dx=1.0 / fs)),
        })
    if not cycles:
        features["resp_valid_cycle_fraction"] = float(0.0) if candidate_cycles > 0 else float("nan")
        return features
    arrays = {key: np.asarray([row[key] for row in cycles], dtype=np.float64) for key in cycles[0]}
    valid_cycle = np.isfinite(arrays["length_s"])
    features.update({
        "resp_rate_bpm": safe_nanstat(np.median, 60.0 / arrays["length_s"][valid_cycle]),
        "resp_cycle_length_median_s": safe_nanstat(np.median, arrays["length_s"][valid_cycle]),
        "resp_cycle_length_iqr_s": nan_iqr(arrays["length_s"][valid_cycle]),
        "resp_amplitude_median": safe_nanstat(np.median, arrays["amplitude"][valid_cycle]),
        "resp_amplitude_iqr": nan_iqr(arrays["amplitude"][valid_cycle]),
        "resp_rise_time_median_s": safe_nanstat(np.median, arrays["rise_s"]),
        "resp_fall_time_median_s": safe_nanstat(np.median, arrays["fall_s"]),
        "resp_rise_fall_ratio_median": safe_nanstat(np.median, arrays["rise_fall_ratio"]),
        "resp_rise_slope_median": safe_nanstat(np.median, arrays["rise_slope"]),
        "resp_fall_slope_median": safe_nanstat(np.median, arrays["fall_slope"]),
        "resp_cycle_area_median": safe_nanstat(np.median, arrays["area"]),
        "resp_morphology_consistency": median_template_correlation(segments, template_points),
        "resp_valid_cycle_fraction": float(np.sum(valid_cycle) / candidate_cycles) if candidate_cycles > 0 else float("nan"),
    })
    return features
