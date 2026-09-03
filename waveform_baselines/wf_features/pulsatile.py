from __future__ import annotations

import numpy as np
from scipy import signal

from .quality import assess_signal_quality
from .utils import butter_filter, finite_runs, median_template_correlation, nan_iqr, robust_scale, safe_nanstat, safe_ratio, slope


def _detect_peaks_by_finite_run(detector_signal: np.ndarray, fs: int, min_distance_s: float, prominence: float) -> np.ndarray:
    peaks: list[int] = []
    min_distance = max(int(min_distance_s * fs), 1)
    min_run_samples = max(int(0.5 * fs), min_distance + 1)
    for start, end in finite_runs(np.isfinite(detector_signal)):
        segment = detector_signal[start:end]
        if segment.size < min_run_samples:
            continue
        local_peaks, _ = signal.find_peaks(segment, distance=min_distance, prominence=prominence)
        peaks.extend((start + local_peaks).tolist())
    if not peaks:
        return np.asarray([], dtype=int)
    return np.unique(np.asarray(peaks, dtype=int))


def _refine_local_extreme(
    values: np.ndarray,
    center: int,
    radius: int,
    mode: str,
    lower_bound: int,
    upper_bound: int,
) -> int | None:
    start = max(lower_bound, center - radius)
    end = min(upper_bound, center + radius + 1)
    if end <= start:
        return None
    local = np.asarray(values[start:end], dtype=np.float64)
    finite = np.isfinite(local)
    if not finite.any():
        return None
    finite_offsets = np.flatnonzero(finite)
    if mode == "max":
        chosen = finite_offsets[int(np.argmax(local[finite]))]
    elif mode == "min":
        chosen = finite_offsets[int(np.argmin(local[finite]))]
    else:
        raise ValueError(f"Unsupported mode {mode!r}")
    return start + int(chosen)


def _segment_between_troughs(filtered: np.ndarray, peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if peaks.size < 3:
        empty = np.asarray([], dtype=int)
        return empty, empty, empty
    interpeak_troughs: list[int] = []
    valid_pairs: list[int] = []
    for idx in range(peaks.size - 1):
        left = int(peaks[idx])
        right = int(peaks[idx + 1])
        if right <= left + 1:
            continue
        local = np.asarray(filtered[left : right + 1], dtype=np.float64)
        if local.size < 2 or not np.isfinite(local).all():
            continue
        trough = left + int(np.argmin(local))
        if trough <= left or trough >= right:
            continue
        interpeak_troughs.append(trough)
        valid_pairs.append(idx)
    if len(interpeak_troughs) < 2:
        empty = np.asarray([], dtype=int)
        return empty, empty, empty
    troughs = np.asarray(interpeak_troughs, dtype=int)
    pair_idx = np.asarray(valid_pairs, dtype=int)
    onsets: list[int] = []
    offsets: list[int] = []
    beat_peaks: list[int] = []
    for idx in range(troughs.size - 1):
        if pair_idx[idx + 1] != pair_idx[idx] + 1:
            continue
        onset = int(troughs[idx])
        offset = int(troughs[idx + 1])
        if offset <= onset:
            continue
        peak = int(peaks[pair_idx[idx] + 1])
        if not (onset < peak < offset):
            continue
        onsets.append(onset)
        offsets.append(offset)
        beat_peaks.append(peak)
    return (
        np.asarray(onsets, dtype=int),
        np.asarray(offsets, dtype=int),
        np.asarray(beat_peaks, dtype=int),
    )


def _refine_trough_sequence(
    measurement_signal: np.ndarray,
    detector_troughs: np.ndarray,
    radius: int,
) -> np.ndarray:
    refined = np.full(detector_troughs.size, -1, dtype=int)
    for pos, trough in enumerate(detector_troughs.tolist()):
        lower_bound = 0 if pos == 0 else int(detector_troughs[pos - 1]) + 1
        upper_bound = measurement_signal.size if pos + 1 == detector_troughs.size else int(detector_troughs[pos + 1])
        idx = _refine_local_extreme(
            measurement_signal,
            int(trough),
            radius,
            "min",
            lower_bound,
            upper_bound,
        )
        if idx is None:
            continue
        refined[pos] = int(idx)
    return refined


def _extract_pulsatile_beats(
    measurement_signal: np.ndarray,
    detector_signal: np.ndarray,
    morphology_signal: np.ndarray,
    fs: int,
    signal_kind: str,
    template_points: int,
    peak_search_radius_s: float = 0.08,
    trough_search_radius_s: float = 0.12,
    min_peak_distance_s: float = 0.3,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    prominence = max(0.2 * robust_scale(detector_signal), 1e-3)
    peaks = _detect_peaks_by_finite_run(detector_signal, fs=fs, min_distance_s=min_peak_distance_s, prominence=prominence)
    base_features = {}
    beat_arrays: dict[str, np.ndarray] = {}
    if peaks.size < 3:
        return base_features, beat_arrays
    onsets, offsets, beat_peaks = _segment_between_troughs(detector_signal, peaks)
    detector_troughs = np.unique(np.concatenate([onsets, offsets])) if onsets.size and offsets.size else np.asarray([], dtype=int)
    beat_segments: list[np.ndarray] = []
    beat_rows = []
    peak_radius = max(int(round(peak_search_radius_s * fs)), 1)
    trough_radius = max(int(round(trough_search_radius_s * fs)), 1)
    refined_troughs = _refine_trough_sequence(measurement_signal, detector_troughs, trough_radius)
    if refined_troughs.size != detector_troughs.size:
        return base_features, beat_arrays
    trough_lookup = {
        int(det): int(ref)
        for det, ref in zip(detector_troughs.tolist(), refined_troughs.tolist())
        if int(ref) >= 0
    }
    refined_peak_lookup: dict[int, int] = {}
    for onset, offset, detector_peak in zip(onsets.tolist(), offsets.tolist(), beat_peaks.tolist()):
        if offset - onset < max(int(0.2 * fs), 2):
            continue
        peaks_in_beat = peaks[(peaks > onset) & (peaks < offset)]
        if peaks_in_beat.size == 0:
            continue
        if peaks_in_beat.size > 1:
            detector_peak = int(peaks_in_beat[np.argmax(detector_signal[peaks_in_beat])])
        if not (onset < detector_peak < offset):
            continue
        local_onset = trough_lookup.get(int(onset))
        local_offset = trough_lookup.get(int(offset))
        local_peak = refined_peak_lookup.get(int(detector_peak))
        if local_peak is None:
            local_peak = _refine_local_extreme(measurement_signal, detector_peak, peak_radius, "max", onset + 1, offset)
            if local_peak is not None:
                refined_peak_lookup[int(detector_peak)] = int(local_peak)
        if local_onset is None or local_offset is None or local_peak is None:
            continue
        if not (local_onset < local_peak < local_offset):
            continue
        local_measurement = np.asarray(measurement_signal[local_onset : local_offset + 1], dtype=np.float64)
        local_morphology = np.asarray(morphology_signal[local_onset : local_offset + 1], dtype=np.float64)
        if local_measurement.size < 2 or not np.isfinite(local_measurement).all():
            continue
        if local_morphology.size < 2 or not np.isfinite(local_morphology).all():
            continue
        foot = float(measurement_signal[local_onset])
        peak = float(measurement_signal[local_peak])
        next_foot = float(measurement_signal[local_offset])
        width_s = (local_offset - local_onset) / fs
        rise_s = (local_peak - local_onset) / fs
        decay_s = (local_offset - local_peak) / fs
        amplitude = peak - foot
        beat = local_measurement
        area = float(np.trapezoid(np.maximum(beat - foot, 0.0), dx=1.0 / fs))
        deriv = slope(local_morphology, fs)
        beat_rows.append({
            "rate_bpm": 60.0 / width_s if width_s > 0 else np.nan,
            "amplitude": amplitude,
            "width_s": width_s,
            "rise_s": rise_s,
            "decay_s": decay_s,
            "rise_slope": amplitude / rise_s if rise_s > 1e-8 else np.nan,
            "decay_slope": (peak - next_foot) / decay_s if decay_s > 1e-8 else np.nan,
            "area": area,
            "dpdt_max": float(np.max(deriv)) if deriv.size else np.nan,
            "dpdt_min": float(np.min(deriv)) if deriv.size else np.nan,
            "foot": foot,
            "peak": peak,
            "mean": float(np.mean(beat)),
            "sys_dias_ratio": safe_ratio(rise_s, decay_s),
            "start_idx": float(local_onset),
            "end_idx": float(local_offset),
            "peak_idx": float(local_peak),
        })
        beat_segments.append(local_morphology)
    if not beat_rows:
        return base_features, beat_arrays
    for key in beat_rows[0]:
        beat_arrays[key] = np.asarray([row[key] for row in beat_rows], dtype=np.float64)
    beat_arrays["_segments"] = np.asarray(beat_segments, dtype=object)
    return base_features, beat_arrays


def extract_abp_features(
    abp: np.ndarray,
    fs: int,
    micro_window_samples: int,
    template_points: int,
    detector_low_hz: float = 0.5,
    detector_high_hz: float = 12.0,
    morphology_high_hz: float = 20.0,
    peak_search_radius_s: float = 0.08,
    trough_search_radius_s: float = 0.12,
    min_pulse_bpm: float = 30.0,
    max_pulse_bpm: float = 220.0,
    quality_min_finite_fraction: float = 0.95,
    extreme_value_atol_fraction: float = 0.01,
    max_interp_gap_s: float = 0.2,
) -> dict[str, float]:
    quality = assess_signal_quality(
        abp,
        micro_window_samples=micro_window_samples,
        min_std=0.5,
        min_finite_fraction=quality_min_finite_fraction,
        extreme_value_atol_fraction=extreme_value_atol_fraction,
    )
    max_gap_samples = max(int(round(max_interp_gap_s * fs)), 0)
    detector_signal = butter_filter(
        abp,
        fs=fs,
        low_hz=detector_low_hz,
        high_hz=detector_high_hz,
        order=3,
        max_interp_gap_samples=max_gap_samples,
    )
    morphology_signal = butter_filter(
        abp,
        fs=fs,
        high_hz=morphology_high_hz,
        order=3,
        max_interp_gap_samples=max_gap_samples,
    )
    base_features, beats = _extract_pulsatile_beats(
        abp,
        detector_signal,
        morphology_signal,
        fs,
        "abp",
        template_points,
        peak_search_radius_s=peak_search_radius_s,
        trough_search_radius_s=trough_search_radius_s,
        min_peak_distance_s=60.0 / max_pulse_bpm,
    )
    features: dict[str, float] = {
        "abp_valid_micro_fraction": quality.valid_micro_fraction,
        "abp_missing_micro_fraction": quality.missing_micro_fraction,
        "abp_flatline_fraction": quality.flatline_fraction,
        "abp_extreme_value_fraction": quality.extreme_value_fraction,
    }
    features.update(base_features)
    if not beats:
        return features
    sbp = beats["peak"]
    dbp = beats["foot"]
    map_values = beats["mean"]
    pp = sbp - dbp
    plausible_sbp = (sbp >= 50.0) & (sbp <= 260.0)
    plausible_dbp = (dbp >= 20.0) & (dbp <= 180.0)
    width_tolerance_s = 0.5 / fs
    valid_duration = (beats["width_s"] >= (60.0 / max_pulse_bpm) - width_tolerance_s) & (
        beats["width_s"] <= (60.0 / min_pulse_bpm) + width_tolerance_s
    )
    valid_pulse = plausible_sbp & plausible_dbp & (pp > 0) & valid_duration
    segments = beats.get("_segments", np.asarray([], dtype=object))
    valid_segments = [segments[idx] for idx, flag in enumerate(valid_pulse.tolist()) if flag and idx < len(segments)]
    features.update({
        "abp_sbp_median_mmhg": safe_nanstat(np.median, sbp[valid_pulse]),
        "abp_dbp_median_mmhg": safe_nanstat(np.median, dbp[valid_pulse]),
        "abp_map_median_mmhg": safe_nanstat(np.median, map_values[valid_pulse]),
        "abp_pulse_pressure_median_mmhg": safe_nanstat(np.median, pp[valid_pulse]),
        "abp_pulse_rate_bpm": safe_nanstat(np.median, beats["rate_bpm"][valid_pulse]),
        "abp_sbp_sd_mmhg": safe_nanstat(np.std, sbp[valid_pulse]),
        "abp_sbp_iqr_mmhg": nan_iqr(sbp[valid_pulse]),
        "abp_dbp_sd_mmhg": safe_nanstat(np.std, dbp[valid_pulse]),
        "abp_dbp_iqr_mmhg": nan_iqr(dbp[valid_pulse]),
        "abp_map_sd_mmhg": safe_nanstat(np.std, map_values[valid_pulse]),
        "abp_map_iqr_mmhg": nan_iqr(map_values[valid_pulse]),
        "abp_pp_sd_mmhg": safe_nanstat(np.std, pp[valid_pulse]),
        "abp_pp_iqr_mmhg": nan_iqr(pp[valid_pulse]),
        "abp_upstroke_slope_median": safe_nanstat(np.median, beats["rise_slope"][valid_pulse]),
        "abp_dpdt_max_median": safe_nanstat(np.median, beats["dpdt_max"][valid_pulse]),
        "abp_dpdt_min_median": safe_nanstat(np.median, beats["dpdt_min"][valid_pulse]),
        "abp_pulse_area_median": safe_nanstat(np.median, beats["area"][valid_pulse]),
        "abp_pulse_width_median_s": safe_nanstat(np.median, beats["width_s"][valid_pulse]),
        "abp_upstroke_duration_median_s": safe_nanstat(np.median, beats["rise_s"][valid_pulse]),
        "abp_decay_duration_median_s": safe_nanstat(np.median, beats["decay_s"][valid_pulse]),
        "abp_upstroke_decay_ratio_median": safe_nanstat(np.median, beats["sys_dias_ratio"][valid_pulse]),
        "abp_morphology_consistency": median_template_correlation(valid_segments, template_points),
        "abp_valid_pulse_fraction": float(np.mean(valid_pulse)),
        "abp_plausible_sbp_fraction": float(np.mean(plausible_sbp)),
        "abp_plausible_dbp_fraction": float(np.mean(plausible_dbp)),
        "abp_sbp_gt_dbp_fraction": float(np.mean(sbp > dbp)),
    })
    return features


def extract_pleth_features(
    pleth: np.ndarray,
    fs: int,
    micro_window_samples: int,
    template_points: int,
    detector_low_hz: float = 0.5,
    detector_high_hz: float = 8.0,
    morphology_high_hz: float = 8.0,
    peak_search_radius_s: float = 0.08,
    trough_search_radius_s: float = 0.12,
    quality_min_finite_fraction: float = 0.95,
    extreme_value_atol_fraction: float = 0.01,
    max_interp_gap_s: float = 0.2,
) -> dict[str, float]:
    quality = assess_signal_quality(
        pleth,
        micro_window_samples=micro_window_samples,
        min_std=1e-4,
        min_finite_fraction=quality_min_finite_fraction,
        extreme_value_atol_fraction=extreme_value_atol_fraction,
    )
    max_gap_samples = max(int(round(max_interp_gap_s * fs)), 0)
    detector_signal = butter_filter(
        pleth,
        fs=fs,
        low_hz=detector_low_hz,
        high_hz=detector_high_hz,
        order=3,
        max_interp_gap_samples=max_gap_samples,
    )
    morphology_signal = butter_filter(
        pleth,
        fs=fs,
        high_hz=morphology_high_hz,
        order=3,
        max_interp_gap_samples=max_gap_samples,
    )
    base_features, beats = _extract_pulsatile_beats(
        pleth,
        detector_signal,
        morphology_signal,
        fs,
        "pleth",
        template_points,
        peak_search_radius_s=peak_search_radius_s,
        trough_search_radius_s=trough_search_radius_s,
        min_peak_distance_s=0.3,
    )
    features: dict[str, float] = {
        "pleth_valid_micro_fraction": quality.valid_micro_fraction,
        "pleth_missing_micro_fraction": quality.missing_micro_fraction,
        "pleth_flatline_fraction": quality.flatline_fraction,
        "pleth_extreme_value_fraction": quality.extreme_value_fraction,
    }
    features.update(base_features)
    if not beats:
        return features
    valid_pulse = (
        (beats["amplitude"] > 0)
        & (beats["width_s"] >= 0.3)
        & (beats["width_s"] <= 2.0)
        & np.isfinite(beats["rise_s"])
        & np.isfinite(beats["decay_s"])
    )
    segments = beats.get("_segments", np.asarray([], dtype=object))
    valid_segments = [segments[idx] for idx, flag in enumerate(valid_pulse.tolist()) if flag and idx < len(segments)]
    features.update({
        "pleth_pulse_rate_bpm": safe_nanstat(np.median, beats["rate_bpm"][valid_pulse]),
        "pleth_amplitude_median": safe_nanstat(np.median, beats["amplitude"][valid_pulse]),
        "pleth_amplitude_iqr": nan_iqr(beats["amplitude"][valid_pulse]),
        "pleth_rise_time_median_s": safe_nanstat(np.median, beats["rise_s"][valid_pulse]),
        "pleth_decay_time_median_s": safe_nanstat(np.median, beats["decay_s"][valid_pulse]),
        "pleth_rise_slope_median": safe_nanstat(np.median, beats["rise_slope"][valid_pulse]),
        "pleth_decay_slope_median": safe_nanstat(np.median, beats["decay_slope"][valid_pulse]),
        "pleth_width_median_s": safe_nanstat(np.median, beats["width_s"][valid_pulse]),
        "pleth_area_median": safe_nanstat(np.median, beats["area"][valid_pulse]),
        "pleth_morphology_consistency": median_template_correlation(valid_segments, template_points),
        "pleth_valid_pulse_fraction": float(np.mean(valid_pulse)),
    })
    return features
