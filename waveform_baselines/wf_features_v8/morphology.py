from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from waveform_baselines.wf_features.utils import nan_iqr, safe_nanstat, slope


def detect_dicrotic_notch(
    beat: np.ndarray,
    peak_idx: int,
    fs: int,
    score_separation: float = 0.15,
    min_candidate_score: float = 2.4,
) -> tuple[int, float, float] | None:
    arr = np.asarray(beat, dtype=np.float64)
    if arr.size < max(30, int(0.35 * fs)) or not np.isfinite(arr).all():
        return None
    if peak_idx <= 1 or peak_idx >= arr.size - 5:
        return None
    foot = float(arr[0])
    peak = float(arr[peak_idx])
    amp = peak - foot
    if amp <= 1e-8:
        return None
    start = peak_idx + max(2, int(round(0.08 * fs)))
    end = min(arr.size - 3, peak_idx + int(round(0.45 * fs)))
    if end <= start + 4:
        return None
    d1 = slope(arr, fs)
    d2 = slope(d1, fs)
    candidates: list[tuple[float, int]] = []
    for idx in range(start + 1, end - 1):
        is_local_min = arr[idx] < arr[idx - 1] and arr[idx] <= arr[idx + 1]
        derivative_crosses = d1[idx - 1] < 0.0 and d1[idx + 1] > 0.0
        curvature_positive = d2[idx] > 0.0
        preceding_drop = peak - float(arr[idx])
        recovery = float(np.max(arr[idx + 1 : min(end + 1, idx + int(0.12 * fs) + 1)])) - float(arr[idx])
        if is_local_min and derivative_crosses and curvature_positive and preceding_drop >= 0.05 * amp and recovery >= 0.02 * amp:
            notch_frac = idx / max(arr.size - 1, 1)
            if 0.25 <= notch_frac <= 0.70:
                timing_score = 1.0
            else:
                timing_score = max(0.0, 1.0 - min(abs(notch_frac - 0.475) / 0.35, 1.0))
            drop_score = min(preceding_drop / max(0.20 * amp, 1e-8), 2.0) / 2.0
            recovery_score = min(recovery / max(0.08 * amp, 1e-8), 2.0) / 2.0
            curvature_score = min(float(d2[idx]) / max(np.nanpercentile(np.abs(d2[start:end]), 75), 1e-8), 3.0) / 3.0
            cross_score = min(abs(float(d1[idx + 1] - d1[idx - 1])) / max(np.nanpercentile(np.abs(d1[start:end]), 75), 1e-8), 3.0) / 3.0
            candidates.append((timing_score + drop_score + recovery_score + curvature_score + cross_score, idx))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if candidates[0][0] < min_candidate_score:
        return None
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < score_separation:
        return None
    notch = int(candidates[0][1])
    ratio = (float(arr[notch]) - foot) / amp
    if ratio < 0.05 or ratio > 1.05:
        return None
    return notch, (notch - peak_idx) / fs, ratio


def detect_diastolic_peak(
    beat: np.ndarray,
    peak_idx: int,
    notch_idx: int,
    fs: int,
    score_separation: float = 0.10,
    min_candidate_score: float = 2.0,
) -> int | None:
    arr = np.asarray(beat, dtype=np.float64)
    if arr.size < max(30, int(0.35 * fs)) or not np.isfinite(arr).all():
        return None
    if peak_idx <= 0 or peak_idx >= notch_idx or notch_idx <= 1 or notch_idx >= arr.size - 5:
        return None
    foot = float(arr[0])
    peak = float(arr[peak_idx])
    amp = peak - foot
    if amp <= 1e-8:
        return None
    start = notch_idx + max(2, int(round(0.05 * fs)))
    end = min(arr.size - 3, notch_idx + int(round(0.45 * fs)))
    if end <= start + 4:
        return None
    post = arr[start:end]
    peaks, props = find_peaks(post, prominence=max(0.03 * amp, 1e-8))
    if peaks.size == 0:
        return None
    prominences = props.get("prominences", np.zeros(peaks.size, dtype=np.float64))
    scored: list[tuple[float, int]] = []
    for pos, prom in zip(peaks.tolist(), prominences.tolist()):
        cand = start + int(pos)
        rise = float(arr[cand] - arr[notch_idx])
        if rise < 0.02 * amp:
            continue
        frac = cand / max(arr.size - 1, 1)
        timing_score = 1.0 if 0.35 <= frac <= 0.85 else max(0.0, 1.0 - min(abs(frac - 0.60) / 0.35, 1.0))
        prominence_score = min(float(prom) / max(0.08 * amp, 1e-8), 2.0) / 2.0
        rise_score = min(rise / max(0.08 * amp, 1e-8), 2.0) / 2.0
        early_score = 1.0 - 0.35 * ((cand - start) / max(end - start, 1))
        scored.append((timing_score + prominence_score + rise_score + early_score, cand))
    if not scored:
        return None
    scored.sort(reverse=True)
    best = int(scored[0][1])
    if scored[0][0] < min_candidate_score:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < score_separation:
        return None
    if not (arr[best] > arr[best - 1] and arr[best] >= arr[best + 1]):
        return None
    return best


def fit_diastolic_tau(
    beat: np.ndarray,
    notch_idx: int,
    fs: int,
    peak_idx: int | None = None,
    pulse_pressure: float | None = None,
    diastolic_peak_idx: int | None = None,
    morphology_beat: np.ndarray | None = None,
    min_r2: float = 0.80,
    tau_bounds_s: tuple[float, float] = (0.08, 3.0),
    post_notch_exclusion_s: float = 0.08,
    post_rebound_exclusion_s: float = 0.04,
    min_dynamic_range_fraction: float = 0.08,
    tau_bound_margin_fraction: float = 0.03,
) -> tuple[float, float] | None:
    arr = np.asarray(beat, dtype=np.float64)
    if arr.size < max(30, int(0.35 * fs)) or not np.isfinite(arr).all():
        return None
    if notch_idx < 1 or notch_idx >= arr.size - int(0.20 * fs):
        return None
    if pulse_pressure is None:
        if peak_idx is not None and 0 < int(peak_idx) < arr.size:
            pulse_pressure = float(arr[int(peak_idx)] - arr[0])
        else:
            pulse_pressure = float(np.nanmax(arr) - arr[0])
    pulse_pressure = float(pulse_pressure)
    if pulse_pressure <= 1e-8:
        return None
    fit_start = notch_idx + max(1, int(round(post_notch_exclusion_s * fs)))
    rebound_margin = max(1, int(round(post_rebound_exclusion_s * fs)))
    if diastolic_peak_idx is not None and notch_idx < int(diastolic_peak_idx) < arr.size - 2:
        fit_start = max(fit_start, int(diastolic_peak_idx) + rebound_margin)
    else:
        search_signal = np.asarray(morphology_beat if morphology_beat is not None else arr, dtype=np.float64)
        if search_signal.size == arr.size and np.isfinite(search_signal).all():
            search_end = min(arr.size - 2, notch_idx + int(round(0.30 * fs)))
            if search_end > notch_idx + 3:
                if peak_idx is not None and 0 < int(peak_idx) < search_signal.size:
                    pulse_pressure_m = float(search_signal[int(peak_idx)] - search_signal[0])
                else:
                    pulse_pressure_m = float(np.nanmax(search_signal) - search_signal[0])
                prom = max(0.03 * pulse_pressure_m, 1e-8)
                peaks, _ = find_peaks(search_signal[notch_idx:search_end], prominence=prom)
                if peaks.size:
                    fit_start = max(fit_start, notch_idx + int(peaks[0]) + rebound_margin)
    if fit_start >= arr.size - int(0.20 * fs):
        return None
    y = arr[fit_start:]
    t = np.arange(y.size, dtype=np.float64) / fs
    if y.size < max(8, int(0.20 * fs)) or y[0] <= y[-1]:
        return None
    if float(np.ptp(y)) < min_dynamic_range_fraction * pulse_pressure:
        return None
    if np.nanmedian(np.diff(y)) >= 0:
        return None

    def model(tvals: np.ndarray, p_inf: float, amp: float, tau: float) -> np.ndarray:
        return p_inf + amp * np.exp(-tvals / tau)

    lower = [float(np.min(y) - 40.0), 1e-6, tau_bounds_s[0]]
    upper = [float(np.min(y) + 20.0), float(np.ptp(y) * 3.0 + 1.0), tau_bounds_s[1]]
    p0 = [float(y[-1] - 5.0), float(max(y[0] - y[-1], 1.0)), 0.5]
    try:
        popt, _ = curve_fit(model, t, y, p0=p0, bounds=(lower, upper), maxfev=2000)
    except Exception:
        return None
    pred = model(t, *popt)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-12:
        return None
    r2 = 1.0 - ss_res / ss_tot
    tau = float(popt[2])
    bound_margin = tau_bound_margin_fraction * (tau_bounds_s[1] - tau_bounds_s[0])
    if (
        not np.isfinite(tau)
        or tau <= tau_bounds_s[0] + bound_margin
        or tau >= tau_bounds_s[1] - bound_margin
        or r2 < min_r2
    ):
        return None
    return tau, float(r2)


def advanced_abp_morphology_from_beats(beats: dict[str, np.ndarray], fs: int, min_beats: int = 5, config: object | None = None) -> dict[str, float]:
    if not beats:
        return {}
    valid = beats.get("valid", np.ones_like(beats["peak"], dtype=bool))
    raw_segments = beats.get("raw_segments")
    morphology_segments = beats.get("morphology_segments", beats.get("_segments"))
    notch_times = []
    notch_ratios = []
    sys_area = []
    dias_area = []
    area_ratios = []
    sys_area_fractions = []
    diastolic_peak_time_fractions = []
    diastolic_peak_pressure_ratios = []
    notch_to_diastolic_peak_delays = []
    taus = []
    r2s = []
    eligible_morphology_beats = 0
    score_sep = float(getattr(config, "abp_notch_candidate_score_separation", 0.15))
    min_score = float(getattr(config, "abp_notch_min_candidate_score", 2.4))
    dpeak_score_sep = float(getattr(config, "abp_diastolic_peak_candidate_score_separation", 0.10))
    dpeak_min_score = float(getattr(config, "abp_diastolic_peak_min_candidate_score", 2.0))
    if config is not None:
        min_beats = int(getattr(config, "min_abp_morphology_beats", min_beats))
    tau_dyn = float(getattr(config, "tau_min_dynamic_range_fraction", 0.08))
    tau_margin = float(getattr(config, "tau_bound_margin_fraction", 0.03))
    tau_rebound = float(getattr(config, "tau_post_rebound_exclusion_seconds", 0.04))
    for idx, ok in enumerate(valid.tolist()):
        if not ok:
            continue
        peak_idx = int(round(float(beats["peak_idx"][idx] - beats["start_idx"][idx])))
        if raw_segments is None or morphology_segments is None or idx >= len(raw_segments) or idx >= len(morphology_segments):
            continue
        raw_seg = np.asarray(raw_segments[idx], dtype=np.float64)
        morph_seg = np.asarray(morphology_segments[idx], dtype=np.float64)
        if raw_seg.size != morph_seg.size or raw_seg.size < 3 or not np.isfinite(raw_seg).all() or not np.isfinite(morph_seg).all():
            continue
        if peak_idx <= 0 or peak_idx >= raw_seg.size - 1:
            continue
        foot = float(raw_seg[0])
        peak = float(raw_seg[peak_idx])
        amp = peak - foot
        if amp <= 1e-8:
            continue
        eligible_morphology_beats += 1
        notch = detect_dicrotic_notch(morph_seg, peak_idx, fs, score_separation=score_sep, min_candidate_score=min_score)
        if notch is None:
            continue
        notch_idx, notch_time, notch_ratio = notch
        next_foot = float(raw_seg[-1])
        sys = float(np.trapezoid(np.maximum(raw_seg[: notch_idx + 1] - foot, 0.0), dx=1.0 / fs))
        dias = float(np.trapezoid(np.maximum(raw_seg[notch_idx:] - next_foot, 0.0), dx=1.0 / fs))
        baseline = np.linspace(foot, next_foot, raw_seg.size)
        sys_frac_area = float(np.trapezoid(np.maximum(raw_seg[: notch_idx + 1] - baseline[: notch_idx + 1], 0.0), dx=1.0 / fs))
        dias_frac_area = float(np.trapezoid(np.maximum(raw_seg[notch_idx:] - baseline[notch_idx:], 0.0), dx=1.0 / fs))
        if sys_frac_area + dias_frac_area > 1e-12:
            sys_area_fractions.append(sys_frac_area / (sys_frac_area + dias_frac_area))
        if sys_frac_area > 0 and dias_frac_area > 0:
            area_ratios.append(sys_frac_area / dias_frac_area)
        if sys > 0:
            sys_area.append(sys)
        if dias > 0:
            dias_area.append(dias)
        notch_times.append(notch_time)
        notch_ratios.append((float(raw_seg[notch_idx]) - foot) / amp)
        diastolic_peak = detect_diastolic_peak(
            morph_seg,
            peak_idx,
            notch_idx,
            fs,
            score_separation=dpeak_score_sep,
            min_candidate_score=dpeak_min_score,
        )
        if diastolic_peak is not None:
            diastolic_peak_time_fractions.append(float(diastolic_peak / max(raw_seg.size - 1, 1)))
            diastolic_peak_pressure_ratios.append(float((raw_seg[diastolic_peak] - foot) / amp))
            notch_to_diastolic_peak_delays.append(float((diastolic_peak - notch_idx) / fs))
        tau = fit_diastolic_tau(
            raw_seg,
            notch_idx,
            fs,
            peak_idx=peak_idx,
            pulse_pressure=amp,
            diastolic_peak_idx=diastolic_peak,
            morphology_beat=morph_seg,
            min_dynamic_range_fraction=tau_dyn,
            tau_bound_margin_fraction=tau_margin,
            post_rebound_exclusion_s=tau_rebound,
        )
        if tau is not None:
            taus.append(tau[0])
            r2s.append(tau[1])
    if eligible_morphology_beats < min_beats:
        return {}
    out = {
        "abp_dicrotic_notch_presence_fraction": float(len(notch_times) / eligible_morphology_beats),
        "abp_diastolic_peak_presence_fraction": float(len(diastolic_peak_time_fractions) / eligible_morphology_beats),
        "abp_diastolic_tau_valid_fraction": float(len(taus) / eligible_morphology_beats),
    }
    if len(notch_times) >= min_beats:
        out["abp_dicrotic_notch_time_median_s"] = safe_nanstat(np.median, np.asarray(notch_times))
        out["abp_dicrotic_notch_pressure_ratio"] = safe_nanstat(np.median, np.asarray(notch_ratios))
        out["abp_dicrotic_notch_pressure_ratio_iqr"] = nan_iqr(np.asarray(notch_ratios))
    if len(sys_area) >= min_beats:
        out["abp_systolic_area_median"] = safe_nanstat(np.median, np.asarray(sys_area))
    if len(dias_area) >= min_beats:
        out["abp_diastolic_area_median"] = safe_nanstat(np.median, np.asarray(dias_area))
    if len(area_ratios) >= min_beats:
        out["abp_systolic_diastolic_area_ratio"] = safe_nanstat(np.median, np.asarray(area_ratios))
    if len(sys_area_fractions) >= min_beats:
        out["abp_systolic_area_fraction_median"] = safe_nanstat(np.median, np.asarray(sys_area_fractions))
        out["abp_systolic_area_fraction_iqr"] = nan_iqr(np.asarray(sys_area_fractions))
    if len(diastolic_peak_time_fractions) >= min_beats:
        out["abp_diastolic_peak_time_fraction_median"] = safe_nanstat(np.median, np.asarray(diastolic_peak_time_fractions))
        out["abp_diastolic_peak_pressure_ratio_median"] = safe_nanstat(np.median, np.asarray(diastolic_peak_pressure_ratios))
        out["abp_notch_to_diastolic_peak_delay_median_s"] = safe_nanstat(np.median, np.asarray(notch_to_diastolic_peak_delays))
    if len(taus) >= min_beats:
        out["abp_diastolic_tau_median_s"] = safe_nanstat(np.median, np.asarray(taus))
        out["abp_diastolic_decay_fit_r2"] = safe_nanstat(np.median, np.asarray(r2s))
    return out
