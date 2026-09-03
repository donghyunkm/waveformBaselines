from __future__ import annotations

import warnings

import numpy as np
from scipy import signal
try:
    from wfdb import processing as wfdb_processing
except ImportError:  # pragma: no cover - exercised only without wfdb installed
    wfdb_processing = None

from .quality import assess_signal_quality
from .utils import (
    butter_filter,
    count_plausible_intervals,
    finite_runs,
    median_template_correlation,
    nan_iqr,
    interpolate_short_gaps,
    robust_scale,
    safe_nanstat,
    slope,
)


def successive_valid_rr_diffs(rr: np.ndarray, valid_rr_mask: np.ndarray) -> np.ndarray:
    rr = np.asarray(rr, dtype=np.float64)
    valid_rr_mask = np.asarray(valid_rr_mask, dtype=bool)
    if rr.shape != valid_rr_mask.shape:
        raise ValueError("rr and valid_rr_mask must have the same shape")
    if rr.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(rr)[valid_rr_mask[:-1] & valid_rr_mask[1:]]


def _fallback_qrs_energy_peaks(segment: np.ndarray, fs: int, prominence: float) -> np.ndarray:
    arr = np.asarray(segment, dtype=np.float64)
    if arr.size < max(int(0.5 * fs), 3):
        return np.asarray([], dtype=int)
    finite = np.isfinite(arr)
    if not finite.all():
        return np.asarray([], dtype=int)
    centered = arr - np.median(arr)
    deriv = np.gradient(centered)
    energy = deriv * deriv
    win = max(int(round(0.10 * fs)), 1)
    if win > 1:
        energy = np.convolve(energy, np.ones(win, dtype=np.float64) / win, mode="same")
    scale = robust_scale(energy)
    height = max(float(np.median(energy) + 3.0 * scale), 1e-8)
    min_distance = max(int(round(0.25 * fs)), 1)
    candidates, _ = signal.find_peaks(energy, distance=min_distance, height=height)
    if candidates.size == 0:
        abs_signal = np.abs(centered)
        candidates, _ = signal.find_peaks(abs_signal, distance=min_distance, prominence=max(prominence, 1e-3))
    refined: list[int] = []
    radius = max(int(round(0.08 * fs)), 1)
    for idx in candidates.tolist():
        start = max(0, int(idx) - radius)
        end = min(arr.size, int(idx) + radius + 1)
        local = np.abs(centered[start:end])
        if local.size == 0:
            continue
        refined.append(start + int(np.argmax(local)))
    if not refined:
        return np.asarray([], dtype=int)
    refined_arr = np.asarray(sorted(set(refined)), dtype=int)
    keep: list[int] = []
    for idx in refined_arr.tolist():
        if not keep or idx - keep[-1] >= min_distance:
            keep.append(idx)
        else:
            prev = keep[-1]
            if abs(centered[idx]) > abs(centered[prev]):
                keep[-1] = idx
    return np.asarray(keep, dtype=int)


def _inc_diagnostic(diagnostics: dict[str, int] | None, key: str, amount: int = 1) -> None:
    if diagnostics is not None:
        diagnostics[key] = int(diagnostics.get(key, 0)) + amount


def _detect_r_peaks_by_finite_run(
    xqrs_signal: np.ndarray,
    fs: int,
    prominence: float,
    fallback_signal: np.ndarray | None = None,
    detector: str = "xqrs",
    allow_energy_fallback: bool = True,
    diagnostics: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if detector not in {"xqrs", "energy"}:
        raise ValueError(f"Unsupported ECG detector {detector!r}; expected 'xqrs' or 'energy'")
    if detector == "xqrs" and wfdb_processing is None:
        raise RuntimeError(
            "ECG detector is configured as XQRS, but wfdb is unavailable. "
            "Install/activate the expected environment or explicitly configure the energy detector."
        )

    peaks: list[int] = []
    run_ids: list[int] = []
    fallback_arr = np.asarray(fallback_signal if fallback_signal is not None else xqrs_signal, dtype=np.float64)
    xqrs_arr = np.asarray(xqrs_signal, dtype=np.float64)
    min_distance = max(int(round(0.25 * fs)), 1)
    min_run_samples = max(int(round(2.0 * fs)), min_distance + 1)
    for run_id, (start, end) in enumerate(finite_runs(np.isfinite(xqrs_arr))):
        _inc_diagnostic(diagnostics, "ecg_detector_runs_total")
        segment = xqrs_arr[start:end]
        if segment.size < min_run_samples:
            _inc_diagnostic(diagnostics, "ecg_runs_with_no_detection")
            continue
        local_peaks = np.asarray([], dtype=int)
        used_detector = None
        if detector == "xqrs":
            _inc_diagnostic(diagnostics, "ecg_xqrs_runs_attempted")
            try:
                local_peaks = np.asarray(
                    wfdb_processing.xqrs_detect(sig=segment, fs=fs, verbose=False),
                    dtype=int,
                )
            except Exception:
                _inc_diagnostic(diagnostics, "ecg_xqrs_runs_failed")
                _inc_diagnostic(diagnostics, "ecg_xqrs_runs_exception")
                local_peaks = np.asarray([], dtype=int)
            local_peaks = local_peaks[(local_peaks >= 0) & (local_peaks < segment.size)]
            if local_peaks.size > 0:
                used_detector = "xqrs"
                _inc_diagnostic(diagnostics, "ecg_xqrs_runs_used")
            else:
                _inc_diagnostic(diagnostics, "ecg_xqrs_runs_zero_peaks")
                if allow_energy_fallback:
                    local_peaks = _fallback_qrs_energy_peaks(
                        fallback_arr[start:end], fs=fs, prominence=prominence
                    )
                    if local_peaks.size > 0:
                        used_detector = "energy_fallback"
                        _inc_diagnostic(diagnostics, "ecg_energy_fallback_runs_used")
        else:
            local_peaks = _fallback_qrs_energy_peaks(fallback_arr[start:end], fs=fs, prominence=prominence)
            if local_peaks.size > 0:
                used_detector = "energy"
                _inc_diagnostic(diagnostics, "ecg_energy_runs_used")

        if local_peaks.size == 0:
            _inc_diagnostic(diagnostics, "ecg_runs_with_no_detection")
            continue
        local_peaks = local_peaks[(local_peaks >= 0) & (local_peaks < segment.size)]
        if local_peaks.size == 0:
            _inc_diagnostic(diagnostics, "ecg_runs_with_no_detection")
            continue
        global_peaks = start + np.unique(local_peaks)
        peaks.extend(global_peaks.tolist())
        run_ids.extend([run_id] * global_peaks.size)
    if not peaks:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    peaks_arr = np.asarray(peaks, dtype=int)
    run_ids_arr = np.asarray(run_ids, dtype=int)
    order = np.argsort(peaks_arr, kind="stable")
    peaks_arr = peaks_arr[order]
    run_ids_arr = run_ids_arr[order]
    keep = np.concatenate([[True], np.diff(peaks_arr) != 0])
    return peaks_arr[keep], run_ids_arr[keep]


def _rr_validity(peaks: np.ndarray, peak_run_ids: np.ndarray, min_bpm: float = 30.0, max_bpm: float = 220.0, fs: int = 125) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rr = np.diff(peaks) / fs
    if rr.size == 0:
        return rr, np.asarray([], dtype=bool), np.asarray([], dtype=bool)
    same_run = peak_run_ids[:-1] == peak_run_ids[1:]
    plausible = (rr >= 60.0 / max_bpm) & (rr <= 60.0 / min_bpm)
    return rr, same_run, same_run & plausible


def _map_detector_peaks_to_morphology(
    detector_peaks: np.ndarray,
    morphology_signal: np.ndarray,
    search_radius: int,
    min_finite_fraction: float = 0.8,
) -> np.ndarray:
    mapped: list[int] = []
    n = morphology_signal.size
    for peak in detector_peaks.tolist():
        start = max(0, int(peak) - search_radius)
        end = min(n, int(peak) + search_radius + 1)
        local = morphology_signal[start:end]
        finite = np.isfinite(local)
        if local.size == 0 or np.mean(finite) < min_finite_fraction or not finite.any():
            continue
        finite_offsets = np.flatnonzero(finite)
        local_finite = local[finite]
        centered = local_finite - np.median(local_finite)
        local_idx = finite_offsets[int(np.argmax(np.abs(centered)))]
        mapped.append(start + int(local_idx))
    if not mapped:
        return np.asarray([], dtype=int)
    return np.unique(np.asarray(mapped, dtype=int))


def _peak_widths_with_finite_runs(morphology_signal: np.ndarray, peaks: np.ndarray, fs: int) -> np.ndarray:
    widths: list[float] = []
    for start, end in finite_runs(np.isfinite(morphology_signal)):
        run_peaks = peaks[(peaks >= start) & (peaks < end)]
        if run_peaks.size == 0:
            continue
        local_peaks = run_peaks - start
        local_signal = morphology_signal[start:end]
        if local_signal.size < 3:
            continue
        local_energy = np.abs(local_signal - np.median(local_signal))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            local_widths = signal.peak_widths(local_energy, local_peaks, rel_height=0.5)[0] / fs
        widths.extend(local_widths[np.isfinite(local_widths) & (local_widths > 0)].tolist())
    return np.asarray(widths, dtype=np.float64)


def extract_ecg_features(
    ecg: np.ndarray,
    fs: int,
    micro_window_samples: int,
    template_points: int,
    hrv_min_beats: int,
    hrv_min_successive_pairs: int = 3,
    detector: str = "xqrs",
    allow_energy_fallback: bool = True,
    detector_low_hz: float = 5.0,
    detector_high_hz: float = 20.0,
    morphology_low_hz: float = 0.5,
    morphology_high_hz: float = 40.0,
    peak_search_radius_s: float = 0.08,
    quality_min_finite_fraction: float = 0.95,
    extreme_value_atol_fraction: float = 0.01,
    max_interp_gap_s: float = 0.2,
    diagnostics: dict[str, int] | None = None,
) -> dict[str, float]:
    quality = assess_signal_quality(
        ecg,
        micro_window_samples=micro_window_samples,
        min_std=1e-3,
        min_finite_fraction=quality_min_finite_fraction,
        extreme_value_atol_fraction=extreme_value_atol_fraction,
    )
    max_gap_samples = max(int(round(max_interp_gap_s * fs)), 0)
    xqrs_signal, xqrs_usable = interpolate_short_gaps(ecg, max_gap_samples)
    xqrs_signal = np.asarray(xqrs_signal, dtype=np.float64)
    xqrs_signal[~xqrs_usable] = np.nan
    detector_signal = butter_filter(
        ecg,
        fs=fs,
        low_hz=detector_low_hz,
        high_hz=detector_high_hz,
        order=3,
        max_interp_gap_samples=max_gap_samples,
    )
    morphology_signal = butter_filter(
        ecg,
        fs=fs,
        low_hz=morphology_low_hz,
        high_hz=morphology_high_hz,
        order=3,
        max_interp_gap_samples=max_gap_samples,
    )
    prominence = max(0.2 * robust_scale(detector_signal), 1e-3)
    peaks, peak_run_ids = _detect_r_peaks_by_finite_run(
        xqrs_signal,
        fs=fs,
        prominence=prominence,
        fallback_signal=detector_signal,
        detector=detector,
        allow_energy_fallback=allow_energy_fallback,
        diagnostics=diagnostics,
    )
    features: dict[str, float] = {
        "ecg_valid_micro_fraction": quality.valid_micro_fraction,
        "ecg_missing_micro_fraction": quality.missing_micro_fraction,
        "ecg_flatline_fraction": quality.flatline_fraction,
        "ecg_extreme_value_fraction": quality.extreme_value_fraction,
    }
    if peaks.size < 2:
        features.update({
            "ecg_plausible_beat_fraction": float("nan"),
            "ecg_max_abs_slope": safe_nanstat(np.max, np.abs(slope(morphology_signal, fs))),
            "ecg_morphology_consistency": float("nan"),
        })
        return features

    rr, same_run_rr_mask, valid_rr_mask = _rr_validity(peaks, peak_run_ids, fs=fs)
    valid_rr = rr[valid_rr_mask]
    search_radius = max(int(round(peak_search_radius_s * fs)), 1)
    morph_peaks = _map_detector_peaks_to_morphology(peaks, morphology_signal, search_radius)
    morph_peaks = morph_peaks[np.isfinite(ecg[morph_peaks])]
    if morph_peaks.size < 2:
        features.update({
            "ecg_plausible_beat_fraction": count_plausible_intervals(rr[same_run_rr_mask], min_bpm=30.0, max_bpm=220.0),
            "ecg_max_abs_slope": safe_nanstat(np.max, np.abs(slope(morphology_signal, fs))),
            "ecg_morphology_consistency": float("nan"),
        })
        return features
    r_amp = ecg[morph_peaks]
    widths = _peak_widths_with_finite_runs(morphology_signal, morph_peaks, fs)
    beat_segments: list[np.ndarray] = []
    left = int(0.12 * fs)
    right = int(0.18 * fs)
    for peak in morph_peaks:
        start = peak - left
        end = peak + right
        if start < 0 or end > ecg.size:
            continue
        segment = morphology_signal[start:end]
        if np.isfinite(segment).all():
            beat_segments.append(segment)

    diff_rr = successive_valid_rr_diffs(rr, valid_rr_mask)
    hr = 60.0 / valid_rr if valid_rr.size else np.asarray([], dtype=np.float64)
    features.update({
        "ecg_hr_bpm": safe_nanstat(np.median, hr),
        "ecg_rr_median_s": safe_nanstat(np.median, valid_rr),
        "ecg_rr_iqr_s": nan_iqr(valid_rr),
        "ecg_rr_min_s": safe_nanstat(np.min, valid_rr),
        "ecg_rr_max_s": safe_nanstat(np.max, valid_rr),
        "ecg_r_amp_median": safe_nanstat(np.median, r_amp),
        "ecg_r_amp_iqr": nan_iqr(r_amp),
        "ecg_qrs_width_median_s": safe_nanstat(np.median, widths),
        "ecg_qrs_width_iqr_s": nan_iqr(widths),
        "ecg_max_abs_slope": safe_nanstat(np.max, np.abs(slope(morphology_signal, fs))),
        "ecg_morphology_consistency": median_template_correlation(beat_segments, template_points),
        "ecg_plausible_beat_fraction": count_plausible_intervals(rr[same_run_rr_mask], min_bpm=30.0, max_bpm=220.0),
    })
    sdnn = safe_nanstat(np.std, valid_rr) if valid_rr.size + 1 >= hrv_min_beats else float("nan")
    if diff_rr.size >= hrv_min_successive_pairs:
        rmssd = float(np.sqrt(np.mean(np.square(diff_rr))))
        sdsd = safe_nanstat(np.std, diff_rr)
        pnn20 = float(np.mean(np.abs(diff_rr) > 0.020))
        pnn50 = float(np.mean(np.abs(diff_rr) > 0.050))
    else:
        rmssd = float("nan")
        sdsd = float("nan")
        pnn20 = float("nan")
        pnn50 = float("nan")
    features.update({
        "ecg_hrv_sdnn_s": sdnn,
        "ecg_hrv_rmssd_s": rmssd,
        "ecg_hrv_sdsd_s": sdsd,
        "ecg_hrv_pnn20": pnn20,
        "ecg_hrv_pnn50": pnn50,
    })
    return features
