from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.spatial import cKDTree

from waveform_baselines.wf_features.ecg import _detect_r_peaks_by_finite_run, _rr_validity
from waveform_baselines.wf_features.pulsatile import _extract_pulsatile_beats
from waveform_baselines.wf_features.resp import _detect_resp_extrema_by_finite_run
from waveform_baselines.wf_features.utils import butter_filter, finite_runs, finite_values, interpolate_short_gaps, nan_iqr, resample_segment, safe_nanstat

from .config import DEFAULT_V8_EXTRACTION_CONFIG, V8ExtractionConfig
from .definitions import feature_names
from .morphology import advanced_abp_morphology_from_beats


def _nan() -> float:
    return float("nan")


def _cv(values: np.ndarray, min_count: int = 3) -> float:
    arr = finite_values(values)
    if arr.size < min_count:
        return _nan()
    mean = float(np.mean(arr))
    if abs(mean) < 1e-8:
        return _nan()
    return float(np.std(arr) / abs(mean))


def _v8_scale(values: np.ndarray) -> float:
    arr = finite_values(values)
    if arr.size < 2:
        return _nan()
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if scale > 1e-12:
        return float(scale)
    sd = float(np.std(arr))
    return sd if sd > 1e-12 else _nan()


def _outlier_count_positive(values: np.ndarray, k: float) -> float:
    arr = finite_values(values)
    if arr.size < 2:
        return _nan()
    scale = _v8_scale(arr)
    if not np.isfinite(scale):
        return 0.0
    return float(np.sum(arr > float(np.median(arr)) + k * scale))


def _validate_pulse_dict(pulses: dict[str, np.ndarray], required: tuple[str, ...] = ("valid", "run_id", "peak", "foot", "mean", "time_s", "start_idx", "end_idx", "width_s", "amplitude")) -> bool:
    if not pulses:
        return True
    lengths = []
    for key in required:
        if key not in pulses:
            return False
        lengths.append(len(np.asarray(pulses[key], dtype=object)))
    n = lengths[0] if lengths else 0
    if any(length != n for length in lengths):
        return False
    for key in ("raw_segments", "morphology_segments", "derivative_morphology_segments", "vpg_segments", "apg_segments", "jpg_segments"):
        if key in pulses and len(np.asarray(pulses[key], dtype=object)) != n:
            return False
    return True


def _validate_ecg_dict(ecg: dict[str, np.ndarray]) -> bool:
    peaks = np.asarray(ecg.get("peaks", []))
    run_ids = np.asarray(ecg.get("run_ids", []))
    rr = np.asarray(ecg.get("rr_s", []))
    rr_times = np.asarray(ecg.get("rr_times_s", []))
    valid_rr = np.asarray(ecg.get("valid_rr", []))
    same_run = np.asarray(ecg.get("same_run_rr", []))
    if peaks.size and peaks.size != run_ids.size:
        return False
    if not (rr.size == rr_times.size == valid_rr.size == same_run.size):
        return False
    if run_ids.size:
        return run_ids.size == rr.size + 1
    return peaks.size == rr.size + 1 or (peaks.size == 0 and rr.size == 0)


def _validate_resp_dict(resp: dict[str, np.ndarray], required: tuple[str, ...] = ("start_s", "peak_s", "end_s", "length_s", "amplitude", "run_id")) -> bool:
    if not resp:
        return True
    if any(key not in resp for key in required):
        return False
    lengths = [len(np.asarray(resp.get(key, []))) for key in required]
    return bool(lengths and all(length == lengths[0] for length in lengths))


def select_best_continuous_event_run(
    times_s: np.ndarray,
    valid_mask: np.ndarray,
    run_ids: np.ndarray,
    min_count: int = 1,
    min_duration_s: float = 0.0,
) -> np.ndarray:
    times = np.asarray(times_s, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    runs = np.asarray(run_ids, dtype=np.int64)
    if not (times.size == valid.size == runs.size):
        return np.asarray([], dtype=np.int64)
    valid = valid & np.isfinite(times) & (runs >= 0)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.asarray([], dtype=np.int64)
    split_at = np.where((np.diff(idx) != 1) | (runs[idx[1:]] != runs[idx[:-1]]))[0] + 1
    best = np.asarray([], dtype=np.int64)
    best_key = (-np.inf, -1, -np.inf)
    for chunk in np.split(idx, split_at):
        if chunk.size < min_count:
            continue
        duration = float(times[chunk[-1]] - times[chunk[0]]) if chunk.size >= 2 else 0.0
        if duration < min_duration_s:
            continue
        key = (duration, int(chunk.size), float(times[chunk[-1]]))
        if key > best_key:
            best = chunk.astype(np.int64, copy=False)
            best_key = key
    return best


def _corr(x: np.ndarray, y: np.ndarray, min_count: int = 12) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    xv = np.asarray(x[valid], dtype=np.float64)
    yv = np.asarray(y[valid], dtype=np.float64)
    if xv.size < min_count or np.std(xv) < 1e-8 or np.std(yv) < 1e-8:
        return _nan()
    return float(np.corrcoef(xv, yv)[0, 1])


def _linear_slope(times_s: np.ndarray, values: np.ndarray, min_count: int = 3) -> float:
    valid = np.isfinite(times_s) & np.isfinite(values)
    x = np.asarray(times_s[valid], dtype=np.float64)
    y = np.asarray(values[valid], dtype=np.float64)
    if x.size < min_count or np.ptp(x) < 1e-8:
        return _nan()
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return _nan()
    return float(np.dot(x, y - y.mean()) / denom)


def _relative_variation(values: np.ndarray) -> float:
    arr = finite_values(values)
    if arr.size < 2:
        return _nan()
    vmax = float(np.max(arr))
    vmin = float(np.min(arr))
    denom = (vmax + vmin) / 2.0
    if abs(denom) < 1e-8:
        return _nan()
    return float(100.0 * (vmax - vmin) / abs(denom))


def _pvi(values: np.ndarray) -> float:
    arr = finite_values(values)
    if arr.size < 2:
        return _nan()
    vmax = float(np.max(arr))
    vmin = float(np.min(arr))
    if abs(vmax) < 1e-8:
        return _nan()
    return float(100.0 * (vmax - vmin) / abs(vmax))


def _run_id_for_indices(indices: np.ndarray, finite_mask: np.ndarray) -> np.ndarray:
    out = np.full(np.asarray(indices).size, -1, dtype=np.int64)
    for run_id, (start, end) in enumerate(finite_runs(finite_mask)):
        idx = np.asarray(indices, dtype=np.int64)
        out[(idx >= start) & (idx < end)] = run_id
    return out


def _segment_run_id(start_idx: int, end_idx: int, finite_mask: np.ndarray) -> int:
    if end_idx <= start_idx:
        return -1
    for run_id, (start, end) in enumerate(finite_runs(finite_mask)):
        if int(start_idx) >= start and int(end_idx) < end:
            return int(run_id)
    return -1


def _sample_run_id_map(finite_mask: np.ndarray) -> np.ndarray:
    finite = np.asarray(finite_mask, dtype=bool)
    out = np.full(finite.size, -1, dtype=np.int64)
    for run_id, (start, end) in enumerate(finite_runs(finite)):
        out[start:end] = int(run_id)
    return out


def _run_id_for_segment_map(start_idx: int, end_idx: int, run_id_by_sample: np.ndarray) -> int:
    if end_idx <= start_idx or start_idx < 0 or end_idx >= run_id_by_sample.size:
        return -1
    run_id = int(run_id_by_sample[int(start_idx)])
    if run_id < 0 or int(run_id_by_sample[int(end_idx)]) != run_id:
        return -1
    return run_id


def _quantile_from_sorted(sorted_arr: np.ndarray, q: float) -> float:
    n = sorted_arr.size
    if n == 0:
        return _nan()
    pos = (n - 1) * float(q)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(sorted_arr[lo])
    frac = pos - lo
    return float((1.0 - frac) * sorted_arr[lo] + frac * sorted_arr[hi])


def _raw_observability_metadata(raw_signal: np.ndarray, run_id_by_sample: np.ndarray | None = None) -> dict[str, np.ndarray]:
    raw = np.asarray(raw_signal, dtype=np.float64)
    finite = np.isfinite(raw)
    if run_id_by_sample is None or np.asarray(run_id_by_sample).size != raw.size:
        run_map = _sample_run_id_map(finite)
    else:
        run_map = np.asarray(run_id_by_sample, dtype=np.int64)
    finite_values_for_sum = np.where(finite, raw, 0.0).astype(np.float64, copy=False)
    return {
        "_raw_run_id_by_sample": run_map,
        "_raw_bad_prefix": np.concatenate(([0], np.cumsum(~finite, dtype=np.int64))),
        "_raw_sum_prefix": np.concatenate(([0.0], np.cumsum(finite_values_for_sum, dtype=np.float64))),
        "_raw_sum_sq_prefix": np.concatenate(([0.0], np.cumsum(finite_values_for_sum * finite_values_for_sum, dtype=np.float64))),
    }


def _observability_from_pulse_dict(pulse: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    keys = ("_raw_run_id_by_sample", "_raw_bad_prefix", "_raw_sum_prefix", "_raw_sum_sq_prefix")
    if not pulse or any(key not in pulse for key in keys):
        return None
    return {key: np.asarray(pulse[key]) for key in keys}


def _interval_observed_from_metadata(start: int, end: int, raw_size: int, observability: dict[str, np.ndarray] | None) -> tuple[int, bool]:
    if observability is None or raw_size <= 0 or end <= start or start < 0 or end > raw_size:
        return -1, False
    run_map = np.asarray(observability.get("_raw_run_id_by_sample", []), dtype=np.int64)
    bad_prefix = np.asarray(observability.get("_raw_bad_prefix", []), dtype=np.int64)
    sum_prefix = np.asarray(observability.get("_raw_sum_prefix", []), dtype=np.float64)
    sum_sq_prefix = np.asarray(observability.get("_raw_sum_sq_prefix", []), dtype=np.float64)
    if run_map.size != raw_size or bad_prefix.size != raw_size + 1 or sum_prefix.size != raw_size + 1 or sum_sq_prefix.size != raw_size + 1:
        return -1, False
    run = int(run_map[start])
    if run < 0 or int(run_map[end - 1]) != run:
        return -1, False
    if int(bad_prefix[end] - bad_prefix[start]) != 0:
        return -1, False
    n = end - start
    sx = float(sum_prefix[end] - sum_prefix[start])
    sx2 = float(sum_sq_prefix[end] - sum_sq_prefix[start])
    mean = sx / n
    var = max(sx2 / n - mean * mean, 0.0)
    return run, bool(np.sqrt(var) > 1e-6)


def _skew_kurtosis_bias_corrected(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 3:
        return _nan(), _nan()
    centered = arr - float(np.mean(arr))
    m2 = float(np.mean(centered * centered))
    if m2 <= 1e-16:
        return _nan(), _nan()
    m3 = float(np.mean(centered ** 3))
    g1 = m3 / (m2 ** 1.5)
    skew = (np.sqrt(n * (n - 1.0)) / (n - 2.0)) * g1 if n > 2 else _nan()
    if n < 4:
        return float(skew), _nan()
    m4 = float(np.mean(centered ** 4))
    g2 = m4 / (m2 * m2) - 3.0
    kurt = ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0)
    return float(skew), float(kurt)


def _center_scale_segment(segment_values: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(segment_values, dtype=np.float64)
    if arr.size < 3 or not np.isfinite(arr).all():
        return None
    sorted_arr = np.sort(arr)
    median = _quantile_from_sorted(sorted_arr, 0.5)
    centered = arr - median
    scale = _quantile_from_sorted(sorted_arr, 0.95) - _quantile_from_sorted(sorted_arr, 0.05)
    if scale <= 1e-8:
        scale = float(np.std(centered))
    if scale <= 1e-8:
        return None
    return centered / scale


def _template_distances_from_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        matrix = np.empty((0, 0), dtype=np.float64)
    n_rows = matrix.shape[0]
    corrs = np.full(n_rows, np.nan, dtype=np.float64)
    dists = np.full(n_rows, np.nan, dtype=np.float64)
    if n_rows < 3:
        return corrs, dists
    sorted_vals = np.sort(matrix, axis=0)
    order = np.argsort(matrix, axis=0)
    ranks = np.empty_like(order)
    ranks[order, np.arange(matrix.shape[1])] = np.arange(n_rows)[:, None]

    def kth_excluding(k: int) -> np.ndarray:
        return np.where(ranks <= k, sorted_vals[k + 1], sorted_vals[k])

    loo_count = n_rows - 1
    if loo_count % 2:
        templates = kth_excluding(loo_count // 2)
    else:
        templates = 0.5 * (kth_excluding(loo_count // 2 - 1) + kth_excluding(loo_count // 2))
    row_centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    tmpl_centered = templates - np.mean(templates, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(row_centered * row_centered, axis=1) * np.sum(tmpl_centered * tmpl_centered, axis=1))
    valid_corr = denom > 1e-8
    corrs[valid_corr] = np.sum(row_centered[valid_corr] * tmpl_centered[valid_corr], axis=1) / denom[valid_corr]
    dists[:] = np.sqrt(np.mean((matrix - templates) ** 2, axis=1))
    return corrs, dists


def _template_distances(segments: list[np.ndarray], target_len: int, preserve_length: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    retained: list[int] = []
    for pos, seg in enumerate(segments):
        norm = _center_scale_segment(seg)
        if norm is None:
            continue
        resampled = resample_segment(norm, target_len)
        if np.isfinite(resampled).all():
            rows.append(resampled)
            retained.append(pos)
    out_len = len(segments) if preserve_length else len(rows)
    if len(rows) < 3:
        return np.full(out_len, np.nan, dtype=np.float64), np.full(out_len, np.nan, dtype=np.float64)
    corr_vals, dist_vals = _template_distances_from_matrix(np.stack(rows, axis=0))
    corrs = np.full(out_len, np.nan, dtype=np.float64)
    dists = np.full(out_len, np.nan, dtype=np.float64)
    out_idx = np.asarray(retained if preserve_length else list(range(len(rows))), dtype=np.int64)
    corrs[out_idx] = corr_vals
    dists[out_idx] = dist_vals
    return corrs, dists


def _precompute_pleth_shape_primitives(out: dict[str, np.ndarray], fs: int, config: V8ExtractionConfig) -> None:
    valid = np.asarray(out.get("valid", []), dtype=bool)
    n = valid.size
    if n == 0:
        return
    raw_segments = np.asarray(out.get("raw_segments", []), dtype=object)
    morph_segments = np.asarray(out.get("morphology_segments", raw_segments), dtype=object)
    local_peak = np.rint(np.asarray(out.get("peak_idx", []), dtype=np.float64) - np.asarray(out.get("start_idx", []), dtype=np.float64)).astype(int)
    width = np.asarray(out.get("width_s", np.full(n, np.nan)), dtype=np.float64)
    amp = np.asarray(out.get("amplitude", np.full(n, np.nan)), dtype=np.float64)
    for level in config.pleth_width_levels:
        out[f"pleth_width_{int(round(level * 100))}_s"] = np.full(n, np.nan, dtype=np.float64)
    stt = np.full(n, np.nan, dtype=np.float64)
    norm_area = np.full(n, np.nan, dtype=np.float64)
    skew_vals = np.full(n, np.nan, dtype=np.float64)
    kurt_vals = np.full(n, np.nan, dtype=np.float64)
    shape_matrix = np.full((n, config.morphology_template_points), np.nan, dtype=np.float64)
    for idx in np.flatnonzero(valid).tolist():
        if idx >= raw_segments.size:
            continue
        raw = np.asarray(raw_segments[idx], dtype=np.float64)
        morph = np.asarray(morph_segments[idx], dtype=np.float64) if idx < morph_segments.size else raw
        if raw.size < 5 or not np.isfinite(raw).all():
            continue
        peak_i = int(local_peak[idx]) if idx < local_peak.size else -1
        for level in config.pleth_width_levels:
            out[f"pleth_width_{int(round(level * 100))}_s"][idx] = _pulse_width_at_level(raw, peak_i, float(level), fs)
        a = float(amp[idx]) if idx < amp.size else np.nan
        w = float(width[idx]) if idx < width.size else np.nan
        if a > 1e-8 and w > 1e-8:
            area = float(np.trapezoid(np.maximum(raw - raw[0], 0.0), dx=1.0 / fs))
            norm_area[idx] = area / (a * w)
            norm_for_stats = _center_scale_segment(resample_segment((raw - raw[0]) / a, config.morphology_template_points))
            if norm_for_stats is not None:
                skew_vals[idx], kurt_vals[idx] = _skew_kurtosis_bias_corrected(norm_for_stats)
            norm_for_template = _center_scale_segment(raw)
            if norm_for_template is not None:
                resampled = resample_segment(norm_for_template, config.morphology_template_points)
                if np.isfinite(resampled).all():
                    shape_matrix[idx] = resampled
        if morph.size >= 3 and np.isfinite(morph).all() and a > 1e-8 and 0 < peak_i < morph.size:
            d = np.gradient(morph[: peak_i + 1]) * fs
            max_d = float(np.max(d))
            if max_d > 1e-8:
                stt[idx] = a / max_d
    out["pleth_slope_transit_time_s"] = stt
    out["pleth_normalized_area"] = norm_area
    out["pleth_pulse_skewness"] = skew_vals
    out["pleth_pulse_kurtosis"] = kurt_vals
    out["pleth_shape_matrix"] = shape_matrix


def _savgol_derivative_by_finite_run(x: np.ndarray, fs: int, config: V8ExtractionConfig, deriv: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    out = np.full(arr.size, np.nan, dtype=np.float64)
    poly = int(config.pleth_derivative_polynomial_order)
    for start, end in finite_runs(np.isfinite(arr)):
        seg = arr[start:end]
        if seg.size < max(poly + 3, 7) or not np.isfinite(seg).all():
            continue
        local_poly = min(poly, max(2, seg.size - 2))
        win = _odd_savgol_window(fs, config.pleth_derivative_smoothing_seconds, seg.size, local_poly)
        if win <= local_poly or win < 5:
            continue
        deriv_seg = signal.savgol_filter(
            seg,
            window_length=win,
            polyorder=local_poly,
            deriv=deriv,
            delta=1.0 / fs,
            mode="interp",
        )
        half = win // 2
        if deriv_seg.size <= 2 * half:
            continue
        deriv_seg[:half] = np.nan
        deriv_seg[-half:] = np.nan
        out[start:end] = deriv_seg
    return out


def _map_detector_peaks_to_morphology_aligned(detector_peaks: np.ndarray, morphology_signal: np.ndarray, search_radius: int, min_finite_fraction: float = 0.8) -> np.ndarray:
    n = morphology_signal.size
    peaks = np.asarray(detector_peaks, dtype=np.int64)
    mapped = np.full(peaks.size, -1, dtype=np.int64)
    polarities: list[float] = []
    local_infos: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray] | None] = []
    for peak in peaks:
        start = max(0, int(peak) - search_radius)
        end = min(n, int(peak) + search_radius + 1)
        local = morphology_signal[start:end]
        finite = np.isfinite(local)
        if local.size == 0 or np.mean(finite) < min_finite_fraction or not finite.any():
            local_infos.append(None)
            continue
        finite_offsets = np.flatnonzero(finite)
        local_finite = local[finite]
        centered = local_finite - np.median(local_finite)
        maxima = signal.argrelextrema(local_finite, np.greater_equal, order=1)[0]
        minima = signal.argrelextrema(local_finite, np.less_equal, order=1)[0]
        extrema = maxima.tolist() + minima.tolist()
        if not extrema:
            extrema = [int(np.argmax(np.abs(centered)))]
        target_offset = int(peak) - start
        best = min(extrema, key=lambda j: (-abs(float(centered[j])), abs(int(finite_offsets[j]) - target_offset)))
        polarity = float(np.sign(centered[best]))
        if polarity != 0.0:
            polarities.append(polarity)
        positive = maxima[centered[maxima] > 0]
        negative = minima[centered[minima] < 0]
        local_infos.append((start, target_offset, finite_offsets, positive, negative))
    polarity_arr = np.asarray(polarities, dtype=np.float64)
    dominant = 1.0 if polarity_arr.size == 0 or np.sum(polarity_arr >= 0) >= np.sum(polarity_arr < 0) else -1.0
    for pos, info in enumerate(local_infos):
        if info is None:
            continue
        start, target_offset, finite_offsets, positive, negative = info
        wanted = positive if dominant > 0 else negative
        if wanted.size:
            nearest = int(wanted[np.argmin(np.abs(finite_offsets[wanted] - target_offset))])
            mapped[pos] = start + int(finite_offsets[nearest])
    return mapped


def detect_ecg_events(
    ecg: np.ndarray,
    fs: int,
    config: V8ExtractionConfig,
    include_qrs_template: bool = True,
    include_morphology: bool = True,
) -> dict[str, np.ndarray]:
    max_gap = max(int(round(config.max_interpolated_gap_seconds * fs)), 0)
    xqrs_signal, xqrs_usable = interpolate_short_gaps(ecg, max_gap)
    xqrs_signal = np.asarray(xqrs_signal, dtype=np.float64)
    xqrs_signal[~xqrs_usable] = np.nan
    detector_signal = butter_filter(ecg, fs=fs, low_hz=5.0, high_hz=20.0, order=3, max_interp_gap_samples=max_gap)
    scale = _v8_scale(detector_signal)
    prominence = max(0.2 * scale if np.isfinite(scale) else 0.0, 1e-3)
    peaks, run_ids = _detect_r_peaks_by_finite_run(
        xqrs_signal,
        fs=fs,
        prominence=prominence,
        fallback_signal=detector_signal,
        detector=config.ecg_detector,
        allow_energy_fallback=config.ecg_allow_energy_fallback,
    )
    rr, same_run, valid_rr = _rr_validity(peaks, run_ids, fs=fs)
    rr_times = peaks[1:] / fs if peaks.size >= 2 else np.asarray([], dtype=np.float64)
    core = {
        "peaks": peaks.astype(np.int64),
        "detector_peak_indices": peaks.astype(np.int64),
        "run_ids": run_ids.astype(np.int64),
        "rr_s": rr.astype(np.float64),
        "rr_times_s": rr_times.astype(np.float64),
        "valid_rr": valid_rr.astype(bool),
        "same_run_rr": same_run.astype(bool),
    }
    if not include_morphology:
        return core
    morphology_signal = butter_filter(ecg, fs=fs, low_hz=0.5, high_hz=40.0, order=3, max_interp_gap_samples=max_gap)
    search_radius = max(int(round(0.08 * fs)), 1)
    mapped = _map_detector_peaks_to_morphology_aligned(peaks, morphology_signal, search_radius)
    qrs_left = int(round(0.12 * fs))
    qrs_right = int(round(0.18 * fs))
    morphology_run_map = _sample_run_id_map(np.isfinite(morphology_signal))
    morphology_run_ids = np.full(mapped.size, -1, dtype=np.int64)
    mapped_ok = (mapped >= 0) & (mapped < morphology_run_map.size)
    morphology_run_ids[mapped_ok] = morphology_run_map[mapped[mapped_ok]]
    qrs_segments: list[np.ndarray] = []
    qrs_valid: list[bool] = []
    r_amplitudes: list[float] = []
    for peak, run_id in zip(mapped.tolist(), morphology_run_ids.tolist()):
        if int(peak) < 0:
            qrs_segments.append(np.asarray([], dtype=np.float64))
            qrs_valid.append(False)
            r_amplitudes.append(np.nan)
            continue
        start = int(peak) - qrs_left
        end = int(peak) + qrs_right + 1
        ok = (
            run_id >= 0
            and start >= 0
            and end <= morphology_signal.size
            and _run_id_for_segment_map(start, end - 1, morphology_run_map) == run_id
        )
        if ok:
            seg = np.asarray(morphology_signal[start:end], dtype=np.float64)
            ok = bool(seg.size == qrs_left + qrs_right + 1 and np.isfinite(seg).all())
        else:
            seg = np.asarray([], dtype=np.float64)
        qrs_segments.append(seg)
        qrs_valid.append(ok)
        r_amplitudes.append(float(morphology_signal[int(peak)]) if ok else np.nan)
    qrs_corrs = _template_distances(qrs_segments, config.morphology_template_points)[0] if include_qrs_template else np.full(len(qrs_segments), np.nan, dtype=np.float64)
    core.update({
        "morphology_peak_indices": mapped.astype(np.int64),
        "morphology_peak_indices_for_detector": mapped.astype(np.int64),
        "morphology_run_ids": morphology_run_ids.astype(np.int64),
        "r_peak_amplitudes": np.asarray(r_amplitudes, dtype=np.float64),
        "qrs_segments": np.asarray(qrs_segments, dtype=object),
        "qrs_segment_valid": np.asarray(qrs_valid, dtype=bool),
        "qrs_template_correlations": qrs_corrs,
    })
    return core


def detect_pulses(
    x: np.ndarray,
    fs: int,
    kind: str,
    config: V8ExtractionConfig,
    include_pleth_shape_primitives: bool = False,
    include_segments: bool = True,
    include_observability: bool = True,
    include_raw_signal: bool = True,
) -> dict[str, np.ndarray]:
    max_gap = max(int(round(config.max_interpolated_gap_seconds * fs)), 0)
    if kind == "abp":
        detector = butter_filter(x, fs=fs, low_hz=0.5, high_hz=12.0, order=3, max_interp_gap_samples=max_gap)
        morphology = butter_filter(x, fs=fs, high_hz=20.0, order=3, max_interp_gap_samples=max_gap)
        min_distance = 60.0 / 220.0
    elif kind == "pleth":
        detector = butter_filter(x, fs=fs, low_hz=0.5, high_hz=8.0, order=3, max_interp_gap_samples=max_gap)
        morphology = butter_filter(x, fs=fs, high_hz=8.0, order=3, max_interp_gap_samples=max_gap)
        derivative_morphology = None
        vpg_full = None
        apg_full = None
        jpg_full = None
        if config.enable_pleth_derivative_fiducials:
            derivative_morphology = butter_filter(x, fs=fs, low_hz=0.5, high_hz=12.0, order=3, max_interp_gap_samples=max_gap)
            vpg_full = _savgol_derivative_by_finite_run(derivative_morphology, fs, config, deriv=1)
            apg_full = _savgol_derivative_by_finite_run(derivative_morphology, fs, config, deriv=2)
            jpg_full = _savgol_derivative_by_finite_run(derivative_morphology, fs, config, deriv=3)
        min_distance = 0.3
    else:
        raise ValueError(f"Unsupported pulse kind {kind!r}")
    _, beats = _extract_pulsatile_beats(
        x,
        detector,
        morphology,
        fs,
        kind,
        config.morphology_template_points,
        min_peak_distance_s=min_distance,
    )
    if not beats:
        return {}
    out = {key: value for key, value in beats.items() if not key.startswith("_")}
    if "_segments" in beats:
        out["_segments"] = beats["_segments"]
    if kind == "abp":
        sbp = out["peak"]
        dbp = out["foot"]
        pp = sbp - dbp
        valid = (
            (sbp >= 50.0)
            & (sbp <= 260.0)
            & (dbp >= 20.0)
            & (dbp <= 180.0)
            & (pp > 0)
            & (out["width_s"] >= 60.0 / 220.0 - 0.5 / fs)
            & (out["width_s"] <= 60.0 / 30.0 + 0.5 / fs)
        )
    else:
        valid = (out["amplitude"] > 0) & (out["width_s"] >= 0.3) & (out["width_s"] <= 2.0)
    out["valid"] = valid.astype(bool)
    out["time_s"] = out["peak_idx"] / fs
    out["foot_time_s"] = out["start_idx"] / fs
    if include_raw_signal:
        out["raw_signal"] = np.asarray(x, dtype=np.float64)
    start_idx_arr = np.rint(np.asarray(out["start_idx"], dtype=np.float64)).astype(np.int64)
    end_idx_arr = np.rint(np.asarray(out["end_idx"], dtype=np.float64)).astype(np.int64)
    if include_observability:
        raw_run_map = _sample_run_id_map(np.isfinite(x))
        out.update(_raw_observability_metadata(x, raw_run_map))
        run_ids = [_run_id_for_segment_map(int(start_i), int(end_i), raw_run_map) for start_i, end_i in zip(start_idx_arr.tolist(), end_idx_arr.tolist())]
    else:
        runs = np.asarray(finite_runs(np.isfinite(x)), dtype=np.int64)
        run_ids_arr = np.full(start_idx_arr.size, -1, dtype=np.int64)
        if runs.size:
            run_starts = runs[:, 0]
            run_ends = runs[:, 1]
            rid = np.searchsorted(run_starts, start_idx_arr, side="right") - 1
            ok = (rid >= 0) & (start_idx_arr >= 0) & (end_idx_arr >= start_idx_arr)
            if np.any(ok):
                ok_idx = np.flatnonzero(ok)
                ok[ok_idx] = end_idx_arr[ok_idx] < run_ends[rid[ok_idx]]
            run_ids_arr[ok] = rid[ok]
        run_ids = run_ids_arr.tolist()
    raw_segments: list[np.ndarray] = []
    morphology_segments: list[np.ndarray] = []
    for start_i, end_i in zip(start_idx_arr.tolist(), end_idx_arr.tolist()):
        if include_segments:
            raw_seg = np.asarray(x[start_i : end_i + 1], dtype=np.float64) if 0 <= start_i < end_i < x.size else np.asarray([], dtype=np.float64)
            morph_seg = np.asarray(morphology[start_i : end_i + 1], dtype=np.float64) if 0 <= start_i < end_i < morphology.size else np.asarray([], dtype=np.float64)
            raw_segments.append(raw_seg)
            morphology_segments.append(morph_seg)
    if include_segments:
        out["raw_segments"] = np.asarray(raw_segments, dtype=object)
        out["morphology_segments"] = np.asarray(morphology_segments, dtype=object)
    if kind == "pleth" and include_pleth_shape_primitives:
        if not include_segments:
            raise ValueError("include_pleth_shape_primitives requires include_segments=True")
        _precompute_pleth_shape_primitives(out, fs, config)
    if kind == "pleth" and config.enable_pleth_derivative_fiducials and include_segments:
        derivative_segments: list[np.ndarray] = []
        vpg_segments: list[np.ndarray] = []
        apg_segments: list[np.ndarray] = []
        jpg_segments: list[np.ndarray] = []
        assert derivative_morphology is not None and vpg_full is not None and apg_full is not None and jpg_full is not None
        for start_f, end_f in zip(out["start_idx"].tolist(), out["end_idx"].tolist()):
            start_i = int(round(start_f))
            end_i = int(round(end_f))
            derivative_segments.append(
                np.asarray(derivative_morphology[start_i : end_i + 1], dtype=np.float64)
                if 0 <= start_i < end_i < derivative_morphology.size
                else np.asarray([], dtype=np.float64)
            )
            vpg_segments.append(
                np.asarray(vpg_full[start_i : end_i + 1], dtype=np.float64)
                if 0 <= start_i < end_i < vpg_full.size
                else np.asarray([], dtype=np.float64)
            )
            apg_segments.append(
                np.asarray(apg_full[start_i : end_i + 1], dtype=np.float64)
                if 0 <= start_i < end_i < apg_full.size
                else np.asarray([], dtype=np.float64)
            )
            jpg_segments.append(
                np.asarray(jpg_full[start_i : end_i + 1], dtype=np.float64)
                if 0 <= start_i < end_i < jpg_full.size
                else np.asarray([], dtype=np.float64)
            )
        out["derivative_morphology_segments"] = np.asarray(derivative_segments, dtype=object)
        out["vpg_segments"] = np.asarray(vpg_segments, dtype=object)
        out["apg_segments"] = np.asarray(apg_segments, dtype=object)
        out["jpg_segments"] = np.asarray(jpg_segments, dtype=object)
    out["run_id"] = np.asarray(run_ids, dtype=np.int64)
    return out


def detect_resp_cycles(resp: np.ndarray, fs: int, config: V8ExtractionConfig) -> dict[str, np.ndarray]:
    max_gap = max(int(round(config.max_interpolated_gap_seconds * fs)), 0)
    filtered = butter_filter(resp, fs=fs, low_hz=0.05, high_hz=1.5, order=2, max_interp_gap_samples=max_gap)
    scale = _v8_scale(filtered)
    prominence = max(0.2 * scale if np.isfinite(scale) else 0.0, 1e-4)
    ordered = _detect_resp_extrema_by_finite_run(filtered, fs=fs, prominence=prominence, min_cycle_s=config.resp_min_cycle_seconds)
    rows = []
    for i in range(len(ordered) - 2):
        a_idx, a_sign, a_run = ordered[i]
        b_idx, b_sign, b_run = ordered[i + 1]
        c_idx, c_sign, c_run = ordered[i + 2]
        if a_run != b_run or a_run != c_run or not (a_sign < 0 and b_sign > 0 and c_sign < 0):
            continue
        segment = filtered[a_idx:c_idx]
        if segment.size < 2 or not np.isfinite(segment).all():
            continue
        length_s = (c_idx - a_idx) / fs
        if length_s < config.resp_min_cycle_seconds or length_s > config.resp_max_cycle_seconds:
            continue
        amp = abs(float(filtered[b_idx] - filtered[a_idx]))
        rows.append((a_idx / fs, b_idx / fs, c_idx / fs, length_s, amp, float(a_run)))
    if not rows:
        return {
            "start_s": np.asarray([], dtype=np.float64),
            "peak_s": np.asarray([], dtype=np.float64),
            "end_s": np.asarray([], dtype=np.float64),
            "length_s": np.asarray([], dtype=np.float64),
            "amplitude": np.asarray([], dtype=np.float64),
            "run_id": np.asarray([], dtype=np.int64),
            "filtered": filtered,
            "ordered_extrema": np.asarray([(idx / fs, sign, run) for idx, sign, run in ordered], dtype=np.float64),
        }
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "start_s": arr[:, 0],
        "peak_s": arr[:, 1],
        "end_s": arr[:, 2],
        "length_s": arr[:, 3],
        "amplitude": arr[:, 4],
        "run_id": arr[:, 5].astype(np.int64),
        "filtered": filtered,
        "ordered_extrema": np.asarray([(idx / fs, sign, run) for idx, sign, run in ordered], dtype=np.float64),
    }


def detect_resp_pause_durations(
    resp: np.ndarray,
    fs: int,
    config: V8ExtractionConfig,
    filtered: np.ndarray | None = None,
    ordered_extrema: np.ndarray | None = None,
) -> np.ndarray | None:
    arr = np.asarray(resp, dtype=np.float64)
    finite = np.isfinite(arr)
    if arr.size == 0 or np.mean(finite) < config.resp_pause_min_finite_fraction:
        return None
    finite_arr = arr[finite]
    if finite_arr.size < int(20 * fs) or np.std(finite_arr) < 1e-4:
        return None
    if filtered is None:
        max_gap = max(int(round(config.max_interpolated_gap_seconds * fs)), 0)
        filtered_arr = butter_filter(arr, fs=fs, low_hz=0.05, high_hz=1.5, order=2, max_interp_gap_samples=max_gap)
    else:
        filtered_arr = np.asarray(filtered, dtype=np.float64)
        if filtered_arr.size != arr.size:
            filtered_arr = np.asarray([], dtype=np.float64)
    if filtered_arr.size == 0:
        return None
    scale = _v8_scale(filtered_arr)
    if not np.isfinite(scale) or scale < 1e-4:
        return None
    if ordered_extrema is None:
        prominence = max(0.2 * scale, 1e-4)
        extrema = _detect_resp_extrema_by_finite_run(filtered_arr, fs=fs, prominence=prominence, min_cycle_s=config.resp_min_cycle_seconds)
    else:
        extrema_arr = np.asarray(ordered_extrema, dtype=np.float64)
        if extrema_arr.ndim != 2 or extrema_arr.shape[1] < 3:
            extrema = []
        else:
            extrema = [(int(round(t * fs)), int(sign), int(run)) for t, sign, run in extrema_arr[:, :3]]
    if len(extrema) < 4:
        return None
    pauses: list[float] = []
    for left, right in zip(extrema[:-1], extrema[1:]):
        left_idx, _, left_run = left
        right_idx, _, right_run = right
        if left_run != right_run:
            continue
        gap_s = (right_idx - left_idx) / fs
        if gap_s < config.resp_pause_min_seconds:
            continue
        context = int(round(config.resp_pause_context_seconds * fs))
        pre = arr[max(0, left_idx - context) : left_idx]
        middle = arr[left_idx:right_idx]
        post = arr[right_idx : min(arr.size, right_idx + context)]
        if not (np.isfinite(pre).all() and np.isfinite(middle).all() and np.isfinite(post).all()):
            continue
        pre_std = float(np.std(pre))
        post_std = float(np.std(post))
        if pre_std < 1e-4 or post_std < 1e-4:
            continue
        if np.std(middle) <= config.resp_pause_suppression_ratio * min(pre_std, post_std):
            pauses.append(float(gap_s))
    if not pauses:
        return np.asarray([], dtype=np.float64)
    return np.asarray(pauses, dtype=np.float64)


def detect_resp_pauses(resp: np.ndarray, fs: int, config: V8ExtractionConfig, filtered: np.ndarray | None = None, ordered_extrema: np.ndarray | None = None) -> tuple[float, float]:
    pauses = detect_resp_pause_durations(resp, fs, config, filtered=filtered, ordered_extrema=ordered_extrema)
    if pauses is None:
        return _nan(), _nan()
    if pauses.size == 0:
        return 0.0, 0.0
    return float(pauses.size), float(np.max(pauses))


def _resp_variation_features(prefix: str, pulses: dict[str, np.ndarray], resp: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    out: dict[str, float] = {}
    if not pulses or not resp:
        return out
    valid = np.asarray(pulses["valid"], dtype=bool)
    pulse_times = np.asarray(pulses["time_s"], dtype=np.float64)
    if valid.size != pulse_times.size:
        return out
    amplitude = np.asarray(pulses["amplitude"], dtype=np.float64)
    area = np.asarray(pulses["area"], dtype=np.float64)
    width = np.asarray(pulses["width_s"], dtype=np.float64)
    cycle_values: dict[str, list[float]] = {"amplitude": [], "area": [], "width_s": []}
    if prefix == "abp":
        peak = np.asarray(pulses["peak"], dtype=np.float64)
        foot = np.asarray(pulses["foot"], dtype=np.float64)
        mean = np.asarray(pulses["mean"], dtype=np.float64)
        cycle_values.update({"sbp": [], "dbp": [], "map": [], "pp": [], "spv": []})
    else:
        cycle_values["pvi"] = []
    for start, end in zip(resp["start_s"], resp["end_s"]):
        left = int(np.searchsorted(pulse_times, start, side="left"))
        right = int(np.searchsorted(pulse_times, end, side="left"))
        if right <= left:
            continue
        local = valid[left:right]
        if int(np.sum(local)) < config.min_pulses_per_resp_cycle:
            continue
        sl = slice(left, right)
        cycle_values["amplitude"].append(_relative_variation(amplitude[sl][local]))
        cycle_values["area"].append(_relative_variation(area[sl][local]))
        cycle_values["width_s"].append(_relative_variation(width[sl][local]))
        if prefix == "abp":
            sbp = peak[sl][local]
            dbp = foot[sl][local]
            mapv = mean[sl][local]
            pp = sbp - dbp
            cycle_values["sbp"].append(_relative_variation(sbp))
            cycle_values["dbp"].append(_relative_variation(dbp))
            cycle_values["map"].append(_relative_variation(mapv))
            cycle_values["pp"].append(_relative_variation(pp))
            cycle_values["spv"].append(float(np.max(sbp) - np.min(sbp)))
        else:
            cycle_values["pvi"].append(_pvi(amplitude[sl][local]))
    if len(cycle_values["amplitude"]) < config.min_resp_cycles_for_variation:
        return out
    if prefix == "abp":
        out["abp_ppv_pct"] = safe_nanstat(np.median, np.asarray(cycle_values["pp"]))
        out["abp_spv_mmhg"] = safe_nanstat(np.median, np.asarray(cycle_values["spv"]))
        out["abp_sbp_resp_variation_pct"] = safe_nanstat(np.median, np.asarray(cycle_values["sbp"]))
        out["abp_dbp_resp_variation_pct"] = safe_nanstat(np.median, np.asarray(cycle_values["dbp"]))
        out["abp_map_resp_variation_pct"] = safe_nanstat(np.median, np.asarray(cycle_values["map"]))
        out["abp_pulse_area_resp_variation_pct"] = safe_nanstat(np.median, np.asarray(cycle_values["area"]))
    else:
        out["pleth_resp_amplitude_variation_pct"] = safe_nanstat(np.median, np.asarray(cycle_values["pvi"]))
        out["pleth_area_resp_variation_pct"] = safe_nanstat(np.median, np.asarray(cycle_values["area"]))
        out["pleth_width_resp_variation_pct"] = safe_nanstat(np.median, np.asarray(cycle_values["width_s"]))
    return out


def pair_forward(
    source_times: np.ndarray,
    target_times: np.ndarray,
    bounds_ms: tuple[float, float],
    source_valid: np.ndarray | None = None,
    target_valid: np.ndarray | None = None,
) -> np.ndarray:
    pairs = _observable_forward_pairs(source_times, target_times, bounds_ms, source_valid, target_valid)
    return pairs[:, 2] if pairs.size else np.asarray([], dtype=np.float64)


def _observable_forward_pairs(
    source_times: np.ndarray,
    target_times: np.ndarray,
    bounds_ms: tuple[float, float],
    source_valid: np.ndarray | None = None,
    target_valid: np.ndarray | None = None,
    target_raw_signal: np.ndarray | None = None,
    target_run_ids: np.ndarray | None = None,
    fs: int | None = None,
    target_observability: dict[str, np.ndarray] | None = None,
    return_source_observable: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    if source_valid is None:
        source_valid = np.isfinite(source_times)
    if target_valid is None:
        target_valid = np.isfinite(target_times)
    source_valid = np.asarray(source_valid, dtype=bool) & np.isfinite(source_times)
    target_valid = np.asarray(target_valid, dtype=bool) & np.isfinite(target_times)
    empty_pairs = np.empty((0, 3), dtype=np.float64)
    empty_source_observable = np.zeros(source_times.size, dtype=bool)
    if source_valid.size != source_times.size or target_valid.size != target_times.size:
        return (empty_pairs, empty_source_observable) if return_source_observable else empty_pairs
    raw = np.asarray(target_raw_signal, dtype=np.float64) if target_raw_signal is not None else None
    target_runs = np.asarray(target_run_ids, dtype=np.int64) if target_run_ids is not None else np.full(target_times.size, -1, dtype=np.int64)
    if target_run_ids is not None and target_runs.size != target_times.size:
        return (empty_pairs, empty_source_observable) if return_source_observable else empty_pairs
    if raw is not None and fs is not None and target_observability is None:
        target_observability = _raw_observability_metadata(raw)
    lo, hi = bounds_ms[0] / 1000.0, bounds_ms[1] / 1000.0
    target_pos = 0
    rows: list[tuple[float, float, float]] = []
    source_observable = np.zeros(source_times.size, dtype=bool)
    for src_idx, src in enumerate(source_times.tolist()):
        if not source_valid[src_idx]:
            continue
        interval_run = None
        observed = True
        if raw is not None and fs is not None:
            start = max(0, int(np.floor((src + lo) * fs)))
            end = min(raw.size, int(np.ceil((src + hi) * fs)))
            interval_run, observed = _interval_observed_from_metadata(start, end, raw.size, target_observability)
            if not observed:
                continue
        source_observable[src_idx] = True
        target_pos = max(target_pos, int(np.searchsorted(target_times, src + lo, side="left")))
        scan = target_pos
        while scan < target_times.size and target_times[scan] <= src + hi:
            run_ok = True
            if interval_run is not None:
                run_ok = bool(target_runs[scan] == interval_run)
            if target_valid[scan] and run_ok:
                rows.append((float(src_idx), float(scan), float((target_times[scan] - src) * 1000.0)))
                target_pos = scan + 1
                break
            scan += 1
    pairs = np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 3), dtype=np.float64)
    return (pairs, source_observable) if return_source_observable else pairs


def build_cross_signal_pair_cache(ecg: dict[str, np.ndarray], abp: dict[str, np.ndarray], pleth: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, np.ndarray]:
    ecg_peaks = np.asarray(ecg.get("peaks", []))
    ecg_runs = np.asarray(ecg.get("run_ids", []))
    if ecg_peaks.size != ecg_runs.size:
        return {}
    ecg_times = ecg_peaks.astype(np.float64) / config.sampling_rate_hz
    ecg_valid = ecg_runs.astype(np.int64) >= 0
    out: dict[str, np.ndarray] = {}
    if abp and _validate_pulse_dict(abp, required=("valid", "run_id", "foot_time_s")):
        abp_times = np.asarray(abp.get("foot_time_s", np.asarray([])), dtype=np.float64)
        abp_valid = np.asarray(abp.get("valid", np.asarray([], dtype=bool)), dtype=bool) & (np.asarray(abp.get("run_id", np.full(abp_times.size, -1)), dtype=np.int64) >= 0)
        pairs, source_observable = _observable_forward_pairs(
            ecg_times,
            abp_times,
            config.ecg_abp_pat_bounds_ms,
            ecg_valid,
            abp_valid,
            np.asarray(abp.get("raw_signal", []), dtype=np.float64) if "raw_signal" in abp else None,
            np.asarray(abp.get("run_id", []), dtype=np.int64),
            config.sampling_rate_hz,
            _observability_from_pulse_dict(abp),
            return_source_observable=True,
        )
        out["ecg_abp"] = pairs
        out["ecg_abp_source_observable"] = source_observable
    if pleth and _validate_pulse_dict(pleth, required=("valid", "run_id", "foot_time_s")):
        pleth_times = np.asarray(pleth.get("foot_time_s", np.asarray([])), dtype=np.float64)
        pleth_valid = np.asarray(pleth.get("valid", np.asarray([], dtype=bool)), dtype=bool) & (np.asarray(pleth.get("run_id", np.full(pleth_times.size, -1)), dtype=np.int64) >= 0)
        pairs, source_observable = _observable_forward_pairs(
            ecg_times,
            pleth_times,
            config.ecg_pleth_pat_bounds_ms,
            ecg_valid,
            pleth_valid,
            np.asarray(pleth.get("raw_signal", []), dtype=np.float64) if "raw_signal" in pleth else None,
            np.asarray(pleth.get("run_id", []), dtype=np.int64),
            config.sampling_rate_hz,
            _observability_from_pulse_dict(pleth),
            return_source_observable=True,
        )
        out["ecg_pleth"] = pairs
        out["ecg_pleth_source_observable"] = source_observable
    if abp and pleth and _validate_pulse_dict(abp, required=("valid", "run_id", "foot_time_s")) and _validate_pulse_dict(pleth, required=("valid", "run_id", "foot_time_s")):
        abp_times = np.asarray(abp.get("foot_time_s", np.asarray([])), dtype=np.float64)
        abp_valid = np.asarray(abp.get("valid", np.asarray([], dtype=bool)), dtype=bool) & (np.asarray(abp.get("run_id", np.full(abp_times.size, -1)), dtype=np.int64) >= 0)
        pleth_times = np.asarray(pleth.get("foot_time_s", np.asarray([])), dtype=np.float64)
        pleth_valid = np.asarray(pleth.get("valid", np.asarray([], dtype=bool)), dtype=bool) & (np.asarray(pleth.get("run_id", np.full(pleth_times.size, -1)), dtype=np.int64) >= 0)
        out["abp_pleth"] = _observable_forward_pairs(
            abp_times,
            pleth_times,
            config.abp_pleth_delay_bounds_ms,
            abp_valid,
            pleth_valid,
            np.asarray(pleth.get("raw_signal", []), dtype=np.float64) if "raw_signal" in pleth else None,
            np.asarray(pleth.get("run_id", []), dtype=np.int64),
            config.sampling_rate_hz,
            _observability_from_pulse_dict(pleth),
        )
    return out


def timing_features(ecg: dict[str, np.ndarray], abp: dict[str, np.ndarray], pleth: dict[str, np.ndarray], config: V8ExtractionConfig, pair_cache: dict[str, np.ndarray] | None = None) -> dict[str, float]:
    ecg_peaks = np.asarray(ecg.get("peaks", []))
    ecg_runs = np.asarray(ecg.get("run_ids", []))
    if ecg_peaks.size != ecg_runs.size or (abp and not _validate_pulse_dict(abp, required=("valid", "run_id", "foot_time_s"))) or (pleth and not _validate_pulse_dict(pleth, required=("valid", "run_id", "foot_time_s"))):
        return {}
    out: dict[str, float] = {}
    ecg_times = ecg_peaks.astype(np.float64) / config.sampling_rate_hz
    ecg_valid = ecg_runs.astype(np.int64) >= 0
    abp_times = np.asarray(abp.get("foot_time_s", np.asarray([])), dtype=np.float64) if abp else np.asarray([])
    abp_valid = (np.asarray(abp.get("valid", np.asarray([], dtype=bool)), dtype=bool) & (np.asarray(abp.get("run_id", np.full(abp_times.size, -1)), dtype=np.int64) >= 0)) if abp else np.asarray([], dtype=bool)
    pleth_times = np.asarray(pleth.get("foot_time_s", np.asarray([])), dtype=np.float64) if pleth else np.asarray([])
    pleth_valid = (np.asarray(pleth.get("valid", np.asarray([], dtype=bool)), dtype=bool) & (np.asarray(pleth.get("run_id", np.full(pleth_times.size, -1)), dtype=np.int64) >= 0)) if pleth else np.asarray([], dtype=bool)
    if pair_cache is None:
        pair_cache = build_cross_signal_pair_cache(ecg, abp, pleth, config)
    pairs = {
        "ecg_abp_pat": pair_cache.get("ecg_abp", np.empty((0, 3), dtype=np.float64))[:, 2],
        "ecg_pleth_pat": pair_cache.get("ecg_pleth", np.empty((0, 3), dtype=np.float64))[:, 2],
        "abp_pleth_delay": pair_cache.get("abp_pleth", np.empty((0, 3), dtype=np.float64))[:, 2],
    }
    for name, delays in pairs.items():
        if finite_values(delays).size >= config.min_timing_pairs:
            out[f"{name}_median_ms"] = safe_nanstat(np.median, delays)
            out[f"{name}_iqr_ms"] = nan_iqr(delays)
            out[f"{name}_sd_ms"] = safe_nanstat(np.std, delays)
    return out


def _paired_sbp_rr_for_lag(ecg: dict[str, np.ndarray], abp: dict[str, np.ndarray], lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    abp_valid = np.asarray(abp.get("valid", []), dtype=bool)
    sbp_all = np.asarray(abp.get("peak", []), dtype=np.float64)
    sbp_times_all = np.asarray(abp.get("time_s", []), dtype=np.float64)
    abp_run = np.asarray(abp.get("run_id", []), dtype=np.int64)
    rr = np.asarray(ecg.get("rr_s", []), dtype=np.float64)
    rr_times = np.asarray(ecg.get("rr_times_s", []), dtype=np.float64)
    rr_valid = np.asarray(ecg.get("valid_rr", []), dtype=bool)
    same_run = np.asarray(ecg.get("same_run_rr", rr_valid), dtype=bool)
    ecg_run = np.asarray(ecg.get("run_ids", []), dtype=np.int64)
    if abp_run.size != sbp_all.size or ecg_run.size < rr.size + 1:
        n_trans = max(sbp_all.size - 1, 0)
        return (
            np.full(sbp_all.size, np.nan, dtype=np.float64),
            np.full(sbp_all.size, np.nan, dtype=np.float64),
            np.zeros(sbp_all.size, dtype=bool),
            np.zeros(n_trans, dtype=bool),
        )
    rr_run = ecg_run[1:]
    sbp = np.full(sbp_all.size, np.nan, dtype=np.float64)
    paired_rr = np.full(sbp_all.size, np.nan, dtype=np.float64)
    transition_valid = np.zeros(max(sbp_all.size - 1, 0), dtype=bool)
    obs_valid = np.zeros(sbp_all.size, dtype=bool)
    if sbp_all.size == 0 or rr.size == 0:
        return sbp, paired_rr, obs_valid, transition_valid

    rr_idx0 = np.searchsorted(rr_times, sbp_times_all, side="left").astype(np.int64)
    rr_idx = rr_idx0 + int(lag)
    ok = abp_valid & (abp_run >= 0) & (rr_idx >= 0) & (rr_idx < rr.size) & (rr_idx0 >= 0) & (rr_idx0 < rr.size)
    if np.any(ok):
        valid_positions = np.flatnonzero(ok)
        target_rr_idx = rr_idx[valid_positions]
        ok_valid = rr_valid[target_rr_idx] & same_run[target_rr_idx] & (rr_run[target_rr_idx] >= 0)
        valid_positions = valid_positions[ok_valid]
        for offset in range(abs(int(lag)) + 1):
            if valid_positions.size == 0:
                break
            step_idx = rr_idx0[valid_positions] + offset if lag >= 0 else rr_idx[valid_positions] + offset
            in_bounds = (step_idx >= 0) & (step_idx < rr.size)
            valid_positions = valid_positions[in_bounds]
            step_idx = step_idx[in_bounds]
            target_rr_idx = rr_idx[valid_positions]
            step_ok = rr_valid[step_idx] & same_run[step_idx] & (rr_run[step_idx] == rr_run[target_rr_idx])
            valid_positions = valid_positions[step_ok]
        if valid_positions.size:
            obs_valid[valid_positions] = True
            sbp[valid_positions] = sbp_all[valid_positions]
            paired_rr[valid_positions] = rr[rr_idx[valid_positions]] * 1000.0

    if transition_valid.size:
        transition_valid[:] = (
            obs_valid[:-1]
            & obs_valid[1:]
            & (abp_run[:-1] == abp_run[1:])
            & (abp_run[:-1] >= 0)
        )
        for idx in np.flatnonzero(transition_valid):
            left_rr = int(rr_idx[idx])
            right_rr = int(rr_idx[idx + 1])
            lo = min(left_rr, right_rr)
            hi = max(left_rr, right_rr)
            if lo < 0 or hi >= rr.size or not np.all(rr_valid[lo : hi + 1] & same_run[lo : hi + 1]) or not np.all(rr_run[lo : hi + 1] == rr_run[left_rr]):
                transition_valid[idx] = False
    return sbp, paired_rr, obs_valid, transition_valid


def baroreflex_features(ecg: dict[str, np.ndarray], abp: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    if not abp or not _validate_ecg_dict(ecg) or not _validate_pulse_dict(abp, required=("valid", "run_id", "time_s", "peak")):
        return {}
    abp_times = np.asarray(abp.get("time_s", []), dtype=np.float64)
    abp_valid_all = np.asarray(abp.get("valid", []), dtype=bool)
    abp_run_all = np.asarray(abp.get("run_id", []), dtype=np.int64)
    eligible_idx = select_best_continuous_event_run(
        abp_times,
        abp_valid_all & np.isfinite(np.asarray(abp.get("peak", []), dtype=np.float64)),
        abp_run_all,
        min_count=3,
        min_duration_s=config.min_baroreflex_coverage_seconds,
    )
    if eligible_idx.size == 0:
        return {}
    abp = {key: (np.asarray(value)[eligible_idx] if key in {"valid", "run_id", "time_s", "peak"} and len(np.asarray(value)) == abp_times.size else value) for key, value in abp.items()}
    lag_cache = {lag: _paired_sbp_rr_for_lag(ecg, abp, lag) for lag in (0, 1, 2)}
    sbp0, rr0, obs0, _ = lag_cache[0]
    base = _corr(sbp0[obs0], rr0[obs0], config.min_baroreflex_pairs)
    best_abs_corr = _nan()
    best_lag = _nan()
    for lag, (sbp_lag, rr_lag, obs_lag, _) in lag_cache.items():
        corr = _corr(sbp_lag[obs_lag], rr_lag[obs_lag], config.min_baroreflex_pairs)
        if np.isfinite(corr) and (not np.isfinite(best_abs_corr) or abs(corr) > best_abs_corr):
            best_abs_corr = abs(corr)
            best_lag = float(lag)

    min_len = 3
    min_sbp_step = 1.0
    min_rr_step_ms = 4.0
    candidate_count = 0
    sequence_slopes: list[float] = []
    sbp0_all = np.asarray(abp.get("peak", []), dtype=np.float64)
    abp_valid = np.asarray(abp.get("valid", []), dtype=bool)
    abp_run = np.asarray(abp.get("run_id", []), dtype=np.int64)
    base_transition = abp_valid[:-1] & abp_valid[1:] & (abp_run[:-1] == abp_run[1:]) & (abp_run[:-1] >= 0)
    ds0 = np.diff(sbp0_all)
    for direction in (1.0, -1.0):
        monotone = base_transition & (direction * ds0 >= min_sbp_step)
        pos = 0
        while pos < monotone.size:
            if not monotone[pos]:
                pos += 1
                continue
            run_start = pos
            while pos < monotone.size and monotone[pos]:
                pos += 1
            run_end = pos + 1
            if run_end - run_start >= min_len:
                pairable = []
                concordant = []
                for lag, (sbp, rr_lagged, _, transition_valid) in lag_cache.items():
                    if transition_valid.size == 0 or run_end - 1 > transition_valid.size:
                        continue
                    x = sbp[run_start:run_end]
                    y = rr_lagged[run_start:run_end]
                    fully_pairable = bool(
                        np.all(transition_valid[run_start : run_end - 1])
                        and np.isfinite(x).all()
                        and np.isfinite(y).all()
                    )
                    if not fully_pairable:
                        continue
                    pairable.append(lag)
                    rr_steps = np.diff(y)
                    if np.all(direction * rr_steps >= min_rr_step_ms):
                        seq_slope = _linear_slope(x, y, min_len)
                        if np.isfinite(seq_slope) and seq_slope > 0:
                            corr = float(np.corrcoef(x, y)[0, 1]) if x.size >= 3 and np.std(x) > 1e-8 and np.std(y) > 1e-8 else 0.0
                            concordant.append((abs(corr), seq_slope))
                if pairable:
                    candidate_count += 1
                    if concordant:
                        concordant.sort(reverse=True)
                        sequence_slopes.append(float(concordant[0][1]))
            pos += 1
    accepted = len(sequence_slopes)
    return {
        "sbp_rr_corr_5m": base,
        "sbp_rr_max_abs_lag_corr_5m": best_abs_corr,
        "sbp_rr_optimal_lag_beats_5m": best_lag,
        "baroreflex_gain_5m_ms_per_mmhg": safe_nanstat(np.median, np.asarray(sequence_slopes)) if accepted >= 3 else _nan(),
        "baroreflex_sequence_count_5m": float(accepted),
        "baroreflex_sequence_fraction_5m": float(accepted / candidate_count) if candidate_count > 0 else _nan(),
    }


def sample_entropy(rr: np.ndarray, m: int = 2, r_fraction: float = 0.2, min_count: int = 30) -> float:
    x = finite_values(rr)
    if x.size < min_count or np.std(x) < 1e-8:
        return _nan()
    r = r_fraction * float(np.std(x))

    def count(dim: int) -> int:
        windows = np.lib.stride_tricks.sliding_window_view(x, dim)
        if windows.shape[0] < 2:
            return 0
        tree = cKDTree(windows)
        pairs = tree.query_pairs(r, p=np.inf, output_type="ndarray")
        return int(pairs.shape[0])

    a = count(m + 1)
    b = count(m)
    if a <= 0 or b <= 0:
        return _nan()
    return float(-np.log(a / b))


def poincare(rr: np.ndarray, min_pairs: int = 20) -> tuple[float, float, float]:
    x = finite_values(rr)
    if x.size < min_pairs + 1:
        return _nan(), _nan(), _nan()
    diff = np.diff(x)
    sd1 = np.sqrt(0.5) * float(np.std(diff))
    sdnn = float(np.std(x))
    sd2_sq = max(2.0 * sdnn * sdnn - 0.5 * float(np.std(diff)) ** 2, 0.0)
    sd2 = float(np.sqrt(sd2_sq))
    ratio = sd1 / sd2 if sd2 > 1e-8 else _nan()
    return sd1, sd2, ratio


def dfa_alpha1(rr: np.ndarray, min_count: int = 40) -> float:
    x = finite_values(rr)
    if x.size < min_count or np.std(x) < 1e-8:
        return _nan()
    y = np.cumsum(x - np.mean(x))
    scales = np.asarray([4, 6, 8, 10, 12, 16], dtype=int)
    fluct = []
    used = []
    for scale in scales:
        n_seg = y.size // scale
        if n_seg < 2:
            continue
        vals = []
        for seg_idx in range(n_seg):
            seg = y[seg_idx * scale : (seg_idx + 1) * scale]
            xs = np.arange(scale, dtype=np.float64)
            coef = np.polyfit(xs, seg, 1)
            detrended = seg - np.polyval(coef, xs)
            vals.append(float(np.sqrt(np.mean(detrended * detrended))))
        f = float(np.sqrt(np.mean(np.square(vals))))
        if f > 0:
            fluct.append(f)
            used.append(float(scale))
    if len(fluct) < 3:
        return _nan()
    return float(np.polyfit(np.log(used), np.log(fluct), 1)[0])


def lomb_hrv(rr_times: np.ndarray, rr: np.ndarray, min_count: int = 30, min_coverage_s: float = 240.0) -> dict[str, float]:
    valid = np.isfinite(rr_times) & np.isfinite(rr)
    t = np.asarray(rr_times[valid], dtype=np.float64)
    x = np.asarray(rr[valid], dtype=np.float64)
    if x.size < min_count or np.ptp(t) < min_coverage_s or np.std(x) < 1e-8:
        return {}
    t = t - t[0]
    x = x - np.mean(x)
    freqs = np.linspace(0.0033, 0.40, 512)
    raw = signal.lombscargle(t, x, 2.0 * np.pi * freqs, normalize=True)
    raw_area = float(np.trapezoid(raw, freqs))
    if raw_area <= 1e-12:
        return {}
    pxx = raw * (float(np.var(x)) / raw_area)
    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.trapezoid(pxx[mask], freqs[mask])) if np.any(mask) else _nan()
    total = band(0.0033, 0.40)
    lf = band(0.04, 0.15)
    hf = band(0.15, 0.40)
    return {
        "hrv_total_power_5m": total,
        "hrv_lf_power_5m": lf,
        "hrv_hf_power_5m": hf,
        "hrv_lf_hf_ratio_5m": lf / hf if np.isfinite(lf) and np.isfinite(hf) and hf > 1e-12 else _nan(),
    }


def _longest_valid_rr_run(
    ecg: dict[str, np.ndarray],
    start_s: float,
    end_s: float,
    min_count: int = 1,
    min_duration_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    rr = np.asarray(ecg.get("rr_s", []), dtype=np.float64)
    rr_times = np.asarray(ecg.get("rr_times_s", []), dtype=np.float64)
    valid_rr = np.asarray(ecg.get("valid_rr", []), dtype=bool)
    same_run_rr = np.asarray(ecg.get("same_run_rr", []), dtype=bool)
    peak_runs = np.asarray(ecg.get("run_ids", []), dtype=np.int64)
    if not (rr.size == rr_times.size == valid_rr.size == same_run_rr.size) or peak_runs.size < rr.size + 1:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    rr_run = peak_runs[1:]
    valid = valid_rr & same_run_rr & (rr_run >= 0) & (rr_times > start_s) & (rr_times <= end_s)
    if rr.size >= 2:
        continuous = np.ones(rr.size, dtype=bool)
        continuous[1:] = same_run_rr[1:] & (rr_run[1:] == rr_run[:-1])
        valid = valid & continuous
    idx = select_best_continuous_event_run(rr_times, valid, rr_run, min_count=min_count, min_duration_s=min_duration_s)
    return rr_times[idx], rr[idx]


def nonlinear_hrv_features(ecg: dict[str, np.ndarray], start_s: float, end_s: float, config: V8ExtractionConfig) -> dict[str, float]:
    if not _validate_ecg_dict(ecg):
        return {}
    if end_s - start_s + 1e-9 < config.rolling_history_seconds:
        return {}
    rr_times, rr = _longest_valid_rr_run(
        ecg,
        start_s,
        end_s,
        min_count=config.min_hrv_rr_5m,
        min_duration_s=config.min_hrv_coverage_seconds,
    )
    if rr.size < config.min_hrv_rr_5m or _event_span_seconds(rr_times, np.arange(rr_times.size, dtype=np.int64)) < config.min_hrv_coverage_seconds:
        return {}
    sd1, sd2, ratio = poincare(rr)
    out = {
        "hrv_sampen_5m": sample_entropy(rr, m=2, r_fraction=0.2, min_count=config.min_hrv_rr_5m),
        "hrv_poincare_sd1_5m": sd1,
        "hrv_poincare_sd2_5m": sd2,
        "hrv_poincare_sd1_sd2_ratio_5m": ratio,
        "hrv_dfa_alpha1_5m": dfa_alpha1(rr, min_count=config.min_dfa_rr_5m),
    }
    out.update(lomb_hrv(rr_times, rr, min_count=config.min_hrv_rr_5m, min_coverage_s=config.min_hrv_coverage_seconds))
    return out


def burden_instability_features(ecg: dict[str, np.ndarray], abp: dict[str, np.ndarray], pleth: dict[str, np.ndarray], resp: dict[str, np.ndarray], resp_signal: np.ndarray, config: V8ExtractionConfig, minute_start_s: float, minute_end_s: float) -> dict[str, float]:
    out: dict[str, float] = {}
    if abp:
        valid = abp["valid"]
        t = abp["time_s"]
        sbp = abp["peak"]
        dbp = abp["foot"]
        mapv = abp["mean"]
        in_min = valid & (t >= minute_start_s) & (t < minute_end_s)
        out["abp_map_min"] = safe_nanstat(np.min, mapv[in_min])
        out["abp_sbp_min"] = safe_nanstat(np.min, sbp[in_min])
        if np.sum(in_min) > 0:
            out["abp_map_below_70_beat_fraction"] = float(np.mean(mapv[in_min] < 70.0))
            out["abp_map_below_75_beat_fraction"] = float(np.mean(mapv[in_min] < 75.0))
            out["abp_sbp_below_95_beat_fraction"] = float(np.mean(sbp[in_min] < 95.0))
            out["abp_sbp_below_100_beat_fraction"] = float(np.mean(sbp[in_min] < 100.0))
        out["map_slope_30s"] = _linear_slope(t[in_min & (t >= minute_end_s - 30.0)], mapv[in_min & (t >= minute_end_s - 30.0)])
        out["sbp_slope_30s"] = _linear_slope(t[in_min & (t >= minute_end_s - 30.0)], sbp[in_min & (t >= minute_end_s - 30.0)])
        out["map_last_10s"] = safe_nanstat(np.median, mapv[in_min & (t >= minute_end_s - 10.0)])
        out["map_last_30s"] = safe_nanstat(np.median, mapv[in_min & (t >= minute_end_s - 30.0)])
        out["sbp_last_30s"] = safe_nanstat(np.median, sbp[in_min & (t >= minute_end_s - 30.0)])
        map_s1 = _linear_slope(t[in_min & (t < minute_end_s - 30.0)], mapv[in_min & (t < minute_end_s - 30.0)])
        sbp_s1 = _linear_slope(t[in_min & (t < minute_end_s - 30.0)], sbp[in_min & (t < minute_end_s - 30.0)])
        out["map_acceleration"] = (out["map_slope_30s"] - map_s1) / 30.0 if np.isfinite(out["map_slope_30s"]) and np.isfinite(map_s1) else _nan()
        out["sbp_acceleration"] = (out["sbp_slope_30s"] - sbp_s1) / 30.0 if np.isfinite(out["sbp_slope_30s"]) and np.isfinite(sbp_s1) else _nan()
    rr_valid = ecg["valid_rr"]
    rr_t = ecg["rr_times_s"]
    rr = ecg["rr_s"]
    hr = 60.0 / rr
    in_min_rr = rr_valid & (rr_t >= minute_start_s) & (rr_t < minute_end_s)
    out["ecg_hr_max"] = safe_nanstat(np.max, hr[in_min_rr])
    out["ecg_hr_p95"] = safe_nanstat(lambda x: np.percentile(x, 95.0), hr[in_min_rr])
    if np.sum(in_min_rr) > 0:
        out["ecg_hr_above_100_rr_fraction"] = float(np.mean(hr[in_min_rr] > 100.0))
        out["ecg_hr_above_120_rr_fraction"] = float(np.mean(hr[in_min_rr] > 120.0))
    out["hr_slope_30s"] = _linear_slope(rr_t[in_min_rr & (rr_t >= minute_end_s - 30.0)], hr[in_min_rr & (rr_t >= minute_end_s - 30.0)])
    out["hr_last_30s"] = safe_nanstat(np.median, hr[in_min_rr & (rr_t >= minute_end_s - 30.0)])
    hr_s1 = _linear_slope(rr_t[in_min_rr & (rr_t < minute_end_s - 30.0)], hr[in_min_rr & (rr_t < minute_end_s - 30.0)])
    out["hr_acceleration"] = (out["hr_slope_30s"] - hr_s1) / 30.0 if np.isfinite(out["hr_slope_30s"]) and np.isfinite(hr_s1) else _nan()
    if np.sum(in_min_rr) >= 3:
        h = hr[in_min_rr]
        tt = rr_t[in_min_rr]
        accel = np.abs(np.diff(h) / np.maximum(np.diff(tt), 1e-8))
        out["ecg_hr_abs_acceleration_p95"] = safe_nanstat(lambda x: np.percentile(x, 95.0), accel)
    if pleth:
        pv = pleth["valid"] & (pleth["time_s"] >= minute_start_s) & (pleth["time_s"] < minute_end_s)
        out["pleth_amp_slope_30s"] = _linear_slope(pleth["time_s"][pv & (pleth["time_s"] >= minute_end_s - 30.0)], pleth["amplitude"][pv & (pleth["time_s"] >= minute_end_s - 30.0)])
    if resp:
        rv = (resp["end_s"] >= minute_start_s) & (resp["end_s"] < minute_end_s)
        rates = 60.0 / resp["length_s"]
        out["resp_rate_slope_30s"] = _linear_slope(resp["end_s"][rv & (resp["end_s"] >= minute_end_s - 30.0)], rates[rv & (resp["end_s"] >= minute_end_s - 30.0)])
        lengths = resp["length_s"][rv]
        amps = resp["amplitude"][rv]
        out["resp_cycle_length_cv"] = _cv(lengths)
        out["resp_amplitude_cv"] = _cv(amps)
        if finite_values(amps).size >= 3:
            med_amp = float(np.median(amps))
            out["resp_low_amplitude_fraction"] = float(np.mean(amps < 0.25 * med_amp)) if med_amp > 1e-8 else _nan()
    pause_count, longest_pause = detect_resp_pauses(resp_signal, config.sampling_rate_hz, config, filtered=resp.get("filtered"), ordered_extrema=resp.get("ordered_extrema"))
    out["resp_pause_count"] = pause_count
    out["resp_longest_pause_s"] = longest_pause
    return out


def _max_abs_lagged_resp_corr(filtered: np.ndarray, times: np.ndarray, values: np.ndarray, lags: np.ndarray, fs: int, min_count: int = 12) -> float:
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.size == 0 or values.size != times.size or lags.size == 0:
        return _nan()
    idx = np.round((times[None, :] + lags[:, None]) * fs).astype(np.int64)
    in_bounds = (idx >= 0) & (idx < filtered.size)
    resp_values = np.full(idx.shape, np.nan, dtype=np.float64)
    resp_values[in_bounds] = filtered[idx[in_bounds]]
    valid = np.isfinite(resp_values) & np.isfinite(values)[None, :]
    counts = np.sum(valid, axis=1)
    enough = counts >= int(min_count)
    if not np.any(enough):
        return _nan()
    vals = np.where(valid, values[None, :], 0.0)
    resp_clean = np.where(valid, resp_values, 0.0)
    n = counts.astype(np.float64)
    mean_x = np.divide(np.sum(resp_clean, axis=1), n, out=np.full(lags.size, np.nan, dtype=np.float64), where=counts > 0)
    mean_y = np.divide(np.sum(vals, axis=1), n, out=np.full(lags.size, np.nan, dtype=np.float64), where=counts > 0)
    dx = np.where(valid, resp_values - mean_x[:, None], 0.0)
    dy = np.where(valid, values[None, :] - mean_y[:, None], 0.0)
    ssx = np.sum(dx * dx, axis=1)
    ssy = np.sum(dy * dy, axis=1)
    std_x = np.sqrt(np.divide(ssx, n, out=np.zeros_like(ssx), where=counts > 0))
    std_y = np.sqrt(np.divide(ssy, n, out=np.zeros_like(ssy), where=counts > 0))
    denom = np.sqrt(ssx * ssy)
    ok = enough & (std_x >= 1e-8) & (std_y >= 1e-8) & (denom > 0.0)
    if not np.any(ok):
        return _nan()
    corrs = np.full(lags.size, np.nan, dtype=np.float64)
    corrs[ok] = np.sum(dx[ok] * dy[ok], axis=1) / denom[ok]
    return float(np.nanmax(np.abs(corrs[ok])))


def coupling_features(ecg: dict[str, np.ndarray], abp: dict[str, np.ndarray], pleth: dict[str, np.ndarray], resp: dict[str, np.ndarray], minute_start_s: float, minute_end_s: float, config: V8ExtractionConfig = DEFAULT_V8_EXTRACTION_CONFIG) -> dict[str, float]:
    if not resp or "filtered" not in resp:
        return {}
    filtered = np.asarray(resp["filtered"], dtype=np.float64)
    fs = int(config.sampling_rate_hz)
    max_lag = float(config.resp_coupling_max_lag_seconds)
    step = float(config.resp_coupling_lag_step_seconds)
    lags = np.arange(-max_lag, max_lag + 0.5 * step, step)
    out: dict[str, float] = {}
    rr_mask = ecg["valid_rr"] & (ecg["rr_times_s"] >= minute_start_s) & (ecg["rr_times_s"] < minute_end_s)
    rr = ecg["rr_s"][rr_mask]
    rt = ecg["rr_times_s"][rr_mask]
    out["resp_rr_max_abs_correlation"] = _max_abs_lagged_resp_corr(filtered, rt, rr, lags, fs)
    if abp:
        bv = abp["valid"] & (abp["time_s"] >= minute_start_s) & (abp["time_s"] < minute_end_s)
        out["resp_sbp_max_abs_correlation"] = _max_abs_lagged_resp_corr(filtered, abp["time_s"][bv], abp["peak"][bv], lags, fs)
        out["resp_pp_max_abs_correlation"] = _max_abs_lagged_resp_corr(filtered, abp["time_s"][bv], abp["peak"][bv] - abp["foot"][bv], lags, fs)
    if pleth:
        pv = pleth["valid"] & (pleth["time_s"] >= minute_start_s) & (pleth["time_s"] < minute_end_s)
        out["resp_pleth_amplitude_max_abs_correlation"] = _max_abs_lagged_resp_corr(filtered, pleth["time_s"][pv], pleth["amplitude"][pv], lags, fs)
    return out


def rhythm_features(ecg: dict[str, np.ndarray], start_s: float, end_s: float, config: V8ExtractionConfig) -> dict[str, float]:
    if not _validate_ecg_dict(ecg):
        return {}
    if end_s - start_s + 1e-9 < config.rolling_rhythm_history_seconds:
        return {}
    rr = np.asarray(ecg.get("rr_s", []), dtype=np.float64)
    valid = np.asarray(ecg.get("valid_rr", []), dtype=bool)
    same_run = np.asarray(ecg.get("same_run_rr", []), dtype=bool)
    if rr.size < config.minimum_rhythm_rr_count or valid.size != rr.size:
        return {}
    rr_times = np.asarray(ecg.get("rr_times_s", np.arange(rr.size)), dtype=np.float64)
    in_window = (rr_times > start_s) & (rr_times <= end_s)
    valid = valid & in_window
    rr_run_ids = np.asarray(ecg.get("run_ids", []), dtype=np.int64)[1:] if np.asarray(ecg.get("run_ids", [])).size >= 2 else np.full(rr.size, -1, dtype=np.int64)
    longest_times, longest_rr = _longest_valid_rr_run(
        ecg,
        start_s,
        end_s,
        min_count=config.minimum_rhythm_rr_count,
        min_duration_s=config.min_hrv_coverage_seconds,
    )
    out: dict[str, float] = {}
    adj_diff = np.abs(np.diff(longest_rr)) if longest_rr.size >= 2 else np.asarray([], dtype=np.float64)
    if longest_rr.size >= config.minimum_rhythm_rr_count and adj_diff.size >= 3 and _event_span_seconds(longest_times, np.arange(longest_times.size, dtype=np.int64)) >= config.min_hrv_coverage_seconds:
        med_rr = float(np.median(longest_rr))
        out["ecg_rr_irregularity_index_5m"] = float(np.median(adj_diff) / med_rr) if med_rr > 1e-8 else _nan()

    eligible_adj = valid[:-1] & valid[1:] & same_run[:-1] & same_run[1:] & (rr_run_ids[:-1] == rr_run_ids[1:])

    short_long_flags = np.zeros(rr.size, dtype=bool)
    compensatory_flags = np.zeros(rr.size, dtype=bool)
    eligible_positions = 0
    flank = int(config.minimum_local_baseline_intervals)
    for idx in range(rr.size - 1):
        if not eligible_adj[idx]:
            continue
        run_id = rr_run_ids[idx]
        lo = max(0, idx - flank - 1)
        hi = min(rr.size, idx + flank + 2)
        local = valid[lo:hi] & (rr_run_ids[lo:hi] == run_id)
        local_idx = idx - lo
        local[local_idx : min(local_idx + 2, local.size)] = False
        if int(np.sum(local)) < config.minimum_local_baseline_intervals:
            continue
        baseline = float(np.median(rr[lo:hi][local]))
        if baseline <= 1e-8:
            continue
        eligible_positions += 1
        is_short_long = rr[idx] < config.short_rr_ratio_max * baseline and rr[idx + 1] > config.long_rr_ratio_min * baseline
        if is_short_long:
            short_long_flags[idx] = True
            pair_ratio = (rr[idx] + rr[idx + 1]) / (2.0 * baseline)
            if abs(pair_ratio - 1.0) <= config.compensatory_sum_ratio_tolerance:
                compensatory_flags[idx] = True
    if eligible_positions > 0:
        out["ecg_short_long_rr_pattern_fraction_5m"] = float(np.sum(short_long_flags) / eligible_positions)
        out["ecg_compensatory_pause_pattern_fraction_5m"] = float(np.sum(compensatory_flags) / eligible_positions)

    qrs_segments_all = list(np.asarray(ecg.get("qrs_segments", []), dtype=object))
    qrs_valid_all = np.asarray(ecg.get("qrs_segment_valid", []), dtype=bool)
    peak_times = np.asarray(ecg.get("detector_peak_indices", ecg.get("peaks", [])), dtype=np.float64) / config.sampling_rate_hz
    qrs_window = (peak_times >= start_s) & (peak_times <= end_s)
    qrs_segments = [seg if ok else np.asarray([], dtype=np.float64) for seg, ok in zip(qrs_segments_all, (qrs_valid_all & qrs_window).tolist())]
    corr, dist = _template_distances(qrs_segments, config.morphology_template_points)
    comparable = np.isfinite(dist)
    if int(np.sum(comparable)) >= 5:
        outlier = comparable & ((dist > config.qrs_outlier_distance_threshold) | (np.isfinite(corr) & (corr < config.qrs_outlier_correlation_threshold)))
        out["ecg_qrs_morphology_outlier_fraction_5m"] = float(np.sum(outlier) / np.sum(comparable))
        out["ecg_qrs_template_distance_p95_5m"] = safe_nanstat(lambda x: np.percentile(x, 95.0), dist[comparable])
        if eligible_positions > 0:
            peaks = np.asarray(ecg.get("peaks", []), dtype=np.int64)
            ectopic = 0
            for idx in np.flatnonzero(short_long_flags):
                if idx + 1 >= peaks.size:
                    continue
                seg_pos = idx + 1
                if 0 <= seg_pos < outlier.size and outlier[seg_pos]:
                    ectopic += 1
            out["ecg_ectopic_like_beat_fraction_5m"] = float(ectopic / eligible_positions)
    return out


def _pulse_width_at_level(raw_segment: np.ndarray, peak_idx: int, level: float, fs: int) -> float:
    seg = np.asarray(raw_segment, dtype=np.float64)
    if seg.size < 5 or not np.isfinite(seg).all():
        return _nan()
    foot = float(seg[0])
    peak_idx = int(peak_idx)
    if peak_idx <= 0 or peak_idx >= seg.size - 1:
        return _nan()
    peak = float(seg[peak_idx])
    amp = peak - foot
    if peak_idx <= 0 or amp <= 1e-8:
        return _nan()
    norm = (seg - foot) / amp
    asc_candidates = np.flatnonzero((norm[: peak_idx + 1][:-1] <= level) & (norm[: peak_idx + 1][1:] >= level))
    desc_part = norm[peak_idx:]
    desc_candidates = np.flatnonzero((desc_part[:-1] >= level) & (desc_part[1:] <= level))
    if asc_candidates.size == 0 or desc_candidates.size == 0:
        return _nan()
    def interp(idx: int, arr: np.ndarray) -> float:
        y0 = float(arr[idx])
        y1 = float(arr[idx + 1])
        frac = 0.0 if abs(y1 - y0) < 1e-12 else (level - y0) / (y1 - y0)
        return (idx + frac) / fs
    asc = interp(int(asc_candidates[0]), norm)
    desc = (peak_idx / fs) + interp(int(desc_candidates[0]), desc_part)
    width = desc - asc
    return float(width) if width > 0 else _nan()


def pleth_shape_features(pleth: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    if not pleth:
        return {}
    valid = np.asarray(pleth["valid"], dtype=bool) & (np.asarray(pleth.get("run_id", []), dtype=np.int64) >= 0)
    if int(np.sum(valid)) < 3:
        return {}
    fs = config.sampling_rate_hz
    width = np.asarray(pleth["width_s"], dtype=np.float64)
    rise = np.asarray(pleth["rise_s"], dtype=np.float64)
    out: dict[str, float] = {}
    cached_shape = "pleth_shape_matrix" in pleth
    if cached_shape:
        for level in config.pleth_width_levels:
            vals = np.asarray(pleth.get(f"pleth_width_{int(round(level * 100))}_s", []), dtype=np.float64)
            out[f"pleth_width_{int(round(level * 100))}_median_s"] = safe_nanstat(np.median, vals[valid] if vals.size == valid.size else np.asarray([], dtype=np.float64))
        ratio = np.divide(rise, width, out=np.full_like(rise, np.nan, dtype=np.float64), where=width > 1e-8)
        out["pleth_crest_time_fraction_median"] = safe_nanstat(np.median, ratio[valid])
        for key, feature_name in (
            ("pleth_slope_transit_time_s", "pleth_slope_transit_time_median_s"),
            ("pleth_normalized_area", "pleth_normalized_area_median"),
            ("pleth_pulse_skewness", "pleth_pulse_skewness_median"),
            ("pleth_pulse_kurtosis", "pleth_pulse_kurtosis_median"),
        ):
            vals = np.asarray(pleth.get(key, []), dtype=np.float64)
            out[feature_name] = safe_nanstat(np.median, vals[valid] if vals.size == valid.size else np.asarray([], dtype=np.float64))
        matrix = np.asarray(pleth.get("pleth_shape_matrix", []), dtype=np.float64)
        matrix_valid = valid & (matrix.ndim == 2) & (matrix.shape[0] == valid.size)
        if matrix.ndim == 2 and matrix.shape[0] == valid.size:
            matrix_valid = valid & np.all(np.isfinite(matrix), axis=1)
            _, dist = _template_distances_from_matrix(matrix[matrix_valid])
        else:
            dist = np.asarray([], dtype=np.float64)
        comparable = np.isfinite(dist)
        if int(np.sum(comparable)) >= 5:
            out["pleth_morphology_outlier_fraction"] = float(np.mean(dist[comparable] > config.pleth_morphology_outlier_threshold))
            out["pleth_template_distance_p95"] = safe_nanstat(lambda x: np.percentile(x, 95.0), dist[comparable])
        return out

    raw_segments = np.asarray(pleth.get("raw_segments", []), dtype=object)
    morph_segments = np.asarray(pleth.get("morphology_segments", raw_segments), dtype=object)
    local_peak = np.rint(np.asarray(pleth.get("peak_idx", []), dtype=np.float64) - np.asarray(pleth.get("start_idx", []), dtype=np.float64)).astype(int)
    for level in config.pleth_width_levels:
        vals = [_pulse_width_at_level(raw_segments[idx], int(local_peak[idx]), float(level), fs) for idx in np.flatnonzero(valid) if idx < raw_segments.size and idx < local_peak.size]
        out[f"pleth_width_{int(round(level * 100))}_median_s"] = safe_nanstat(np.median, np.asarray(vals, dtype=np.float64))
    amp = np.asarray(pleth["amplitude"], dtype=np.float64)
    ratio = np.divide(rise, width, out=np.full_like(rise, np.nan, dtype=np.float64), where=width > 1e-8)
    out["pleth_crest_time_fraction_median"] = safe_nanstat(np.median, ratio[valid])
    stt: list[float] = []
    norm_area: list[float] = []
    skew_vals: list[float] = []
    kurt_vals: list[float] = []
    shape_segments: list[np.ndarray] = []
    for idx in np.flatnonzero(valid).tolist():
        if idx >= raw_segments.size:
            continue
        raw = np.asarray(raw_segments[idx], dtype=np.float64)
        morph = np.asarray(morph_segments[idx], dtype=np.float64) if idx < morph_segments.size else raw
        if raw.size < 5 or not np.isfinite(raw).all():
            continue
        a = float(amp[idx])
        w = float(width[idx])
        if a > 1e-8 and w > 1e-8:
            area = float(np.trapezoid(np.maximum(raw - raw[0], 0.0), dx=1.0 / fs))
            norm_area.append(area / (a * w))
            norm = _center_scale_segment(resample_segment((raw - raw[0]) / a, config.morphology_template_points))
            if norm is not None:
                skew, kurt = _skew_kurtosis_bias_corrected(norm)
                skew_vals.append(skew)
                kurt_vals.append(kurt)
                shape_segments.append(raw)
        peak_i = int(local_peak[idx]) if idx < local_peak.size else -1
        if morph.size >= 3 and np.isfinite(morph).all() and a > 1e-8 and 0 < peak_i < morph.size:
            d = np.gradient(morph[: peak_i + 1]) * fs
            max_d = float(np.max(d))
            if max_d > 1e-8:
                stt.append(a / max_d)
    out["pleth_slope_transit_time_median_s"] = safe_nanstat(np.median, np.asarray(stt, dtype=np.float64))
    out["pleth_normalized_area_median"] = safe_nanstat(np.median, np.asarray(norm_area, dtype=np.float64))
    out["pleth_pulse_skewness_median"] = safe_nanstat(np.median, np.asarray(skew_vals, dtype=np.float64))
    out["pleth_pulse_kurtosis_median"] = safe_nanstat(np.median, np.asarray(kurt_vals, dtype=np.float64))
    _, dist = _template_distances(shape_segments, config.morphology_template_points)
    comparable = np.isfinite(dist)
    if int(np.sum(comparable)) >= 5:
        out["pleth_morphology_outlier_fraction"] = float(np.mean(dist[comparable] > config.pleth_morphology_outlier_threshold))
        out["pleth_template_distance_p95"] = safe_nanstat(lambda x: np.percentile(x, 95.0), dist[comparable])
    return out


def pleth_fiducial_features(pleth: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    if not pleth:
        return {}
    per = _pleth_fiducials_per_beat(pleth, config)
    checked = int(per["eligible_count"])
    if checked < 5:
        return {}
    notch_valid = np.isfinite(per["notch_idx"])
    secondary_valid = np.isfinite(per["diastolic_peak_idx"])
    out = {
        "pleth_notch_presence_fraction": float(np.sum(notch_valid) / checked),
        "pleth_diastolic_peak_presence_fraction": float(np.sum(secondary_valid) / checked),
    }
    if int(np.sum(secondary_valid)) >= 3:
        out["pleth_diastolic_peak_time_fraction_median"] = safe_nanstat(np.median, per["diastolic_peak_time_fraction"][secondary_valid])
        out["pleth_reflection_index_median"] = safe_nanstat(np.median, per["reflection_index"][secondary_valid])
    if int(np.sum(notch_valid)) >= 5:
        out["pleth_notch_time_fraction_median"] = safe_nanstat(np.median, per["notch_time_fraction"][notch_valid])
        out["pleth_notch_amplitude_ratio_median"] = safe_nanstat(np.median, per["notch_amplitude_ratio"][notch_valid])
        out["pleth_systolic_area_fraction_median"] = safe_nanstat(np.median, per["systolic_area_fraction"][notch_valid])
    both = notch_valid & secondary_valid
    if int(np.sum(both)) >= 5:
        out["pleth_notch_to_diastolic_peak_time_fraction_median"] = safe_nanstat(np.median, per["notch_to_diastolic_peak_time_fraction"][both])
    return out


def _pleth_fiducials_per_beat(pleth: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, np.ndarray | int]:
    key = (
        float(config.pleth_notch_min_drop_fraction),
        float(config.pleth_notch_min_recovery_fraction),
        float(config.pleth_notch_min_candidate_score),
        float(config.pleth_notch_candidate_score_separation),
        int(config.sampling_rate_hz),
    )
    cached = pleth.get("_pleth_fiducials")
    if isinstance(cached, dict) and cached.get("_config_key") == key:
        return cached
    valid = np.asarray(pleth["valid"], dtype=bool)
    segments = np.asarray(pleth.get("morphology_segments", []), dtype=object)
    local_peak = np.rint(np.asarray(pleth.get("peak_idx", []), dtype=np.float64) - np.asarray(pleth.get("start_idx", []), dtype=np.float64)).astype(int)
    fs = config.sampling_rate_hz
    n = valid.size
    notch_idx = np.full(n, np.nan, dtype=np.float64)
    secondary_idx = np.full(n, np.nan, dtype=np.float64)
    notch_time_fraction = np.full(n, np.nan, dtype=np.float64)
    notch_amp_ratio = np.full(n, np.nan, dtype=np.float64)
    secondary_time_fraction = np.full(n, np.nan, dtype=np.float64)
    reflection = np.full(n, np.nan, dtype=np.float64)
    notch_to_secondary_fraction = np.full(n, np.nan, dtype=np.float64)
    systolic_area_fraction = np.full(n, np.nan, dtype=np.float64)
    checked = 0
    for idx in np.flatnonzero(valid).tolist():
        if idx >= segments.size:
            continue
        seg = np.asarray(segments[idx], dtype=np.float64)
        if seg.size < int(0.35 * fs) or not np.isfinite(seg).all():
            continue
        peak_idx = int(local_peak[idx]) if idx < local_peak.size else -1
        if peak_idx <= 0 or peak_idx >= seg.size - 4:
            continue
        amp = float(seg[peak_idx] - seg[0])
        if not np.isfinite(amp) or amp <= 1e-8:
            continue
        checked += 1
        d1 = np.gradient(seg) * fs
        d2 = np.gradient(d1) * fs
        start = peak_idx + max(2, int(0.08 * fs))
        end = min(seg.size - 3, peak_idx + int(0.6 * fs), seg.size - max(3, int(0.08 * fs)))
        notch_candidates: list[tuple[float, int]] = []
        curv_scale = max(float(np.nanpercentile(np.abs(d2[start:end]), 75.0)) if end > start else 0.0, 1e-8)
        d1_scale = max(float(np.nanpercentile(np.abs(d1[start:end]), 75.0)) if end > start else 0.0, 1e-8)
        for pos in range(start + 1, end):
            is_local_min = seg[pos] <= seg[pos - 1] and seg[pos] < seg[pos + 1]
            derivative_crosses = d1[pos - 1] < 0.0 <= d1[pos + 1]
            curvature_positive = d2[pos] > 0.0
            fall = float(seg[peak_idx] - seg[pos])
            recovery_end = min(seg.size, pos + max(3, int(0.18 * fs)))
            recovery = float(np.max(seg[pos + 1 : recovery_end]) - seg[pos]) if recovery_end > pos + 2 else 0.0
            amp_ratio = float((seg[pos] - seg[0]) / amp)
            if (
                not is_local_min
                or not derivative_crosses
                or not curvature_positive
                or fall < config.pleth_notch_min_drop_fraction * amp
                or recovery < config.pleth_notch_min_recovery_fraction * amp
                or amp_ratio < -0.10
                or amp_ratio > 1.05
            ):
                continue
            frac = pos / max(seg.size - 1, 1)
            timing_score = 1.0 if 0.35 <= frac <= 0.85 else max(0.0, 1.0 - min(abs(frac - 0.60) / 0.35, 1.0))
            fall_score = min(fall / max(0.20 * amp, 1e-8), 2.0) / 2.0
            recovery_score = min(recovery / max(0.12 * amp, 1e-8), 2.0) / 2.0
            curvature_score = min(float(d2[pos]) / curv_scale, 3.0) / 3.0
            cross_score = min(abs(float(d1[pos + 1] - d1[pos - 1])) / d1_scale, 3.0) / 3.0
            notch_candidates.append((timing_score + fall_score + recovery_score + curvature_score + cross_score, pos))
        notch = None
        if notch_candidates:
            notch_candidates.sort(reverse=True)
            best_score, best_pos = notch_candidates[0]
            separated = len(notch_candidates) == 1 or best_score - notch_candidates[1][0] >= config.pleth_notch_candidate_score_separation
            if best_score >= config.pleth_notch_min_candidate_score and separated:
                notch = int(best_pos)
        if notch is None:
            continue
        notch_idx[idx] = notch
        notch_time_fraction[idx] = float(notch / max(seg.size - 1, 1))
        notch_amp_ratio[idx] = float((seg[notch] - seg[0]) / amp)
        baseline = np.linspace(float(seg[0]), float(seg[-1]), seg.size)
        sys_area = float(np.trapezoid(np.maximum(seg[: notch + 1] - baseline[: notch + 1], 0.0), dx=1.0 / fs))
        dias_area = float(np.trapezoid(np.maximum(seg[notch:] - baseline[notch:], 0.0), dx=1.0 / fs))
        if sys_area + dias_area > 1e-12:
            systolic_area_fraction[idx] = sys_area / (sys_area + dias_area)
        peak_start = notch + max(2, int(0.05 * fs))
        peak_end = min(seg.size - max(3, int(0.08 * fs)), notch + int(0.45 * fs))
        if peak_end > peak_start + 3:
            post = seg[peak_start:peak_end]
            peaks, props = signal.find_peaks(post, prominence=max(0.04 * amp, 1e-6))
            scored: list[tuple[float, int]] = []
            for rel, prom in zip(peaks.tolist(), props.get("prominences", np.zeros(peaks.size)).tolist()):
                sec = peak_start + int(rel)
                rise = float(seg[sec] - seg[notch])
                if rise < config.pleth_notch_min_recovery_fraction * amp:
                    continue
                frac = sec / max(seg.size - 1, 1)
                timing_score = 1.0 if 0.45 <= frac <= 0.90 else max(0.0, 1.0 - min(abs(frac - 0.65) / 0.35, 1.0))
                prominence_score = min(float(prom) / max(0.10 * amp, 1e-8), 2.0) / 2.0
                rise_score = min(rise / max(0.10 * amp, 1e-8), 2.0) / 2.0
                early_score = 1.0 - 0.25 * ((sec - peak_start) / max(peak_end - peak_start, 1))
                scored.append((timing_score + prominence_score + rise_score + early_score, sec))
            if scored:
                scored.sort(reverse=True)
                best_score, sec = scored[0]
                separated = len(scored) == 1 or best_score - scored[1][0] >= config.pleth_notch_candidate_score_separation
                if best_score >= config.pleth_notch_min_candidate_score and separated:
                    secondary_idx[idx] = sec
                    secondary_time_fraction[idx] = float(sec / max(seg.size - 1, 1))
                    reflection[idx] = float((seg[sec] - seg[0]) / amp)
                    notch_to_secondary_fraction[idx] = float((sec - notch) / max(seg.size - 1, 1))
    result = {
        "_config_key": key,
        "eligible_count": checked,
        "notch_idx": notch_idx,
        "diastolic_peak_idx": secondary_idx,
        "notch_time_fraction": notch_time_fraction,
        "notch_amplitude_ratio": notch_amp_ratio,
        "diastolic_peak_time_fraction": secondary_time_fraction,
        "reflection_index": reflection,
        "notch_to_diastolic_peak_time_fraction": notch_to_secondary_fraction,
        "systolic_area_fraction": systolic_area_fraction,
    }
    pleth["_pleth_fiducials"] = result
    return result


def _odd_savgol_window(fs: int, seconds: float, n: int, polyorder: int) -> int:
    win = max(int(round(seconds * fs)), polyorder + 2)
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n if n % 2 == 1 else n - 1
    return max(win, polyorder + 2 + ((polyorder + 2) % 2 == 0))


def _first_local_min(x: np.ndarray, start: int, end: int, threshold: float) -> int | None:
    if end <= start + 2:
        return None
    peaks, props = signal.find_peaks(-x[start:end], prominence=threshold)
    if peaks.size:
        return start + int(peaks[0])
    return None


def _local_maxima(x: np.ndarray, start: int, end: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if end <= start + 2:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    peaks, props = signal.find_peaks(x[start:end], prominence=threshold)
    return start + peaks.astype(np.int64), np.asarray(props.get("prominences", np.zeros(peaks.size)), dtype=np.float64)


def _first_zero_crossing(x: np.ndarray, start: int, end: int) -> int | None:
    if end <= start + 1:
        return None
    region = x[start:end]
    crossings = np.flatnonzero(np.diff(np.signbit(region)) != 0)
    return start + int(crossings[0] + 1) if crossings.size else None


def _unique_zero_crossing(x: np.ndarray, start: int, end: int, direction: str | None = None) -> int | None:
    if end <= start + 1:
        return None
    region = np.asarray(x[start:end], dtype=np.float64)
    if region.size < 2 or not np.isfinite(region).all():
        return None
    crossings = []
    for rel in np.flatnonzero(np.diff(np.signbit(region)) != 0).tolist():
        left = float(region[rel])
        right = float(region[rel + 1])
        if direction == "pos_to_neg" and not (left > 0.0 and right <= 0.0):
            continue
        if direction == "neg_to_pos" and not (left < 0.0 and right >= 0.0):
            continue
        crossings.append(start + rel + 1)
    return int(crossings[0]) if len(crossings) == 1 else None


def _pleth_derivative_fiducials_for_segment(
    seg: np.ndarray,
    peak_idx: int,
    fs: int,
    config: V8ExtractionConfig,
    notch_idx: int | None = None,
    diastolic_peak_idx: int | None = None,
    vpg_segment: np.ndarray | None = None,
    apg_segment: np.ndarray | None = None,
    jpg_segment: np.ndarray | None = None,
) -> dict[str, float]:
    arr = np.asarray(seg, dtype=np.float64)
    keys = ["u", "v", "w", "a", "b", "c", "d", "e", "b_over_a", "c_over_a", "d_over_a", "e_over_a", "aging"]
    missing = {key: _nan() for key in keys}
    if arr.size < max(20, int(0.30 * fs)) or not np.isfinite(arr).all() or peak_idx <= 2 or peak_idx >= arr.size - 5:
        return missing
    amp = float(arr[peak_idx] - arr[0])
    if amp <= 1e-8:
        return missing
    norm = (arr - arr[0]) / amp
    jpg = None
    if vpg_segment is not None and apg_segment is not None:
        vpg_raw = np.asarray(vpg_segment, dtype=np.float64)
        apg_raw = np.asarray(apg_segment, dtype=np.float64)
        if vpg_raw.size != arr.size or apg_raw.size != arr.size or not np.isfinite(vpg_raw).all() or not np.isfinite(apg_raw).all():
            return missing
        vpg = vpg_raw / amp
        apg = apg_raw / amp
        if jpg_segment is not None:
            jpg_raw = np.asarray(jpg_segment, dtype=np.float64)
            if jpg_raw.size == arr.size and np.isfinite(jpg_raw).all():
                jpg = jpg_raw / amp
    else:
        poly = min(config.pleth_derivative_polynomial_order, max(2, arr.size - 2))
        win = _odd_savgol_window(fs, config.pleth_derivative_smoothing_seconds, arr.size, poly)
        if win <= poly or win < 5:
            return missing
        vpg = signal.savgol_filter(norm, window_length=win, polyorder=poly, deriv=1, delta=1.0 / fs, mode="interp")
        apg = signal.savgol_filter(norm, window_length=win, polyorder=poly, deriv=2, delta=1.0 / fs, mode="interp")
        jpg = signal.savgol_filter(norm, window_length=win, polyorder=poly, deriv=3, delta=1.0 / fs, mode="interp")
    vpg_range = max(float(np.ptp(vpg)), 1e-8)
    apg_range = max(float(np.ptp(apg)), 1e-8)
    vpg_prom = max(config.pleth_derivative_minimum_prominence * vpg_range, config.pleth_vpg_min_prominence_fraction * vpg_range)
    apg_prom = max(config.pleth_derivative_minimum_prominence * apg_range, config.pleth_apg_min_prominence_fraction * apg_range)
    out = dict(missing)
    dpp = int(diastolic_peak_idx) if diastolic_peak_idx is not None and np.isfinite(diastolic_peak_idx) else arr.size - 1
    dn = int(notch_idx) if notch_idx is not None and np.isfinite(notch_idx) else max(peak_idx + 1, int(0.45 * arr.size))
    dpp = min(max(dpp, peak_idx + 1), arr.size - 1)
    dn = min(max(dn, peak_idx + 1), arr.size - 2)

    u_region = vpg[1 : peak_idx + 1]
    if u_region.size < 3:
        return out
    u = 1 + int(np.argmax(u_region))
    if u <= 1 or u >= peak_idx - 1 or not np.isfinite(vpg[u]) or vpg[u] < vpg_prom:
        return out
    out["u"] = float(u)
    if u < dpp - 2:
        v_region = vpg[u + 1 : dpp + 1]
        if v_region.size >= 3:
            v = u + 1 + int(np.argmin(v_region))
            if vpg[u] - vpg[v] >= vpg_prom:
                out["v"] = float(v)
    w = None
    w_candidates, _ = _local_maxima(vpg, dn + 1, min(arr.size, dpp + 1), 0.5 * vpg_prom)
    if w_candidates.size:
        w = int(w_candidates[0])
    else:
        zc = _first_zero_crossing(apg, dn + 1, min(arr.size, dpp + 1))
        if zc is not None and zc > dn:
            w = zc
    if w is not None:
        out["w"] = float(w)

    a_region = apg[1 : peak_idx + 1]
    if a_region.size < 3:
        return out
    a = 1 + int(np.argmax(a_region))
    if a <= 1 or a >= peak_idx - 1 or not np.isfinite(apg[a]) or apg[a] < apg_prom:
        return out
    out["a"] = float(a)
    b = _first_local_min(apg, a + 1, min(arr.size, a + int(0.35 * fs)), 0.5 * apg_prom)
    if b is None:
        return out
    out["b"] = float(b)
    e_upper = min(dpp, int(round(0.60 * max(arr.size - 1, 1))), arr.size - 1)
    if e_upper <= b + 2:
        return out
    e_peaks, _ = _local_maxima(apg, b + 1, e_upper + 1, 0.4 * apg_prom)
    if e_peaks.size == 0:
        return out
    e = int(e_peaks[np.argmax(apg[e_peaks])])
    out["e"] = float(e)
    c_peaks, _ = _local_maxima(apg, b + 1, e, 0.25 * apg_prom)
    if c_peaks.size:
        c = int(c_peaks[np.argmax(apg[c_peaks])])
        out["c"] = float(c)
    else:
        c = _unique_zero_crossing(jpg, b + 1, e, direction="pos_to_neg") if jpg is not None else None
        if c is not None:
            out["c"] = float(c)
    if c is not None:
        d = _first_local_min(apg, c + 1, e, 0.25 * apg_prom)
        if d is None:
            d = _unique_zero_crossing(jpg, c + 1, e, direction="neg_to_pos") if jpg is not None else None
        if d is not None and c < d < e:
            out["d"] = float(d)
    aval = float(apg[a])
    if abs(aval) <= 1e-8:
        return out
    for fid, ratio_name in (("b", "b_over_a"), ("c", "c_over_a"), ("d", "d_over_a"), ("e", "e_over_a")):
        if np.isfinite(out[fid]):
            out[ratio_name] = float(apg[int(out[fid])] / aval)
    if all(np.isfinite(out[key]) for key in ("b_over_a", "c_over_a", "d_over_a", "e_over_a")):
        out["aging"] = float(out["b_over_a"] - out["c_over_a"] - out["d_over_a"] - out["e_over_a"])
    return out


def pleth_derivative_fiducial_features(pleth: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    if not pleth or not _validate_pulse_dict(pleth, required=("valid", "morphology_segments", "peak_idx", "start_idx")):
        return {}
    valid = np.asarray(pleth.get("valid", []), dtype=bool)
    segments = np.asarray(pleth.get("derivative_morphology_segments", pleth.get("morphology_segments", [])), dtype=object)
    vpg_segments = np.asarray(pleth.get("vpg_segments", []), dtype=object)
    apg_segments = np.asarray(pleth.get("apg_segments", []), dtype=object)
    jpg_segments = np.asarray(pleth.get("jpg_segments", []), dtype=object)
    local_peak = np.rint(np.asarray(pleth.get("peak_idx", []), dtype=np.float64) - np.asarray(pleth.get("start_idx", []), dtype=np.float64)).astype(int)
    per_basic = _pleth_fiducials_per_beat(pleth, config)
    eligible = 0
    rows: list[dict[str, float]] = []
    fs = config.sampling_rate_hz
    for idx in np.flatnonzero(valid).tolist():
        if idx >= segments.size or idx >= local_peak.size:
            continue
        seg = np.asarray(segments[idx], dtype=np.float64)
        peak_i = int(local_peak[idx])
        if seg.size < 5 or not np.isfinite(seg).all() or peak_i <= 0 or peak_i >= seg.size - 1:
            continue
        eligible += 1
        notch = per_basic["notch_idx"][idx] if idx < len(per_basic["notch_idx"]) else np.nan
        dpp = per_basic["diastolic_peak_idx"][idx] if idx < len(per_basic["diastolic_peak_idx"]) else np.nan
        row = _pleth_derivative_fiducials_for_segment(
            seg,
            peak_i,
            fs,
            config,
            int(notch) if np.isfinite(notch) else None,
            int(dpp) if np.isfinite(dpp) else None,
            np.asarray(vpg_segments[idx], dtype=np.float64) if idx < vpg_segments.size else None,
            np.asarray(apg_segments[idx], dtype=np.float64) if idx < apg_segments.size else None,
            np.asarray(jpg_segments[idx], dtype=np.float64) if idx < jpg_segments.size else None,
        )
        row["length"] = float(max(seg.size - 1, 1))
        rows.append(row)
    if eligible < config.min_pleth_derivative_beats:
        return {}
    complete = [row for row in rows if all(np.isfinite(row.get(key, np.nan)) for key in ("u", "v", "w", "a", "b", "c", "d", "e"))]
    out = {"pleth_derivative_fiducial_valid_fraction": float(len(complete) / eligible)}
    for key, name in (
        ("u", "pleth_vpg_u_time_fraction_median"),
        ("v", "pleth_vpg_v_time_fraction_median"),
        ("w", "pleth_vpg_w_time_fraction_median"),
        ("a", "pleth_apg_a_time_fraction_median"),
        ("b", "pleth_apg_b_time_fraction_median"),
        ("c", "pleth_apg_c_time_fraction_median"),
        ("d", "pleth_apg_d_time_fraction_median"),
        ("e", "pleth_apg_e_time_fraction_median"),
    ):
        vals = np.asarray([row[key] / row["length"] for row in rows if np.isfinite(row.get(key, np.nan))], dtype=np.float64)
        if vals.size >= config.min_pleth_derivative_beats:
            out[name] = safe_nanstat(np.median, vals)
    for key, name in (
        ("b_over_a", "pleth_apg_b_over_a_median"),
        ("c_over_a", "pleth_apg_c_over_a_median"),
        ("d_over_a", "pleth_apg_d_over_a_median"),
        ("e_over_a", "pleth_apg_e_over_a_median"),
        ("aging", "pleth_apg_aging_index_median"),
    ):
        vals = np.asarray([row[key] for row in rows if np.isfinite(row.get(key, np.nan))], dtype=np.float64)
        if vals.size >= config.min_pleth_derivative_beats:
            out[name] = safe_nanstat(np.median, vals)
    return out


def pleth_experimental_morphology_dynamics_features(pleth_history: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    if not pleth_history:
        return {}
    out: dict[str, float] = {}
    per = _pleth_fiducials_per_beat(pleth_history, config)
    reflection = np.asarray(per["reflection_index"], dtype=np.float64)
    times = np.asarray(pleth_history.get("time_s", []), dtype=np.float64)
    if config.enable_pleth_fiducials:
        reflection_idx = _longest_valid_pulse_run(
            pleth_history,
            np.isfinite(reflection) & np.isfinite(times),
            min_count=config.min_pleth_morphology_dynamics_beats,
            min_duration_s=config.min_pleth_morphology_dynamics_coverage_seconds,
        )
        if (
            reflection_idx.size >= config.min_pleth_morphology_dynamics_beats
            and _event_span_seconds(times, reflection_idx) >= config.min_pleth_morphology_dynamics_coverage_seconds
        ):
            out["pleth_reflection_index_iqr_5m"] = nan_iqr(reflection[reflection_idx])
            out["pleth_reflection_index_slope_5m"] = _linear_slope(times[reflection_idx], reflection[reflection_idx], config.min_pleth_morphology_dynamics_beats)

    if not config.enable_pleth_derivative_fiducials:
        return out
    valid = np.asarray(pleth_history.get("valid", []), dtype=bool)
    segments = np.asarray(pleth_history.get("derivative_morphology_segments", pleth_history.get("morphology_segments", [])), dtype=object)
    vpg_segments = np.asarray(pleth_history.get("vpg_segments", []), dtype=object)
    apg_segments = np.asarray(pleth_history.get("apg_segments", []), dtype=object)
    jpg_segments = np.asarray(pleth_history.get("jpg_segments", []), dtype=object)
    local_peak = np.rint(np.asarray(pleth_history.get("peak_idx", []), dtype=np.float64) - np.asarray(pleth_history.get("start_idx", []), dtype=np.float64)).astype(int)
    b_over_a = np.full(valid.size, np.nan, dtype=np.float64)
    fs = config.sampling_rate_hz
    per_basic = _pleth_fiducials_per_beat(pleth_history, config)
    for idx in np.flatnonzero(valid).tolist():
        if idx >= segments.size or idx >= local_peak.size:
            continue
        notch = per_basic["notch_idx"][idx] if idx < len(per_basic["notch_idx"]) else np.nan
        dpp = per_basic["diastolic_peak_idx"][idx] if idx < len(per_basic["diastolic_peak_idx"]) else np.nan
        row = _pleth_derivative_fiducials_for_segment(
            np.asarray(segments[idx], dtype=np.float64),
            int(local_peak[idx]),
            fs,
            config,
            int(notch) if np.isfinite(notch) else None,
            int(dpp) if np.isfinite(dpp) else None,
            np.asarray(vpg_segments[idx], dtype=np.float64) if idx < vpg_segments.size else None,
            np.asarray(apg_segments[idx], dtype=np.float64) if idx < apg_segments.size else None,
            np.asarray(jpg_segments[idx], dtype=np.float64) if idx < jpg_segments.size else None,
        )
        if np.isfinite(row.get("b_over_a", np.nan)):
            b_over_a[idx] = float(row["b_over_a"])
    ratio_idx = _longest_valid_pulse_run(
        pleth_history,
        np.isfinite(b_over_a),
        min_count=config.min_pleth_derivative_beats,
        min_duration_s=config.min_pleth_morphology_dynamics_coverage_seconds,
    )
    if ratio_idx.size >= config.min_pleth_derivative_beats and _event_span_seconds(times, ratio_idx) >= config.min_pleth_morphology_dynamics_coverage_seconds:
        out["pleth_apg_b_over_a_iqr_5m"] = nan_iqr(b_over_a[ratio_idx])
    return out


def _adjacent_pulse_diffs(pulses: dict[str, np.ndarray], values: np.ndarray) -> np.ndarray:
    valid = np.asarray(pulses["valid"], dtype=bool)
    run = np.asarray(pulses.get("run_id", np.full(valid.size, -1)), dtype=np.int64)
    ok = valid[:-1] & valid[1:] & (run[:-1] == run[1:]) & (run[:-1] >= 0)
    return np.diff(np.asarray(values, dtype=np.float64))[ok] if ok.size else np.asarray([], dtype=np.float64)


def _pulse_transition_valid(pulses: dict[str, np.ndarray]) -> np.ndarray:
    valid = np.asarray(pulses["valid"], dtype=bool)
    run = np.asarray(pulses.get("run_id", np.full(valid.size, -1)), dtype=np.int64)
    if valid.size < 2:
        return np.asarray([], dtype=bool)
    return valid[:-1] & valid[1:] & (run[:-1] == run[1:]) & (run[:-1] >= 0)


def _adjacent_pulse_ratios(pulses: dict[str, np.ndarray], values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    ok = _pulse_transition_valid(pulses)
    positive = ok & (arr[:-1] > 1e-8) & (arr[1:] > 1e-8)
    if positive.size == 0:
        return np.asarray([], dtype=np.float64)
    return np.maximum(arr[1:][positive] / arr[:-1][positive], arr[:-1][positive] / arr[1:][positive])


def abp_lability_features(abp: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    if not abp:
        return {}
    valid = np.asarray(abp["valid"], dtype=bool)
    run = np.asarray(abp.get("run_id", np.full(valid.size, -1)), dtype=np.int64)
    if int(np.sum(valid)) < 3:
        return {}
    sbp = np.asarray(abp["peak"], dtype=np.float64)
    dbp = np.asarray(abp["foot"], dtype=np.float64)
    mapv = np.asarray(abp["mean"], dtype=np.float64)
    pp = sbp - dbp
    out: dict[str, float] = {}
    for prefix, series in (("sbp", sbp), ("map", mapv), ("pp", pp), ("dbp", dbp)):
        diffs = _adjacent_pulse_diffs(abp, series)
        if diffs.size >= config.minimum_abp_successive_pairs:
            out[f"abp_{prefix}_arv_mmhg"] = float(np.mean(np.abs(diffs)))
            out[f"abp_{prefix}_rmssd_mmhg"] = float(np.sqrt(np.mean(diffs * diffs)))
            if prefix in {"sbp", "map", "dbp"}:
                out[f"abp_{prefix}_abs_successive_change_p95_mmhg"] = float(np.percentile(np.abs(diffs), 95.0))
    for k in (3, 5):
        drops: list[float] = []
        for idx in range(mapv.size - k + 1):
            window_valid = valid[idx : idx + k]
            window_run = run[idx : idx + k]
            if np.all(window_valid) and np.all(window_run == window_run[0]) and window_run[0] >= 0:
                drops.append(float(mapv[idx] - mapv[idx + k - 1]))
        if drops:
            out[f"abp_max_{k}beat_map_drop_mmhg"] = max(max(drops), 0.0)
    transition_valid = _pulse_transition_valid(abp)
    decline = transition_valid & (np.diff(mapv) <= -config.abp_decline_min_mmhg)
    if transition_valid.size and np.any(transition_valid):
        longest_transitions = 0
        current = 0
        for flag in decline.tolist():
            current = current + 1 if flag else 0
            longest_transitions = max(longest_transitions, current)
        out["abp_longest_consecutive_map_decline_beats"] = float(longest_transitions + 1 if longest_transitions else 0)
        out["abp_consecutive_map_decline_fraction"] = float(np.sum(decline) / np.sum(transition_valid))
    segments = [seg if ok else np.asarray([], dtype=np.float64) for seg, ok in zip(np.asarray(abp.get("morphology_segments", []), dtype=object), valid.tolist())]
    _, dist = _template_distances(segments, config.morphology_template_points)
    comparable = np.isfinite(dist)
    if int(np.sum(comparable)) >= 5:
        out["abp_morphology_outlier_fraction"] = float(np.mean(dist[comparable] > config.abp_morphology_outlier_threshold))
        out["abp_template_distance_p95"] = safe_nanstat(lambda x: np.percentile(x, 95.0), dist[comparable])
    return out


def _longest_valid_pulse_run(pulses: dict[str, np.ndarray], extra_valid: np.ndarray | None = None, min_count: int = 1, min_duration_s: float = 0.0) -> np.ndarray:
    valid = np.asarray(pulses.get("valid", []), dtype=bool)
    run = np.asarray(pulses.get("run_id", []), dtype=np.int64)
    times = np.asarray(pulses.get("time_s", []), dtype=np.float64)
    if not (run.size == valid.size == times.size):
        return np.asarray([], dtype=np.int64)
    if extra_valid is not None:
        extra = np.asarray(extra_valid, dtype=bool)
        if extra.size != valid.size:
            return np.asarray([], dtype=np.int64)
        valid = valid & extra
    return select_best_continuous_event_run(times, valid, run, min_count=min_count, min_duration_s=min_duration_s)


def _event_span_seconds(times_s: np.ndarray, idx: np.ndarray) -> float:
    if idx.size < 2:
        return 0.0
    vals = np.asarray(times_s, dtype=np.float64)[idx]
    vals = vals[np.isfinite(vals)]
    return float(np.max(vals) - np.min(vals)) if vals.size >= 2 else 0.0


def abp_nonlinear_dynamics_features(abp_history: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    if not abp_history or not _validate_pulse_dict(abp_history, required=("valid", "run_id", "peak", "foot", "mean", "time_s", "foot_time_s")):
        return {}
    sbp = np.asarray(abp_history.get("peak", []), dtype=np.float64)
    dbp = np.asarray(abp_history.get("foot", []), dtype=np.float64)
    mapv = np.asarray(abp_history.get("mean", []), dtype=np.float64)
    peak_time = np.asarray(abp_history.get("time_s", []), dtype=np.float64)
    foot_time = np.asarray(abp_history.get("foot_time_s", []), dtype=np.float64)
    if sbp.size == 0 or not (sbp.size == dbp.size == mapv.size == peak_time.size == foot_time.size):
        return {}
    pp = sbp - dbp
    transition_ok = _pulse_transition_valid(abp_history)
    seq_valid = np.isfinite(sbp) & np.isfinite(dbp) & np.isfinite(mapv) & np.isfinite(pp)
    idx = _longest_valid_pulse_run(
        abp_history,
        seq_valid,
        min_count=config.min_abp_nonlinear_beats,
        min_duration_s=config.min_abp_nonlinear_coverage_seconds,
    )
    out: dict[str, float] = {}
    if idx.size >= config.min_abp_nonlinear_beats and _event_span_seconds(peak_time, idx) >= config.min_abp_nonlinear_coverage_seconds:
        series_map = {
            "sbp": sbp[idx],
            "dbp": dbp[idx],
            "map": mapv[idx],
            "pp": pp[idx],
        }
        for name, values in series_map.items():
            out[f"abp_{name}_sampen_5m"] = sample_entropy(values, min_count=config.min_abp_nonlinear_beats)
        interval_valid = (
            transition_ok
            & seq_valid[:-1]
            & seq_valid[1:]
            & np.isfinite(peak_time[:-1])
            & np.isfinite(foot_time[1:])
            & (foot_time[1:] > peak_time[:-1])
        )
        interval_run = np.asarray(abp_history.get("run_id", []), dtype=np.int64)[:-1]
        trans_idx = select_best_continuous_event_run(
            peak_time[:-1],
            interval_valid,
            interval_run,
            min_count=config.min_abp_nonlinear_beats,
            min_duration_s=config.min_abp_nonlinear_coverage_seconds,
        )
        best_interval = foot_time[trans_idx + 1] - peak_time[trans_idx] if trans_idx.size else np.asarray([], dtype=np.float64)
        if best_interval.size >= config.min_abp_nonlinear_beats:
            out["abp_peak_to_dbp_interval_sampen_5m"] = sample_entropy(best_interval, min_count=config.min_abp_nonlinear_beats)
    for prefix, series in (("sbp", sbp), ("dbp", dbp)):
        diff_idx = _longest_valid_pulse_run(
            abp_history,
            np.isfinite(series) & np.isfinite(peak_time),
            min_count=config.minimum_abp_successive_pairs + 1,
            min_duration_s=config.min_abp_nonlinear_coverage_seconds,
        )
        diffs = np.diff(series[diff_idx]) if diff_idx.size >= 2 else np.asarray([], dtype=np.float64)
        if diffs.size >= config.minimum_abp_successive_pairs and _event_span_seconds(peak_time, diff_idx) >= config.min_abp_nonlinear_coverage_seconds:
            out[f"abp_{prefix}_successive_diff_median_5m"] = safe_nanstat(np.median, diffs)
            out[f"abp_{prefix}_successive_diff_iqr_5m"] = nan_iqr(diffs)
    return out


def morphology_dynamics_features(abp_history: dict[str, np.ndarray], pleth_history: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    out: dict[str, float] = {}
    if abp_history:
        area = np.asarray(abp_history.get("area", []), dtype=np.float64)
        dpdt = np.asarray(abp_history.get("dpdt_max", []), dtype=np.float64)
        times = np.asarray(abp_history.get("time_s", []), dtype=np.float64)
        area_idx = _longest_valid_pulse_run(
            abp_history,
            np.isfinite(area) & np.isfinite(times),
            min_count=config.min_abp_nonlinear_beats,
            min_duration_s=config.min_abp_morphology_dynamics_coverage_seconds,
        )
        if area_idx.size >= config.min_abp_nonlinear_beats and _event_span_seconds(times, area_idx) >= config.min_abp_morphology_dynamics_coverage_seconds:
            out["abp_pulse_area_cv_5m"] = _cv(area[area_idx], min_count=config.min_abp_nonlinear_beats)
        dpdt_idx = _longest_valid_pulse_run(
            abp_history,
            np.isfinite(dpdt) & np.isfinite(times),
            min_count=config.min_abp_nonlinear_beats,
            min_duration_s=config.min_abp_morphology_dynamics_coverage_seconds,
        )
        if dpdt_idx.size >= config.min_abp_nonlinear_beats and _event_span_seconds(times, dpdt_idx) >= config.min_abp_morphology_dynamics_coverage_seconds:
            out["abp_dpdt_max_cv_5m"] = _cv(dpdt[dpdt_idx], min_count=config.min_abp_nonlinear_beats)
            out["abp_dpdt_max_slope_5m"] = _linear_slope(times[dpdt_idx], dpdt[dpdt_idx], min_count=config.min_abp_nonlinear_beats)
    if pleth_history:
        width = np.asarray(pleth_history.get("width_s", []), dtype=np.float64)
        times = np.asarray(pleth_history.get("time_s", []), dtype=np.float64)
        width_idx = _longest_valid_pulse_run(
            pleth_history,
            np.isfinite(width) & np.isfinite(times),
            min_count=config.min_pleth_morphology_dynamics_beats,
            min_duration_s=config.min_pleth_morphology_dynamics_coverage_seconds,
        )
        if width_idx.size >= config.min_pleth_morphology_dynamics_beats and _event_span_seconds(times, width_idx) >= config.min_pleth_morphology_dynamics_coverage_seconds:
            out["pleth_width_cv_5m"] = _cv(width[width_idx], min_count=config.min_pleth_morphology_dynamics_beats)
    return out


def _select_longest_event_run(times_s: np.ndarray, values: np.ndarray, run_ids: np.ndarray, config: V8ExtractionConfig) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    runs = np.asarray(run_ids, dtype=np.int64)
    if times.size != vals.size or runs.size != times.size:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    valid = np.isfinite(times) & np.isfinite(vals) & (runs >= 0)
    idx = select_best_continuous_event_run(
        times,
        valid,
        runs,
        min_count=config.derived_resp_minimum_events,
        min_duration_s=config.derived_resp_minimum_duration_seconds,
    )
    return times[idx], vals[idx]


def _derived_resp_from_surrogate(times_s: np.ndarray, surrogate: np.ndarray, prefix: str, config: V8ExtractionConfig, run_ids: np.ndarray | None = None) -> dict[str, float]:
    if run_ids is None:
        return {}
    t, x = _select_longest_event_run(times_s, surrogate, run_ids, config)
    if x.size < config.derived_resp_minimum_events or t.size < 2 or np.ptp(t) < config.derived_resp_minimum_duration_seconds:
        return {}
    x = signal.detrend(x - np.median(x), type="linear")
    if np.std(x) < 1e-8:
        return {}
    lo = config.derived_resp_frequency_low_hz
    hi = config.derived_resp_frequency_high_hz
    freqs = np.linspace(lo, hi, int(config.derived_resp_frequency_bins))
    power = signal.lombscargle(t - t[0], x, 2.0 * np.pi * freqs, normalize=True)
    if power.size == 0 or not np.isfinite(power).any():
        return {}
    peak_idx = int(np.nanargmax(power))
    p = np.maximum(power, 0.0)
    total = float(np.trapezoid(p, freqs))
    if total <= 1e-12:
        return {}
    peak_freq = float(freqs[peak_idx])
    peak_band = (freqs >= peak_freq - config.derived_resp_peak_half_width_hz) & (freqs <= peak_freq + config.derived_resp_peak_half_width_hz)
    strength = float(np.trapezoid(p[peak_band], freqs[peak_band]) / total) if np.any(peak_band) else _nan()
    if strength < config.derived_resp_minimum_peak_strength:
        return {}
    return {
        f"{prefix}_derived_resp_rate_5m": float(peak_freq * 60.0),
        f"{prefix}_derived_resp_strength_5m": strength,
    }


def _pleth_run_local_resp_surrogate(pleth: dict[str, np.ndarray], config: V8ExtractionConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(pleth.get("valid", []), dtype=bool)
    times = np.asarray(pleth.get("time_s", []), dtype=np.float64)
    run_ids = np.asarray(pleth.get("run_id", []), dtype=np.int64)
    amp = np.asarray(pleth.get("amplitude", []), dtype=np.float64)
    area = np.asarray(pleth.get("area", []), dtype=np.float64)
    if not (valid.size == times.size == run_ids.size == amp.size == area.size):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64)
    eligible = valid & np.isfinite(times) & (run_ids >= 0)
    best = select_best_continuous_event_run(
        times,
        eligible,
        run_ids,
        min_count=config.derived_resp_minimum_events,
        min_duration_s=config.derived_resp_minimum_duration_seconds,
    )
    surrogate = np.full(times.size, np.nan, dtype=np.float64)
    if best.size == 0:
        return times, surrogate, run_ids
    comps: list[np.ndarray] = []
    for values in (amp, area):
        vals = values[best]
        finite = np.isfinite(vals)
        if np.sum(finite) < 3:
            continue
        scale = _v8_scale(vals[finite])
        if not np.isfinite(scale):
            continue
        comp = np.full(best.size, np.nan, dtype=np.float64)
        comp[finite] = (vals[finite] - float(np.median(vals[finite]))) / scale
        comps.append(comp)
    if comps:
        stack = np.stack(comps, axis=0)
        cols = np.isfinite(stack).any(axis=0)
        surrogate[best[cols]] = np.nanmean(stack[:, cols], axis=0)
    return times, surrogate, run_ids


def derived_respiration_features(ecg: dict[str, np.ndarray], pleth: dict[str, np.ndarray], resp: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    out: dict[str, float] = {}
    ecg_times = np.asarray(ecg.get("morphology_peak_indices_for_detector", ecg.get("morphology_peak_indices", [])), dtype=np.float64) / config.sampling_rate_hz
    ecg_runs = np.asarray(ecg.get("morphology_run_ids", ecg.get("run_ids", [])), dtype=np.int64)
    ecg_rates = _derived_resp_from_surrogate(ecg_times, np.asarray(ecg.get("r_peak_amplitudes", []), dtype=np.float64), "ecg", config, ecg_runs)
    out.update(ecg_rates)
    if pleth:
        times, surrogate, run_ids = _pleth_run_local_resp_surrogate(pleth, config)
        out.update(_derived_resp_from_surrogate(times, surrogate, "pleth", config, run_ids))
    rates = []
    labels = []
    if resp:
        resp_idx = _longest_resp_cycle_run(
            resp,
            min_cycles=config.min_resp_rrv_cycles,
            min_duration_s=config.min_resp_rrv_coverage_seconds,
        )
        if resp_idx.size >= config.min_resp_rrv_cycles:
            resp_lengths = np.asarray(resp.get("length_s", []), dtype=np.float64)[resp_idx]
            resp_times = np.asarray(resp.get("end_s", []), dtype=np.float64)[resp_idx]
            if np.ptp(resp_times) >= config.min_resp_rrv_coverage_seconds and np.isfinite(resp_lengths).all():
                direct = float(60.0 / np.median(resp_lengths))
                rates.append(direct)
                labels.append("resp")
    if np.isfinite(out.get("ecg_derived_resp_rate_5m", np.nan)):
        rates.append(out["ecg_derived_resp_rate_5m"])
        labels.append("ecg")
    if np.isfinite(out.get("pleth_derived_resp_rate_5m", np.nan)):
        rates.append(out["pleth_derived_resp_rate_5m"])
        labels.append("pleth")
    lookup = dict(zip(labels, rates))
    if "resp" in lookup and "ecg" in lookup:
        out["resp_ecg_rate_disagreement_bpm"] = abs(lookup["resp"] - lookup["ecg"])
    if "resp" in lookup and "pleth" in lookup:
        out["resp_pleth_rate_disagreement_bpm"] = abs(lookup["resp"] - lookup["pleth"])
    if "ecg" in lookup and "pleth" in lookup:
        out["ecg_pleth_derived_resp_rate_disagreement_bpm"] = abs(lookup["ecg"] - lookup["pleth"])
    if len(rates) >= 2:
        diffs = [abs(rates[i] - rates[j]) for i in range(len(rates)) for j in range(i + 1, len(rates))]
        out["resp_cross_modal_rate_agreement"] = float(1.0 / (1.0 + np.median(diffs) / 10.0))
    return out


def respiratory_pattern_features(resp_signal: np.ndarray, resp: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    out: dict[str, float] = {}
    if not resp:
        resp = detect_resp_cycles(resp_signal, config.sampling_rate_hz, config)
    if not _validate_resp_dict(resp, required=("end_s", "length_s", "amplitude", "run_id")):
        return {}
    lengths = np.asarray(resp.get("length_s", []), dtype=np.float64)
    amps = np.asarray(resp.get("amplitude", []), dtype=np.float64)
    times = np.asarray(resp.get("end_s", []), dtype=np.float64)
    best_idx = _longest_resp_cycle_run(
        resp,
        min_cycles=config.min_resp_rrv_cycles,
        min_duration_s=config.min_resp_rrv_coverage_seconds,
    )
    if best_idx.size >= config.min_resp_rrv_cycles and np.ptp(times[best_idx]) >= config.min_resp_rrv_coverage_seconds:
        best_lengths = lengths[best_idx]
        best_amps = amps[best_idx]
        out["resp_cycle_interval_sampen_5m"] = sample_entropy(best_lengths, min_count=config.min_resp_rrv_cycles)
        out["resp_amplitude_envelope_cv_5m"] = _cv(best_amps, min_count=config.min_resp_rrv_cycles)
        med = float(np.median(best_amps)) if best_amps.size else np.nan
        mad = _v8_scale(best_amps)
        if med > 1e-8 and np.isfinite(mad):
            out["resp_sigh_count_5m"] = float(np.sum(best_amps > med + config.resp_sigh_mad_threshold * mad))
            out["resp_suppressed_amplitude_burden_5m"] = float(np.mean(best_amps < config.resp_suppressed_amplitude_fraction * med))
        if np.std(best_amps) > 1e-8:
            x = signal.detrend(best_amps - np.median(best_amps), type="linear")
            t = times[best_idx]
            freqs = np.linspace(0.001, 0.10, 256)
            power = signal.lombscargle(t - t[0], x, 2.0 * np.pi * freqs, normalize=True)
            total = float(np.trapezoid(np.maximum(power, 0.0), freqs))
            band = (freqs >= config.resp_periodic_modulation_low_hz) & (freqs <= config.resp_periodic_modulation_high_hz)
            if total > 1e-12 and np.any(band):
                out["resp_periodic_breathing_index_5m"] = float(np.trapezoid(np.maximum(power[band], 0.0), freqs[band]) / total)
    pauses = detect_resp_pause_durations(resp_signal, config.sampling_rate_hz, config, filtered=resp.get("filtered"), ordered_extrema=resp.get("ordered_extrema"))
    if pauses is not None:
        finite = np.isfinite(resp_signal)
        observable_s = float(np.sum(finite) / config.sampling_rate_hz)
        out["resp_pause_burden_5m"] = float(np.sum(pauses) / observable_s) if observable_s > 0 else _nan()
    return out


def _longest_resp_cycle_run(resp: dict[str, np.ndarray], min_cycles: int = 1, min_duration_s: float = 0.0) -> np.ndarray:
    lengths = np.asarray(resp.get("length_s", []), dtype=np.float64)
    times = np.asarray(resp.get("end_s", []), dtype=np.float64)
    run_ids = np.asarray(resp.get("run_id", []), dtype=np.int64)
    if not (run_ids.size == lengths.size == times.size):
        return np.asarray([], dtype=np.int64)
    valid = np.isfinite(lengths) & (lengths > 0) & np.isfinite(times) & (run_ids >= 0)
    return select_best_continuous_event_run(times, valid, run_ids, min_count=min_cycles, min_duration_s=min_duration_s)


def respiratory_rate_variability_features(resp_history: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    if not resp_history or not _validate_resp_dict(resp_history, required=("end_s", "length_s", "run_id")):
        return {}
    idx = _longest_resp_cycle_run(
        resp_history,
        min_cycles=config.min_resp_rrv_cycles,
        min_duration_s=config.min_resp_rrv_coverage_seconds,
    )
    if idx.size < config.min_resp_rrv_cycles:
        return {}
    lengths = np.asarray(resp_history.get("length_s", []), dtype=np.float64)[idx]
    times = np.asarray(resp_history.get("end_s", []), dtype=np.float64)[idx]
    if times.size < config.min_resp_rrv_cycles or np.ptp(times) < config.min_resp_rrv_coverage_seconds:
        return {}
    out: dict[str, float] = {
        "resp_interval_sd_5m": safe_nanstat(np.std, lengths),
    }
    if lengths.size >= 3:
        diffs = np.diff(lengths)
        out["resp_interval_rmssd_5m"] = float(np.sqrt(np.mean(diffs * diffs)))
        out["resp_interval_sdsd_5m"] = safe_nanstat(np.std, diffs)
        sd1, sd2, ratio = poincare(lengths, min_pairs=max(3, min(config.min_resp_rrv_cycles - 1, lengths.size - 1)))
        out["resp_poincare_sd1_5m"] = sd1
        out["resp_poincare_sd2_5m"] = sd2
        out["resp_poincare_sd1_sd2_ratio_5m"] = ratio
    if (
        times.size >= config.min_resp_rrv_cycles
        and np.ptp(times) >= config.min_resp_rrv_coverage_seconds
        and np.std(lengths) > 1e-8
    ):
        x = signal.detrend(lengths - np.median(lengths), type="linear")
        dt = np.diff(times)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            return out
        effective_high = min(float(config.resp_rrv_frequency_high_hz), 0.5 / float(np.median(dt)))
        if effective_high <= config.resp_rrv_frequency_low_hz:
            return out
        freqs = np.linspace(config.resp_rrv_frequency_low_hz, effective_high, int(config.resp_rrv_frequency_bins))
        if freqs.size < 8:
            return out
        power = np.maximum(signal.lombscargle(times - times[0], x, 2.0 * np.pi * freqs, normalize=True), 0.0)
        total = float(np.trapezoid(power, freqs))
        if total > 1e-12:
            peak_idx = int(np.argmax(power))
            peak_freq = float(freqs[peak_idx])
            peak_band = (freqs >= peak_freq - config.resp_rrv_peak_half_width_hz) & (freqs <= peak_freq + config.resp_rrv_peak_half_width_hz)
            out["resp_interval_spectral_peak_hz_5m"] = peak_freq
            out["resp_interval_spectral_peak_power_fraction_5m"] = float(np.trapezoid(power[peak_band], freqs[peak_band]) / total) if np.any(peak_band) else _nan()
            p = power / float(np.sum(power))
            p = p[p > 0]
            out["resp_interval_spectral_entropy_5m"] = float(-np.sum(p * np.log(p)) / np.log(power.size)) if power.size > 1 else _nan()
    return out


def _energy_ratio(x: np.ndarray, fs: int, lo: float, hi: float, total_hi: float) -> float:
    arr = np.asarray(x, dtype=np.float64)
    den = 0.0
    num = 0.0
    usable = 0
    min_len = max(2 * fs, 1)
    for start, end in finite_runs(np.isfinite(arr)):
        seg = arr[start:end]
        if seg.size < min_len or np.std(seg) < 1e-8:
            continue
        freqs, pxx = signal.welch(seg - np.median(seg), fs=fs, nperseg=min(seg.size, 4 * fs))
        total = (freqs >= 0.5) & (freqs <= total_hi)
        high = (freqs >= lo) & (freqs <= hi)
        den += (float(np.trapezoid(pxx[total], freqs[total])) if np.any(total) else 0.0) * seg.size
        num += (float(np.trapezoid(pxx[high], freqs[high])) if np.any(high) else 0.0) * seg.size
        usable += seg.size
    if usable < min_len or den <= 1e-12:
        return _nan()
    return num / den


def _baseline_jump_count(x: np.ndarray, fs: int, config: V8ExtractionConfig) -> float:
    arr = np.asarray(x, dtype=np.float64)
    win = max(int(5 * fs), 1)
    rows: list[tuple[int, float]] = []
    for window_idx, start in enumerate(range(0, arr.size, win)):
        seg = arr[start : start + win]
        if seg.size == win and np.mean(np.isfinite(seg)) > 0.95:
            rows.append((window_idx, float(np.nanmedian(seg))))
    if len(rows) < 2:
        return _nan()
    diffs = []
    for (prev_idx, prev_med), (cur_idx, cur_med) in zip(rows[:-1], rows[1:]):
        if cur_idx == prev_idx + 1:
            diffs.append(abs(cur_med - prev_med))
    if len(diffs) < 2:
        return _nan()
    diffs = np.asarray(diffs, dtype=np.float64)
    return _outlier_count_positive(diffs, config.quality_step_change_mad_threshold)


def _pleth_plateau_quantization(pleth_signal: np.ndarray, fs: int) -> tuple[float, float]:
    arr = np.asarray(pleth_signal, dtype=np.float64)
    plateau_num = 0
    plateau_den = 0
    quant_counts: dict[float, int] = {}
    quant_den = 0
    for start, end in finite_runs(np.isfinite(arr)):
        seg = arr[start:end]
        if seg.size < fs:
            continue
        scale = _v8_scale(seg)
        if not np.isfinite(scale) or scale <= 0:
            continue
        diffs = np.abs(np.diff(seg))
        if diffs.size == 0:
            continue
        tol = max(1e-6 * scale, 1e-12)
        near_equal = diffs <= tol
        plateau_num += int(np.sum(near_equal))
        plateau_den += int(diffs.size)
        nd = diffs / scale
        nonzero = nd[nd > tol / scale]
        if nonzero.size >= 10:
            rounded = np.round(nonzero, 3)
            vals, counts = np.unique(rounded, return_counts=True)
            for val, count in zip(vals.tolist(), counts.tolist()):
                quant_counts[float(val)] = quant_counts.get(float(val), 0) + int(count)
            quant_den += int(nonzero.size)
    plateau = float(plateau_num / plateau_den) if plateau_den else _nan()
    quant = float(max(quant_counts.values()) / quant_den) if quant_den and quant_counts else _nan()
    return plateau, quant


def waveform_quality_features(ecg_signal: np.ndarray, abp_signal: np.ndarray, pleth_signal: np.ndarray, resp_signal: np.ndarray, abp: dict[str, np.ndarray], pleth: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    fs = config.sampling_rate_hz
    out: dict[str, float] = {
        "abp_high_frequency_energy_ratio": _energy_ratio(abp_signal, fs, config.quality_high_frequency_low_hz, config.quality_total_high_hz, config.quality_total_high_hz),
        "ecg_baseline_jump_count": _baseline_jump_count(ecg_signal, fs, config),
        "resp_baseline_jump_count": _baseline_jump_count(resp_signal, fs, config),
    }
    if abp:
        valid = np.asarray(abp["valid"], dtype=bool)
        segments = np.asarray(abp.get("raw_segments", []), dtype=object)
        local_peak = np.rint(np.asarray(abp.get("peak_idx", []), dtype=np.float64) - np.asarray(abp.get("start_idx", []), dtype=np.float64)).astype(int)
        flat = []
        zc = []
        for idx in np.flatnonzero(valid).tolist():
            if idx >= segments.size:
                continue
            seg = np.asarray(segments[idx], dtype=np.float64)
            if seg.size < 5 or not np.isfinite(seg).all():
                continue
            peak_i = int(local_peak[idx]) if idx < local_peak.size and 0 < local_peak[idx] < seg.size - 1 else int(np.argmax(seg))
            peak_val = float(seg[peak_i])
            amp = float(peak_val - seg[0])
            if amp <= 1e-8:
                continue
            dwell = float(np.mean(seg >= peak_val - config.flat_top_atol_fraction * amp))
            flat.append(dwell >= config.flat_top_min_duration_fraction)
            post = seg[peak_i:]
            if post.size >= 5:
                d = np.gradient(post)
                zc.append(float(np.sum(np.diff(np.signbit(d)) != 0)))
        if flat:
            out["abp_flat_top_beat_fraction"] = float(np.mean(flat))
        hf = out["abp_high_frequency_energy_ratio"]
        if zc:
            out["abp_ringing_index"] = float(np.median(zc) / 10.0 + (hf if np.isfinite(hf) else 0.0))
        mapv = np.asarray(abp.get("mean", []), dtype=np.float64)
        pp = np.asarray(abp.get("peak", []), dtype=np.float64) - np.asarray(abp.get("foot", []), dtype=np.float64)
        diffs = np.abs(_adjacent_pulse_diffs(abp, mapv))
        if diffs.size >= 5:
            out["abp_step_change_count"] = _outlier_count_positive(diffs, config.quality_step_change_mad_threshold)
        ratios = _adjacent_pulse_ratios(abp, pp)
        if ratios.size >= 4:
            out["abp_scale_change_count"] = float(np.sum(ratios > config.quality_scale_change_ratio_threshold))
    plateau, quant = _pleth_plateau_quantization(pleth_signal, fs)
    out["pleth_plateau_fraction"] = plateau
    out["pleth_quantization_index"] = quant
    if pleth:
        amp = np.asarray(pleth.get("amplitude", []), dtype=np.float64)
        ratios = _adjacent_pulse_ratios(pleth, amp)
        if ratios.size >= 4:
            out["pleth_scale_change_count"] = float(np.sum(ratios > config.quality_scale_change_ratio_threshold))
    return out


def _max_consecutive_true_run_by_source(deficit: np.ndarray, source_run: np.ndarray) -> int:
    deficit = np.asarray(deficit, dtype=bool)
    runs = np.asarray(source_run, dtype=np.int64)
    if deficit.size == 0 or runs.size != deficit.size:
        return 0
    idx = np.flatnonzero(deficit & (runs >= 0))
    if idx.size == 0:
        return 0
    split = np.where((np.diff(idx) != 1) | (runs[idx[1:]] != runs[idx[:-1]]))[0] + 1
    return int(max(chunk.size for chunk in np.split(idx, split)))


def pulse_deficit_features(ecg: dict[str, np.ndarray], target: dict[str, np.ndarray], prefix: str, bounds_ms: tuple[float, float], config: V8ExtractionConfig, pairs: np.ndarray | None = None, source_observable: np.ndarray | None = None) -> dict[str, float]:
    if not target:
        return {}
    source_times = np.asarray(ecg.get("peaks", []), dtype=np.float64) / config.sampling_rate_hz
    source_run = np.asarray(ecg.get("run_ids", []), dtype=np.int64)
    target_times = np.asarray(target.get("foot_time_s", []), dtype=np.float64)
    target_valid = np.asarray(target.get("valid", []), dtype=bool)
    target_run = np.asarray(target.get("run_id", np.full(target_times.size, -1)), dtype=np.int64)
    raw = np.asarray(target.get("raw_signal", []), dtype=np.float64)
    if source_run.size != source_times.size or target_valid.size != target_times.size or target_run.size != target_times.size:
        return {}
    source_valid = source_run >= 0
    target_valid = target_valid & (target_run >= 0)
    target_observability = _observability_from_pulse_dict(target)
    if target_observability is None and raw.size:
        target_observability = _raw_observability_metadata(raw)
    matched_pairs = pairs if pairs is not None else _observable_forward_pairs(
        source_times,
        target_times,
        bounds_ms,
        source_valid,
        target_valid,
        raw,
        target_run,
        config.sampling_rate_hz,
        target_observability,
    )
    matched_source = np.zeros(source_times.size, dtype=bool)
    if matched_pairs.size:
        matched_source[matched_pairs[:, 0].astype(np.int64)] = True
    if source_observable is None or np.asarray(source_observable, dtype=bool).size != source_times.size:
        lo, hi = bounds_ms[0] / 1000.0, bounds_ms[1] / 1000.0
        source_observable_arr = np.zeros(source_times.size, dtype=bool)
        for idx, src in enumerate(source_times.tolist()):
            if idx >= source_run.size or source_run[idx] < 0:
                continue
            start = max(0, int(np.floor((src + lo) * config.sampling_rate_hz)))
            end = min(raw.size, int(np.ceil((src + hi) * config.sampling_rate_hz)))
            _, observed = _interval_observed_from_metadata(start, end, raw.size, target_observability)
            source_observable_arr[idx] = observed
    else:
        source_observable_arr = np.asarray(source_observable, dtype=bool)
    eligible_mask = source_valid & source_observable_arr
    deficit = eligible_mask & ~matched_source
    eligible = int(np.sum(eligible_mask))
    unmatched = int(np.sum(deficit))
    max_run = _max_consecutive_true_run_by_source(deficit, source_run)
    if eligible < config.min_timing_pairs:
        return {}
    return {
        f"{prefix}_pulse_deficit_fraction": float(unmatched / eligible),
        f"{prefix}_max_pulse_deficit_run": float(max_run),
    }


def systolic_time_features(abp: dict[str, np.ndarray], config: V8ExtractionConfig) -> dict[str, float]:
    if not abp:
        return {}
    from .morphology import detect_dicrotic_notch

    fs = config.sampling_rate_hz
    valid = np.asarray(abp.get("valid", []), dtype=bool)
    morph_segments = np.asarray(abp.get("morphology_segments", []), dtype=object)
    raw_segments = np.asarray(abp.get("raw_segments", []), dtype=object)
    width = np.asarray(abp.get("width_s", []), dtype=np.float64)
    pp = np.asarray(abp.get("peak", []), dtype=np.float64) - np.asarray(abp.get("foot", []), dtype=np.float64)
    local_peak = np.rint(np.asarray(abp.get("peak_idx", []), dtype=np.float64) - np.asarray(abp.get("start_idx", []), dtype=np.float64)).astype(int)
    lvets: list[float] = []
    corrected: list[float] = []
    fractions: list[float] = []
    ndpdt: list[float] = []
    for idx in np.flatnonzero(valid).tolist():
        if idx >= morph_segments.size:
            continue
        seg = np.asarray(morph_segments[idx], dtype=np.float64)
        raw = np.asarray(raw_segments[idx], dtype=np.float64) if idx < raw_segments.size else seg
        peak_i = int(local_peak[idx]) if idx < local_peak.size else -1
        if seg.size < 5 or raw.size != seg.size or not np.isfinite(seg).all() or not np.isfinite(raw).all():
            continue
        if peak_i <= 0 or peak_i >= seg.size - 1:
            continue
        if idx < pp.size and pp[idx] > 1e-8:
            ndpdt.append(float(np.max(np.gradient(seg[: peak_i + 1]) * fs) / pp[idx]))
        notch = detect_dicrotic_notch(
            seg,
            peak_i,
            fs,
            score_separation=config.abp_notch_candidate_score_separation,
            min_candidate_score=config.abp_notch_min_candidate_score,
        )
        if notch is None:
            continue
        notch_idx, _, _ = notch
        lvet = float(notch_idx / fs)
        lvets.append(lvet)
        if width[idx] > 1e-8:
            fractions.append(lvet / width[idx])
            corrected.append(lvet / np.sqrt(width[idx]))
    out: dict[str, float] = {}
    if len(lvets) >= config.min_abp_morphology_beats:
        out.update({
            "abp_lvet_median_s": safe_nanstat(np.median, np.asarray(lvets)),
            "abp_lvet_iqr_s": nan_iqr(np.asarray(lvets)),
            "abp_lvet_hr_corrected": safe_nanstat(np.median, np.asarray(corrected)),
            "abp_systolic_time_fraction": safe_nanstat(np.median, np.asarray(fractions)),
        })
    if len(ndpdt) >= config.min_abp_morphology_beats:
        out["abp_normalized_dpdt_max_median_per_s"] = safe_nanstat(np.median, np.asarray(ndpdt))
    return out


_V8_SCHEMA_CACHE: tuple[list[str], dict[str, int]] | None = None


def _v8_feature_schema() -> tuple[list[str], dict[str, int]]:
    global _V8_SCHEMA_CACHE
    if _V8_SCHEMA_CACHE is None:
        names = feature_names()
        _V8_SCHEMA_CACHE = (names, {name: idx for idx, name in enumerate(names)})
    return _V8_SCHEMA_CACHE


def _put_row(row: np.ndarray, mask: np.ndarray, values: dict[str, float], names: list[str], name_to_index: dict[str, int] | None = None) -> None:
    mapping = name_to_index if name_to_index is not None else {name: idx for idx, name in enumerate(names)}
    row[:] = np.nan
    mask[:] = False
    for name, value in values.items():
        idx = mapping.get(name)
        if idx is None:
            raise ValueError(f"v8 feature calculation returned unknown feature names: {[name]}")
        row[idx] = value
        mask[idx] = np.isfinite(value)


def _slice_object_array(values: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.asarray(list(np.asarray(values, dtype=object)[start:end]), dtype=object)


def slice_ecg_events(cache: dict[str, np.ndarray], start_sample: int, end_sample: int, config: V8ExtractionConfig) -> dict[str, np.ndarray]:
    fs = config.sampling_rate_hz
    start_s = start_sample / fs
    peaks = np.asarray(cache.get("peaks", []), dtype=np.int64)
    left = int(np.searchsorted(peaks, start_sample, side="left"))
    right = int(np.searchsorted(peaks, end_sample, side="left"))
    out: dict[str, np.ndarray] = {}
    peak_keys = {
        "peaks",
        "detector_peak_indices",
        "run_ids",
        "morphology_peak_indices",
        "morphology_peak_indices_for_detector",
        "morphology_run_ids",
        "r_peak_amplitudes",
        "qrs_segments",
        "qrs_segment_valid",
        "qrs_template_correlations",
    }
    index_keys = {"peaks", "detector_peak_indices", "morphology_peak_indices", "morphology_peak_indices_for_detector"}
    for key in peak_keys:
        if key not in cache:
            continue
        arr = np.asarray(cache[key], dtype=object if key == "qrs_segments" else None)
        if arr.size != peaks.size:
            continue
        sliced = _slice_object_array(arr, left, right) if key == "qrs_segments" else np.asarray(arr[left:right]).copy()
        if key in index_keys:
            sliced = sliced.astype(np.int64, copy=False) - int(start_sample)
        out[key] = sliced
    if "qrs_segment_valid" in out and "morphology_peak_indices" in cache:
        qrs_left = int(round(0.12 * fs))
        qrs_right = int(round(0.18 * fs))
        mapped = np.asarray(cache.get("morphology_peak_indices", []), dtype=np.int64)[left:right]
        inside = (mapped - qrs_left >= start_sample) & (mapped + qrs_right < end_sample)
        out["qrs_segment_valid"] = np.asarray(out["qrs_segment_valid"], dtype=bool) & inside
    rr_start = left
    rr_end = max(left, right - 1)
    rr_keys = {"rr_s", "rr_times_s", "valid_rr", "same_run_rr"}
    for key in rr_keys:
        if key not in cache:
            continue
        arr = np.asarray(cache[key])
        sliced = np.asarray(arr[rr_start:rr_end]).copy()
        if key == "rr_times_s":
            sliced = sliced.astype(np.float64, copy=False) - start_s
        out[key] = sliced
    for key in rr_keys:
        out.setdefault(key, np.asarray([], dtype=bool if key in {"valid_rr", "same_run_rr"} else np.float64))
    for key in ("peaks", "detector_peak_indices", "run_ids"):
        out.setdefault(key, np.asarray([], dtype=np.int64))
    return out


def slice_pulse_events(cache: dict[str, np.ndarray], signal_window: np.ndarray, start_sample: int, end_sample: int, config: V8ExtractionConfig) -> dict[str, np.ndarray]:
    peak_idx = np.asarray(cache.get("peak_idx", []), dtype=np.int64)
    if peak_idx.size == 0:
        return {}
    left = int(np.searchsorted(peak_idx, start_sample, side="left"))
    right = int(np.searchsorted(peak_idx, end_sample, side="left"))
    if right <= left:
        return {}
    n_events = peak_idx.size
    out: dict[str, np.ndarray] = {}
    index_keys = {"peak_idx", "start_idx", "end_idx"}
    time_keys = {"time_s", "foot_time_s"}
    start_s = start_sample / config.sampling_rate_hz
    for key, value in cache.items():
        if key == "raw_signal" or key.startswith("_"):
            continue
        arr = np.asarray(value, dtype=object if key.endswith("segments") else None)
        if arr.ndim == 0 or arr.shape[0] != n_events:
            continue
        sliced = _slice_object_array(arr, left, right) if key.endswith("segments") else np.asarray(arr[left:right]).copy()
        if key in index_keys:
            sliced = sliced.astype(np.float64 if np.issubdtype(sliced.dtype, np.floating) else np.int64, copy=False) - int(start_sample)
        elif key in time_keys:
            sliced = sliced.astype(np.float64, copy=False) - start_s
        out[key] = sliced
    local_start = np.asarray(cache.get("start_idx", []), dtype=np.float64)[left:right]
    local_end = np.asarray(cache.get("end_idx", []), dtype=np.float64)[left:right]
    fully_inside = (local_start >= start_sample) & (local_end < end_sample)
    if "valid" in out:
        out["valid"] = np.asarray(out["valid"], dtype=bool) & fully_inside
    out["raw_signal"] = np.asarray(signal_window, dtype=np.float64)
    full_run_map = np.asarray(cache.get("_raw_run_id_by_sample", []), dtype=np.int64)
    local_run_map = full_run_map[start_sample:end_sample] if full_run_map.size else None
    out.update(_raw_observability_metadata(out["raw_signal"], local_run_map))
    return out


def slice_resp_events(cache: dict[str, np.ndarray], signal_window: np.ndarray, start_sample: int, end_sample: int, config: V8ExtractionConfig) -> dict[str, np.ndarray]:
    starts = np.asarray(cache.get("start_s", []), dtype=np.float64)
    ends = np.asarray(cache.get("end_s", []), dtype=np.float64)
    if starts.size == 0 or ends.size != starts.size:
        filtered = np.asarray(cache.get("filtered", []), dtype=np.float64)
        return {
            "start_s": np.asarray([], dtype=np.float64),
            "peak_s": np.asarray([], dtype=np.float64),
            "end_s": np.asarray([], dtype=np.float64),
            "length_s": np.asarray([], dtype=np.float64),
            "amplitude": np.asarray([], dtype=np.float64),
            "run_id": np.asarray([], dtype=np.int64),
            "filtered": filtered[start_sample:end_sample] if filtered.size else np.asarray(signal_window, dtype=np.float64),
            "ordered_extrema": np.asarray([], dtype=np.float64),
        }
    fs = config.sampling_rate_hz
    start_s = start_sample / fs
    end_s = end_sample / fs
    left = int(np.searchsorted(ends, start_s, side="left"))
    right = int(np.searchsorted(starts, end_s, side="left"))
    keep = np.arange(left, right, dtype=np.int64)
    keep = keep[(starts[keep] >= start_s) & (ends[keep] <= end_s)]
    out: dict[str, np.ndarray] = {}
    event_keys = {"start_s", "peak_s", "end_s", "length_s", "amplitude", "run_id"}
    for key in event_keys:
        if key not in cache:
            continue
        arr = np.asarray(cache[key])
        sliced = np.asarray(arr[keep]).copy()
        if key in {"start_s", "peak_s", "end_s"}:
            sliced = sliced.astype(np.float64, copy=False) - start_s
        out[key] = sliced
    filtered = np.asarray(cache.get("filtered", []), dtype=np.float64)
    out["filtered"] = filtered[start_sample:end_sample] if filtered.size else np.asarray(signal_window, dtype=np.float64)
    extrema = np.asarray(cache.get("ordered_extrema", []), dtype=np.float64)
    if extrema.size:
        ex = extrema[(extrema[:, 0] >= start_s) & (extrema[:, 0] < end_s)].copy()
        ex[:, 0] -= start_s
        out["ordered_extrema"] = ex
    else:
        out["ordered_extrema"] = np.asarray([], dtype=np.float64)
    return out


def _finalize_feature_row(features: dict[str, float], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(len(names), np.nan, dtype=np.float32)
    mask = np.zeros(len(names), dtype=bool)
    _put_row(values, mask, features, names, _v8_feature_schema()[1])
    return values, mask


def extract_v8_minute_component(
    waveform_1m: np.ndarray,
    config: V8ExtractionConfig = DEFAULT_V8_EXTRACTION_CONFIG,
) -> dict[str, float]:
    minute = np.asarray(waveform_1m, dtype=np.float64)
    expected_shape = (len(config.channel_order), config.feature_window_samples)
    if minute.shape != expected_shape:
        raise ValueError(f"Expected one-minute waveform shape {expected_shape}, got {minute.shape}")
    ecg_min = detect_ecg_events(minute[0], config.sampling_rate_hz, config, include_qrs_template=False, include_morphology=False)
    need_observability = config.enable_cross_signal_timing or config.enable_pulse_deficit_features
    abp = detect_pulses(minute[1], config.sampling_rate_hz, "abp", config, include_observability=need_observability, include_raw_signal=need_observability)
    pleth = detect_pulses(minute[2], config.sampling_rate_hz, "pleth", config, include_observability=need_observability, include_raw_signal=need_observability)
    resp = detect_resp_cycles(minute[3], config.sampling_rate_hz, config)
    features: dict[str, float] = {}
    features.update(_resp_variation_features("abp", abp, resp, config))
    features.update(_resp_variation_features("pleth", pleth, resp, config))
    features.update(pleth_shape_features(pleth, config))
    features.update(abp_lability_features(abp, config))
    features.update(waveform_quality_features(minute[0], minute[1], minute[2], minute[3], abp, pleth, config))
    features.update(burden_instability_features(ecg_min, abp, pleth, resp, minute[3], config, 0.0, 60.0))
    features.update(coupling_features(ecg_min, abp, pleth, resp, 0.0, 60.0, config))
    pair_cache = build_cross_signal_pair_cache(ecg_min, abp, pleth, config) if (config.enable_cross_signal_timing or config.enable_pulse_deficit_features) else None
    if config.enable_cross_signal_timing:
        features.update(timing_features(ecg_min, abp, pleth, config, pair_cache=pair_cache))
    if config.enable_pulse_deficit_features:
        features.update(pulse_deficit_features(ecg_min, abp, "ecg_abp", config.ecg_abp_pat_bounds_ms, config, pairs=(pair_cache or {}).get("ecg_abp"), source_observable=(pair_cache or {}).get("ecg_abp_source_observable")))
        features.update(pulse_deficit_features(ecg_min, pleth, "ecg_pleth", config.ecg_pleth_pat_bounds_ms, config, pairs=(pair_cache or {}).get("ecg_pleth"), source_observable=(pair_cache or {}).get("ecg_pleth_source_observable")))
    if config.enable_pleth_fiducials:
        features.update(pleth_fiducial_features(pleth, config))
    if config.enable_pleth_derivative_fiducials:
        features.update(pleth_derivative_fiducial_features(pleth, config))
    if config.enable_abp_advanced_morphology:
        features.update(advanced_abp_morphology_from_beats(abp, config.sampling_rate_hz, config=config))
    if config.enable_systolic_time_features:
        features.update(systolic_time_features(abp, config))
    return features


def extract_v8_history_component(
    waveform_5m: np.ndarray,
    config: V8ExtractionConfig = DEFAULT_V8_EXTRACTION_CONFIG,
) -> dict[str, float]:
    history = np.asarray(waveform_5m, dtype=np.float64)
    expected_shape = (len(config.channel_order), config.rolling_history_samples)
    if history.shape != expected_shape:
        raise ValueError(f"Expected 5-minute waveform shape {expected_shape}, got {history.shape}")
    ecg_history = detect_ecg_events(history[0], config.sampling_rate_hz, config, include_qrs_template=False, include_morphology=True)
    need_pleth_history_segments = config.enable_pleth_fiducials or config.enable_pleth_derivative_fiducials
    abp_history = detect_pulses(history[1], config.sampling_rate_hz, "abp", config, include_segments=False, include_observability=False, include_raw_signal=False)
    pleth_history = detect_pulses(history[2], config.sampling_rate_hz, "pleth", config, include_segments=need_pleth_history_segments, include_observability=False, include_raw_signal=False)
    resp_history = detect_resp_cycles(history[3], config.sampling_rate_hz, config)
    features: dict[str, float] = {}
    features.update(baroreflex_features(ecg_history, abp_history, config))
    features.update(abp_nonlinear_dynamics_features(abp_history, config))
    features.update(morphology_dynamics_features(abp_history, pleth_history, config))
    features.update(nonlinear_hrv_features(ecg_history, 0.0, float(config.rolling_history_seconds), config))
    history_end_s = float(config.rolling_history_seconds)
    rhythm_start_s = max(0.0, history_end_s - float(config.rolling_rhythm_history_seconds))
    features.update(rhythm_features(ecg_history, rhythm_start_s, history_end_s, config))
    features.update(derived_respiration_features(ecg_history, pleth_history, resp_history, config))
    features.update(respiratory_pattern_features(history[3], resp_history, config))
    features.update(respiratory_rate_variability_features(resp_history, config))
    if config.enable_pleth_fiducials or config.enable_pleth_derivative_fiducials:
        features.update(pleth_experimental_morphology_dynamics_features(pleth_history, config))
    return features


def _assemble_v8_sequence_from_component_getters(
    config: V8ExtractionConfig,
    minute_component,
    history_component,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names, name_to_index = _v8_feature_schema()
    values = np.full((config.n_feature_windows, len(names)), np.nan, dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    for minute_idx in range(config.n_feature_windows):
        features = dict(minute_component(minute_idx))
        end = (minute_idx + 1) * config.feature_window_samples
        if (end / config.sampling_rate_hz) >= config.rolling_history_seconds:
            features.update(history_component(minute_idx))
        _put_row(values[minute_idx], mask[minute_idx], features, names, name_to_index)
    return values, mask, names


def _extract_v8_feature_sequence_from_event_source(
    waveform: np.ndarray,
    config: V8ExtractionConfig,
    ecg_full: dict[str, np.ndarray],
    abp_full: dict[str, np.ndarray],
    pleth_full: dict[str, np.ndarray],
    resp_full: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    def minute_component(minute_idx: int) -> dict[str, float]:
        start = minute_idx * config.feature_window_samples
        end = start + config.feature_window_samples
        minute = waveform[:, start:end]
        ecg_min = slice_ecg_events(ecg_full, start, end, config)
        abp = slice_pulse_events(abp_full, minute[1], start, end, config)
        pleth = slice_pulse_events(pleth_full, minute[2], start, end, config)
        resp = slice_resp_events(resp_full, minute[3], start, end, config)
        features: dict[str, float] = {}
        features.update(_resp_variation_features("abp", abp, resp, config))
        features.update(_resp_variation_features("pleth", pleth, resp, config))
        features.update(pleth_shape_features(pleth, config))
        features.update(abp_lability_features(abp, config))
        features.update(waveform_quality_features(minute[0], minute[1], minute[2], minute[3], abp, pleth, config))
        features.update(burden_instability_features(ecg_min, abp, pleth, resp, minute[3], config, 0.0, 60.0))
        features.update(coupling_features(ecg_min, abp, pleth, resp, 0.0, 60.0, config))
        pair_cache = build_cross_signal_pair_cache(ecg_min, abp, pleth, config) if (config.enable_cross_signal_timing or config.enable_pulse_deficit_features) else None
        if config.enable_cross_signal_timing:
            features.update(timing_features(ecg_min, abp, pleth, config, pair_cache=pair_cache))
        if config.enable_pulse_deficit_features:
            features.update(pulse_deficit_features(ecg_min, abp, "ecg_abp", config.ecg_abp_pat_bounds_ms, config, pairs=(pair_cache or {}).get("ecg_abp"), source_observable=(pair_cache or {}).get("ecg_abp_source_observable")))
            features.update(pulse_deficit_features(ecg_min, pleth, "ecg_pleth", config.ecg_pleth_pat_bounds_ms, config, pairs=(pair_cache or {}).get("ecg_pleth"), source_observable=(pair_cache or {}).get("ecg_pleth_source_observable")))
        if config.enable_pleth_fiducials:
            features.update(pleth_fiducial_features(pleth, config))
        if config.enable_pleth_derivative_fiducials:
            features.update(pleth_derivative_fiducial_features(pleth, config))
        if config.enable_abp_advanced_morphology:
            features.update(advanced_abp_morphology_from_beats(abp, config.sampling_rate_hz, config=config))
        if config.enable_systolic_time_features:
            features.update(systolic_time_features(abp, config))
        return features

    def history_component(minute_idx: int) -> dict[str, float]:
        end = (minute_idx + 1) * config.feature_window_samples
        history_start = end - config.rolling_history_samples
        history = waveform[:, history_start:end]
        ecg_history = slice_ecg_events(ecg_full, history_start, end, config)
        abp_history = slice_pulse_events(abp_full, history[1], history_start, end, config)
        pleth_history = slice_pulse_events(pleth_full, history[2], history_start, end, config)
        resp_history = slice_resp_events(resp_full, history[3], history_start, end, config)
        features: dict[str, float] = {}
        features.update(baroreflex_features(ecg_history, abp_history, config))
        features.update(abp_nonlinear_dynamics_features(abp_history, config))
        features.update(morphology_dynamics_features(abp_history, pleth_history, config))
        features.update(nonlinear_hrv_features(ecg_history, 0.0, float(config.rolling_history_seconds), config))
        history_end_s = float(config.rolling_history_seconds)
        rhythm_start_s = max(0.0, history_end_s - float(config.rolling_rhythm_history_seconds))
        features.update(rhythm_features(ecg_history, rhythm_start_s, history_end_s, config))
        features.update(derived_respiration_features(ecg_history, pleth_history, resp_history, config))
        features.update(respiratory_pattern_features(history[3], resp_history, config))
        features.update(respiratory_rate_variability_features(resp_history, config))
        if config.enable_pleth_fiducials or config.enable_pleth_derivative_fiducials:
            features.update(pleth_experimental_morphology_dynamics_features(pleth_history, config))
        return features

    return _assemble_v8_sequence_from_component_getters(config, minute_component, history_component)


def extract_v8_feature_sequence_reference(
    waveform: np.ndarray,
    config: V8ExtractionConfig = DEFAULT_V8_EXTRACTION_CONFIG,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    waveform = np.asarray(waveform, dtype=np.float64)
    expected_shape = (len(config.channel_order), config.input_samples)
    if waveform.shape != expected_shape:
        raise ValueError(f"Expected waveform shape {expected_shape}, got {waveform.shape}")
    if tuple(config.channel_order) != ("II", "ABP", "PLETH", "RESP"):
        raise ValueError(f"v8 currently requires channel_order ('II', 'ABP', 'PLETH', 'RESP'), got {config.channel_order}")

    def minute_component(minute_idx: int) -> dict[str, float]:
        start = minute_idx * config.feature_window_samples
        end = start + config.feature_window_samples
        return extract_v8_minute_component(waveform[:, start:end], config)

    def history_component(minute_idx: int) -> dict[str, float]:
        end = (minute_idx + 1) * config.feature_window_samples
        history_start = end - config.rolling_history_samples
        return extract_v8_history_component(waveform[:, history_start:end], config)

    return _assemble_v8_sequence_from_component_getters(config, minute_component, history_component)


def _component_config_key(config: V8ExtractionConfig) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), repr(value)) for key, value in config.to_dict().items()))


def _cached_component(
    cache: dict[tuple[object, tuple[tuple[str, str], ...], int, int], dict[str, float]] | None,
    stats: dict[str, int] | None,
    kind: str,
    key: tuple[object, tuple[tuple[str, str], ...], int, int],
    compute,
) -> dict[str, float]:
    if stats is not None:
        stats[f"{kind}_requested"] = int(stats.get(f"{kind}_requested", 0)) + 1
    if cache is not None and key in cache:
        if stats is not None:
            stats[f"{kind}_hits"] = int(stats.get(f"{kind}_hits", 0)) + 1
        return cache[key]
    result = compute()
    if cache is not None:
        cache[key] = result
    if stats is not None:
        stats[f"{kind}_computed"] = int(stats.get(f"{kind}_computed", 0)) + 1
    return result


def extract_v8_feature_sequence_reference_cached_components(
    waveform: np.ndarray,
    input_start_sample: int = 0,
    config: V8ExtractionConfig = DEFAULT_V8_EXTRACTION_CONFIG,
    minute_cache: dict[tuple[object, tuple[tuple[str, str], ...], int, int], dict[str, float]] | None = None,
    history_cache: dict[tuple[object, tuple[tuple[str, str], ...], int, int], dict[str, float]] | None = None,
    cache_key_prefix: object = None,
    cache_stats: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    waveform = np.asarray(waveform, dtype=np.float64)
    expected_shape = (len(config.channel_order), config.input_samples)
    if waveform.shape != expected_shape:
        raise ValueError(f"Expected waveform shape {expected_shape}, got {waveform.shape}")
    if tuple(config.channel_order) != ("II", "ABP", "PLETH", "RESP"):
        raise ValueError(f"v8 currently requires channel_order ('II', 'ABP', 'PLETH', 'RESP'), got {config.channel_order}")
    config_key = _component_config_key(config)
    input_start_sample = int(input_start_sample)

    def minute_component(minute_idx: int) -> dict[str, float]:
        local_start = minute_idx * config.feature_window_samples
        local_end = local_start + config.feature_window_samples
        abs_start = input_start_sample + local_start
        abs_end = input_start_sample + local_end
        key = (cache_key_prefix, config_key, abs_start, abs_end)
        return _cached_component(
            minute_cache,
            cache_stats,
            "minute",
            key,
            lambda: extract_v8_minute_component(waveform[:, local_start:local_end], config),
        )

    def history_component(minute_idx: int) -> dict[str, float]:
        local_end = (minute_idx + 1) * config.feature_window_samples
        local_start = local_end - config.rolling_history_samples
        abs_start = input_start_sample + local_start
        abs_end = input_start_sample + local_end
        key = (cache_key_prefix, config_key, abs_start, abs_end)
        return _cached_component(
            history_cache,
            cache_stats,
            "history",
            key,
            lambda: extract_v8_history_component(waveform[:, local_start:local_end], config),
        )

    return _assemble_v8_sequence_from_component_getters(config, minute_component, history_component)


def extract_v8_reference_component_plan_for_segment(
    segment_waveform: np.ndarray,
    input_start_samples: np.ndarray,
    config: V8ExtractionConfig = DEFAULT_V8_EXTRACTION_CONFIG,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    segment = np.asarray(segment_waveform)
    expected_channels = len(config.channel_order)
    if segment.ndim != 2 or segment.shape[0] != expected_channels:
        raise ValueError(f"Expected segment waveform with {expected_channels} channels, got {segment.shape}")
    if tuple(config.channel_order) != ("II", "ABP", "PLETH", "RESP"):
        raise ValueError(f"v8 currently requires channel_order ('II', 'ABP', 'PLETH', 'RESP'), got {config.channel_order}")
    starts = np.asarray(input_start_samples, dtype=np.int64)
    if starts.ndim != 1:
        raise ValueError("input_start_samples must be a 1D array")
    names, name_to_index = _v8_feature_schema()
    n_samples = starts.size
    values = np.full((n_samples, config.n_feature_windows, len(names)), np.nan, dtype=np.float32)
    mask = np.zeros((n_samples, config.n_feature_windows, len(names)), dtype=bool)
    if n_samples == 0:
        return values, mask, names, {
            "samples": 0,
            "minute_requested": 0,
            "minute_computed": 0,
            "minute_hits": 0,
            "history_requested": 0,
            "history_computed": 0,
            "history_hits": 0,
        }

    fw = int(config.feature_window_samples)
    hw = int(config.rolling_history_samples)
    nwin = int(config.n_feature_windows)
    minute_starts = (starts[:, None] + np.arange(nwin, dtype=np.int64)[None, :] * fw).reshape(-1)
    history_offsets = (np.arange(4, nwin, dtype=np.int64) + 1) * fw - hw
    history_starts = (starts[:, None] + history_offsets[None, :]).reshape(-1)
    min_start = int(min(np.min(minute_starts), np.min(history_starts) if history_starts.size else np.min(minute_starts)))
    max_end = int(max(np.max(minute_starts) + fw, np.max(history_starts) + hw if history_starts.size else np.max(minute_starts) + fw))
    if min_start < 0 or max_end > segment.shape[1]:
        raise IndexError(f"Component intervals [{min_start}, {max_end}) outside segment length {segment.shape[1]}")
    unique_minute, minute_inverse = np.unique(minute_starts, return_inverse=True)
    unique_history, history_inverse = np.unique(history_starts, return_inverse=True)
    minute_ids = minute_inverse.reshape(n_samples, nwin)
    history_ids = history_inverse.reshape(n_samples, nwin - 4)
    minute_values = np.full((unique_minute.size, len(names)), np.nan, dtype=np.float32)
    minute_mask = np.zeros((unique_minute.size, len(names)), dtype=bool)
    history_values = np.full((unique_history.size, len(names)), np.nan, dtype=np.float32)
    history_mask = np.zeros((unique_history.size, len(names)), dtype=bool)

    for out_idx, start in enumerate(unique_minute.tolist()):
        component = extract_v8_minute_component(segment[:, int(start) : int(start) + fw], config)
        _put_row(minute_values[out_idx], minute_mask[out_idx], component, names, name_to_index)
    for out_idx, start in enumerate(unique_history.tolist()):
        component = extract_v8_history_component(segment[:, int(start) : int(start) + hw], config)
        _put_row(history_values[out_idx], history_mask[out_idx], component, names, name_to_index)

    chunk_size = 1024
    for start_row in range(0, n_samples, chunk_size):
        end_row = min(n_samples, start_row + chunk_size)
        values[start_row:end_row] = minute_values[minute_ids[start_row:end_row]]
        mask[start_row:end_row] = minute_mask[minute_ids[start_row:end_row]]
        hvalues = history_values[history_ids[start_row:end_row]]
        hmasks = history_mask[history_ids[start_row:end_row]]
        np.copyto(values[start_row:end_row, 4:, :], hvalues, where=hmasks)
        mask[start_row:end_row, 4:, :] |= hmasks

    stats = {
        "samples": int(n_samples),
        "minute_requested": int(minute_starts.size),
        "minute_computed": int(unique_minute.size),
        "minute_hits": int(minute_starts.size - unique_minute.size),
        "history_requested": int(history_starts.size),
        "history_computed": int(unique_history.size),
        "history_hits": int(history_starts.size - unique_history.size),
        "minute_cache_value_bytes": int(minute_values.nbytes + minute_mask.nbytes),
        "history_cache_value_bytes": int(history_values.nbytes + history_mask.nbytes),
    }
    return values, mask, names, stats


def extract_v8_feature_sequence_cached_global(
    waveform: np.ndarray,
    config: V8ExtractionConfig = DEFAULT_V8_EXTRACTION_CONFIG,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    waveform = np.asarray(waveform, dtype=np.float64)
    expected_shape = (len(config.channel_order), config.input_samples)
    if waveform.shape != expected_shape:
        raise ValueError(f"Expected waveform shape {expected_shape}, got {waveform.shape}")
    if tuple(config.channel_order) != ("II", "ABP", "PLETH", "RESP"):
        raise ValueError(f"v8 currently requires channel_order ('II', 'ABP', 'PLETH', 'RESP'), got {config.channel_order}")
    ecg_full = detect_ecg_events(waveform[0], config.sampling_rate_hz, config, include_qrs_template=False)
    abp_full = detect_pulses(waveform[1], config.sampling_rate_hz, "abp", config)
    pleth_full = detect_pulses(waveform[2], config.sampling_rate_hz, "pleth", config)
    resp_full = detect_resp_cycles(waveform[3], config.sampling_rate_hz, config)
    return _extract_v8_feature_sequence_from_event_source(waveform, config, ecg_full, abp_full, pleth_full, resp_full)


def extract_v8_feature_sequence(
    waveform: np.ndarray,
    config: V8ExtractionConfig = DEFAULT_V8_EXTRACTION_CONFIG,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    return extract_v8_feature_sequence_reference(waveform, config=config)

def extract_v8_feature_sequence_from_signal(
    waveform: np.ndarray,
    input_start: int,
    input_end: int,
    config: V8ExtractionConfig = DEFAULT_V8_EXTRACTION_CONFIG,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if input_start < 0 or input_end <= input_start:
        raise ValueError(f"Invalid input interval [{input_start}, {input_end})")
    window = np.asarray(waveform[:, input_start:input_end], dtype=np.float64).copy()
    if window.shape[1] != config.input_samples:
        raise ValueError(f"Input slice length must equal {config.input_samples} samples, got {window.shape[1]}")
    return extract_v8_feature_sequence(window, config=config)
