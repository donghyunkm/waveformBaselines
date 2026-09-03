from __future__ import annotations

import math

import numpy as np
from scipy import signal


def nan_iqr(values: np.ndarray) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return float("nan")
    q75, q25 = np.percentile(arr, [75.0, 25.0])
    return float(q75 - q25)


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(arr.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, arr.size))
    return runs


def interpolate_short_gaps(values: np.ndarray, max_gap_samples: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if arr.size == 0 or max_gap_samples <= 0 or finite.all():
        return arr.copy(), finite.copy()
    out = arr.copy()
    bridged = finite.copy()
    finite_idx = np.flatnonzero(finite)
    if finite_idx.size == 0:
        return out, bridged
    invalid_idx = np.flatnonzero(~finite)
    if invalid_idx.size == 0:
        return out, bridged
    splits = np.split(invalid_idx, np.flatnonzero(np.diff(invalid_idx) != 1) + 1)
    for gap in splits:
        start = int(gap[0])
        end = int(gap[-1]) + 1
        if gap.size > max_gap_samples:
            continue
        if start == 0 or end >= arr.size:
            continue
        left = start - 1
        right = end
        if not finite[left] or not finite[right]:
            continue
        out[start:end] = np.interp(
            np.arange(start, end, dtype=np.float64),
            np.asarray([left, right], dtype=np.float64),
            arr[[left, right]],
        )
        bridged[start:end] = True
    return out, bridged


def robust_scale(values: np.ndarray) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return 1.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    scale = 1.4826 * mad
    return float(scale if scale > 1e-8 else max(np.std(arr), 1.0))


def safe_nanstat(func, values: np.ndarray) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return float("nan")
    return float(func(arr))


def resample_segment(values: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return np.full(target_len, np.nan, dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, num=arr.size)
    x_new = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(x_new, x_old, arr)


def median_template_correlation(segments: list[np.ndarray], target_len: int) -> float:
    if len(segments) < 3:
        return float("nan")
    template_matrix = np.stack([resample_segment(seg, target_len) for seg in segments], axis=0)
    if not np.isfinite(template_matrix).all():
        return float("nan")
    corrs: list[float] = []
    for idx, row in enumerate(template_matrix):
        others = np.delete(template_matrix, idx, axis=0)
        if others.shape[0] < 2:
            continue
        template = np.median(others, axis=0)
        row_std = np.std(row)
        tmpl_std = np.std(template)
        if row_std < 1e-8 or tmpl_std < 1e-8:
            continue
        corr = np.corrcoef(row, template)[0, 1]
        if np.isfinite(corr):
            corrs.append(float(corr))
    if not corrs:
        return float("nan")
    return float(np.median(corrs))


def extreme_value_fraction(values: np.ndarray, atol_fraction: float = 0.01) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return float("nan")
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    vrange = vmax - vmin
    if vrange < 1e-8:
        return 1.0
    atol = max(vrange * atol_fraction, 1e-8)
    near_min = np.abs(arr - vmin) <= atol
    near_max = np.abs(arr - vmax) <= atol
    return float(np.mean(near_min | near_max))


def flatline_fraction(values: np.ndarray, window_size: int, std_threshold: float = 1e-4) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < window_size or window_size <= 0:
        return float(np.std(arr) < std_threshold) if arr.size else 1.0
    flags = []
    for start in range(0, arr.size, window_size):
        window = arr[start : start + window_size]
        if window.size == 0:
            continue
        flags.append(float(np.nanstd(window) < std_threshold))
    return float(np.mean(flags)) if flags else 1.0


def butter_filter(
    values: np.ndarray,
    fs: int,
    low_hz: float | None = None,
    high_hz: float | None = None,
    order: int = 3,
    max_interp_gap_samples: int = 0,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    filled, usable = interpolate_short_gaps(arr, max_gap_samples=max_interp_gap_samples)
    if not usable.any():
        return np.full_like(arr, np.nan, dtype=np.float64)
    nyq = fs / 2.0
    if low_hz and high_hz:
        btype = "bandpass"
        wn = [max(low_hz / nyq, 1e-4), min(high_hz / nyq, 0.999)]
    elif low_hz:
        btype = "highpass"
        wn = max(low_hz / nyq, 1e-4)
    elif high_hz:
        btype = "lowpass"
        wn = min(high_hz / nyq, 0.999)
    else:
        out = np.full_like(arr, np.nan, dtype=np.float64)
        out[usable] = filled[usable]
        return out
    b, a = signal.butter(order, wn, btype=btype)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    padlen = 3 * max(len(a), len(b))
    for start, end in finite_runs(usable):
        segment = filled[start:end]
        if segment.size <= padlen:
            continue
        out[start:end] = signal.filtfilt(b, a, segment, padlen=padlen)
    return out


def slope(values: np.ndarray, fs: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return np.zeros_like(arr)
    return np.gradient(arr) * fs


def linear_trend(values: np.ndarray, x: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if x is None:
        xv = np.arange(arr.size, dtype=np.float64)
    else:
        xv = np.asarray(x, dtype=np.float64)
        if xv.shape != arr.shape:
            raise ValueError("x and values must have the same shape")
    valid = np.isfinite(arr) & np.isfinite(xv)
    arr = arr[valid]
    xv = xv[valid]
    if arr.size < 2:
        return float("nan")
    xv = xv - xv.mean()
    denom = float(np.dot(xv, xv))
    if denom <= 0:
        return float("nan")
    y = arr - arr.mean()
    return float(np.dot(xv, y) / denom)


def safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-8:
        return float("nan")
    return float(num / den)


def agreement_score(a: float, b: float, scale: float = 10.0) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(1.0 / (1.0 + abs(a - b) / scale))


def micro_window_slices(n_samples: int, micro_samples: int) -> list[slice]:
    return [slice(start, min(start + micro_samples, n_samples)) for start in range(0, n_samples, micro_samples)]


def minute_window_slices(n_samples: int, minute_samples: int) -> list[slice]:
    return [slice(start, start + minute_samples) for start in range(0, n_samples, minute_samples)]


def summarize_micro_validity(flags: list[bool]) -> float:
    if not flags:
        return float("nan")
    return float(np.mean(np.asarray(flags, dtype=np.float64)))


def count_plausible_intervals(intervals_s: np.ndarray, min_bpm: float, max_bpm: float) -> float:
    arr = finite_values(intervals_s)
    if arr.size == 0:
        return float("nan")
    lower = 60.0 / max_bpm
    upper = 60.0 / min_bpm
    return float(np.mean((arr >= lower) & (arr <= upper)))


def pad_nan_matrix(n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((n_rows, n_cols), np.nan, dtype=np.float32),
        np.zeros((n_rows, n_cols), dtype=bool),
    )
