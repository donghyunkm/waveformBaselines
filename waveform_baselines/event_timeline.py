from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ALIGNMENT_RESIDUAL_SECONDS = 1.0


@dataclass(frozen=True)
class ChannelResolution:
    name: str
    index: int
    aliases_considered: tuple[str, ...]
    finite_sample_fraction: float
    valid_minute_fraction: float


@dataclass(frozen=True)
class MinuteTimeline:
    origin_seconds: float
    minute_times: np.ndarray
    minute_values: np.ndarray
    minute_valid: np.ndarray
    event_minutes: np.ndarray
    finite_samples_per_minute: np.ndarray
    alignment_residual_seconds: np.ndarray

    @property
    def n_minutes(self) -> int:
        return int(len(self.minute_times))


def resolve_channel(
    channel_names: list[str] | tuple[str, ...],
    *,
    preferred_name: str,
    aliases: list[str] | tuple[str, ...],
    numerics: np.ndarray | None = None,
    min_finite_fraction: float = 0.0,
) -> ChannelResolution:
    considered = (preferred_name, *tuple(aliases))
    matches = [idx for idx, name in enumerate(channel_names) if str(name) in considered]
    if not matches:
        raise ValueError(
            f"No supported channel found for {preferred_name!r}; "
            f"considered={list(considered)!r}, available={list(channel_names)!r}"
        )
    if len(matches) > 1:
        matched_names = [str(channel_names[idx]) for idx in matches]
        raise ValueError(f"Ambiguous channel match for {preferred_name!r}: {matched_names}")

    idx = int(matches[0])
    finite_fraction = 1.0
    if numerics is not None:
        values = np.asarray(numerics[:, idx, :])
        finite_fraction = float(np.mean(np.isfinite(values))) if values.size else 0.0
        if finite_fraction < min_finite_fraction:
            raise ValueError(
                f"Selected channel {channel_names[idx]!r} has finite fraction "
                f"{finite_fraction:.6f}, below {min_finite_fraction:.6f}"
            )
    return ChannelResolution(
        name=str(channel_names[idx]),
        index=idx,
        aliases_considered=tuple(str(x) for x in considered),
        finite_sample_fraction=finite_fraction,
        valid_minute_fraction=float("nan"),
    )


def timestamp_to_minute_index(
    timestamp_seconds: float,
    origin_seconds: float,
    *,
    tolerance_seconds: float = MAX_ALIGNMENT_RESIDUAL_SECONDS,
) -> tuple[int, float]:
    minute_position = (float(timestamp_seconds) - float(origin_seconds)) / 60.0
    minute_index = int(round(minute_position))
    aligned = float(origin_seconds) + minute_index * 60.0
    residual = abs(float(timestamp_seconds) - aligned)
    if residual > tolerance_seconds:
        raise ValueError(
            f"Timestamp {timestamp_seconds:.6f} is not minute-aligned to origin "
            f"{origin_seconds:.6f}; residual={residual:.6f}s"
        )
    return minute_index, residual


def build_minute_timeline_from_windows(
    *,
    window_times: np.ndarray,
    windows: np.ndarray,
    channel_idx: int,
    threshold: float,
    comparison: str,
    expected_samples_per_minute: int = 60,
    min_valid_fraction_per_minute: float = 1.0 / 60.0,
    timeline_origin_seconds: float | None = None,
) -> MinuteTimeline:
    if windows.ndim != 3:
        raise ValueError(f"Expected windows with shape (N, C, T), got {windows.shape}")
    if len(window_times) != windows.shape[0]:
        raise ValueError("window_times length must match windows rows")
    if windows.shape[0] == 0:
        origin = 0.0 if timeline_origin_seconds is None else float(timeline_origin_seconds)
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float32)
        empty_b = np.zeros(0, dtype=bool)
        return MinuteTimeline(origin, empty_i, empty_f, empty_b, empty_b, empty_i, empty_f)

    n_seconds = int(windows.shape[2])
    starts = np.asarray(np.round(np.asarray(window_times, dtype=np.float64) - n_seconds / 2.0), dtype=np.int64)
    ends = starts + n_seconds
    origin = float(starts.min()) if timeline_origin_seconds is None else float(timeline_origin_seconds)
    total_seconds = int(ends.max() - origin)
    if total_seconds <= 0:
        raise ValueError("Numeric windows do not overlap the requested timeline origin")

    values = np.full(total_seconds, np.nan, dtype=np.float32)
    valid = np.zeros(total_seconds, dtype=bool)
    for start, window in zip(starts.tolist(), windows):
        offset = int(start - origin)
        if offset >= total_seconds or offset + n_seconds <= 0:
            continue
        src_start = max(0, -offset)
        dst_start = max(0, offset)
        n = min(n_seconds - src_start, total_seconds - dst_start)
        if n <= 0:
            continue
        arr = np.asarray(window[channel_idx, src_start:src_start + n], dtype=np.float32)
        finite = np.isfinite(arr)
        if not np.any(finite):
            continue
        target = values[dst_start:dst_start + n]
        target[finite] = arr[finite]
        values[dst_start:dst_start + n] = target
        valid[dst_start:dst_start + n] |= finite

    n_minutes = total_seconds // 60
    minute_times = origin + np.arange(n_minutes, dtype=np.float64) * 60.0
    minute_values = np.full(n_minutes, np.nan, dtype=np.float32)
    minute_valid = np.zeros(n_minutes, dtype=bool)
    finite_counts = np.zeros(n_minutes, dtype=np.int64)
    required_samples = int(np.ceil(expected_samples_per_minute * min_valid_fraction_per_minute))
    required_samples = max(1, required_samples)

    for minute_idx in range(n_minutes):
        lo = minute_idx * 60
        hi = lo + 60
        finite = valid[lo:hi]
        finite_count = int(finite.sum())
        finite_counts[minute_idx] = finite_count
        if finite_count < required_samples:
            continue
        minute_valid[minute_idx] = True
        minute_values[minute_idx] = float(np.median(values[lo:hi][finite]))

    if comparison == "le":
        event_minutes = minute_valid & (minute_values <= threshold)
    elif comparison == "lt":
        event_minutes = minute_valid & (minute_values < threshold)
    elif comparison == "gt":
        event_minutes = minute_valid & (minute_values > threshold)
    elif comparison == "ge":
        event_minutes = minute_valid & (minute_values >= threshold)
    else:
        raise ValueError(f"Unsupported comparison: {comparison}")

    residuals = np.abs(minute_times - (origin + np.arange(n_minutes, dtype=np.float64) * 60.0))
    return MinuteTimeline(
        origin_seconds=origin,
        minute_times=minute_times,
        minute_values=minute_values,
        minute_valid=minute_valid,
        event_minutes=event_minutes,
        finite_samples_per_minute=finite_counts,
        alignment_residual_seconds=residuals.astype(np.float32),
    )


def summarize_numeric(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }
