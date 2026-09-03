from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .task_specs import DEFAULT_EVENT_TASK, DEFAULT_FEATURE_TASK, EventTaskSpec, FeatureRegressionTaskSpec


@dataclass(frozen=True)
class ICUExtractionPaths:
    """Locations for the sibling icuDataExtraction outputs."""

    output_dir: Path = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/output_v2")


@dataclass(frozen=True)
class EventTargetBundleResult:
    targets: np.ndarray
    mask: np.ndarray
    auxiliary_arrays: dict[str, np.ndarray]
    diagnostics: dict[str, object]


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _normalize_patient_ids(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype(str)


def _build_exact_key_index(patient_ids: np.ndarray, times: np.ndarray) -> dict[tuple[str, float], int]:
    return {(pid, float(ts)): idx for idx, (pid, ts) in enumerate(zip(patient_ids.tolist(), times.tolist()))}


def load_waveform_feature_table(output_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load waveform-derived features keyed by `(patient_id, anchor_time)`.

    The 19 physiological features are aggregated across the 109 sub-windows
    using the mean, matching the current task definition for supervision.
    """
    output_dir = _as_path(output_dir)
    x_stats = np.load(output_dir / "X_stats.npy", mmap_mode="r")
    corr = np.load(output_dir / "corr_features_focused.npy", mmap_mode="r")
    patient_ids = _normalize_patient_ids(np.load(output_dir / "patient_ids.npy", allow_pickle=True))
    anchor_times = np.asarray(np.load(output_dir / "window_times.npy"), dtype=np.float64)

    aggregated_features = np.nanmean(x_stats, axis=2, dtype=np.float64).astype(np.float32)
    values = np.concatenate([aggregated_features, np.asarray(corr, dtype=np.float32)], axis=1)
    return patient_ids, anchor_times, values


def build_feature_regression_targets(
    anchors: pd.DataFrame,
    output_dir: str | Path,
    task_spec: FeatureRegressionTaskSpec = DEFAULT_FEATURE_TASK,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align future regression targets to an anchor table.

    `anchors` must contain:
    - `patient_id`
    - `anchor_time`
    """
    required_cols = {"patient_id", "anchor_time"}
    missing = required_cols - set(anchors.columns)
    if missing:
        raise ValueError(f"Anchor table missing columns: {sorted(missing)}")

    patient_ids, anchor_times, value_matrix = load_waveform_feature_table(output_dir)
    index = _build_exact_key_index(patient_ids, anchor_times)

    anchor_patient_ids = anchors["patient_id"].astype(str).to_numpy()
    anchor_reference_times = anchors["anchor_time"].to_numpy(dtype=np.float64)

    n_rows = len(anchors)
    base_dim = value_matrix.shape[1]
    target_dim = base_dim * len(task_spec.horizons_min)
    targets = np.full((n_rows, target_dim), np.nan, dtype=np.float32)
    mask = np.zeros((n_rows, target_dim), dtype=bool)

    horizon_offset_seconds = {
        horizon_min: horizon_min * 60.0 for horizon_min in task_spec.horizons_min
    }
    if task_spec.horizon_mode == "gap":
        gap_shift = task_spec.input_window_minutes * 60.0
        horizon_offset_seconds = {
            horizon_min: gap_shift + horizon_min * 60.0 for horizon_min in task_spec.horizons_min
        }
    elif task_spec.horizon_mode != "center":
        raise ValueError(f"Unsupported feature horizon mode: {task_spec.horizon_mode}")

    for row_idx, (patient_id, reference_time) in enumerate(zip(anchor_patient_ids, anchor_reference_times)):
        for horizon_idx, horizon_min in enumerate(task_spec.horizons_min):
            col_start = horizon_idx * base_dim
            col_end = col_start + base_dim
            future_key = (patient_id, float(reference_time + horizon_offset_seconds[horizon_min]))
            target_row_idx = index.get(future_key)
            if target_row_idx is None:
                continue
            target_values = value_matrix[target_row_idx]
            targets[row_idx, col_start:col_end] = target_values
            mask[row_idx, col_start:col_end] = np.isfinite(target_values)

    return targets, mask


def load_numerics_windows(output_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.memmap]:
    """Load numerics windows keyed by `(patient_id, anchor_time)`."""
    output_dir = _as_path(output_dir)
    patient_ids = _normalize_patient_ids(np.load(output_dir / "numerics_patient_ids.npy", allow_pickle=True))
    anchor_times = np.asarray(np.load(output_dir / "numerics_window_times.npy"), dtype=np.float64)
    numerics = np.load(output_dir / "X_numerics.npy", mmap_mode="r")
    return patient_ids, anchor_times, numerics


def _group_patient_rows(patient_ids: np.ndarray) -> dict[str, np.ndarray]:
    groups = {}
    for pid in np.unique(patient_ids):
        groups[str(pid)] = np.flatnonzero(patient_ids == pid)
    return groups


def _window_interval(center_time: float, n_samples: int = 1200) -> tuple[int, int]:
    start = int(round(center_time - (n_samples / 2)))
    end = start + n_samples
    return start, end


def _range_has_event(
    candidate_times: np.ndarray,
    candidate_windows: np.ndarray,
    interval_start: int,
    interval_end: int,
    channel_idx: int,
    comparison: str,
    threshold: float,
) -> tuple[bool, bool]:
    """
    Return `(has_valid_data, event_detected)` over an absolute-time interval.
    """
    interval_len = interval_end - interval_start
    if interval_len <= 0:
        return False, False

    valid_mask = np.zeros(interval_len, dtype=bool)
    event_mask = np.zeros(interval_len, dtype=bool)

    for center_time, window in zip(candidate_times.tolist(), candidate_windows):
        window_start, window_end = _window_interval(center_time, n_samples=window.shape[1])
        overlap_start = max(interval_start, window_start)
        overlap_end = min(interval_end, window_end)
        if overlap_start >= overlap_end:
            continue

        sample_start = overlap_start - window_start
        sample_end = overlap_end - window_start
        values = np.asarray(window[channel_idx, sample_start:sample_end], dtype=np.float32)
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            continue

        offset_start = overlap_start - interval_start
        offset_end = overlap_end - interval_start
        valid_mask[offset_start:offset_end] |= finite_mask

        if comparison == "lt":
            event_mask[offset_start:offset_end] |= finite_mask & (values < threshold)
        elif comparison == "gt":
            event_mask[offset_start:offset_end] |= finite_mask & (values > threshold)
        else:
            raise ValueError(f"Unsupported comparison: {comparison}")

    has_valid_data = bool(valid_mask.any())
    if not has_valid_data:
        return False, False

    event_detected = bool(event_mask.any())
    return has_valid_data, event_detected


def _range_has_sustained_event(
    candidate_times: np.ndarray,
    candidate_windows: np.ndarray,
    interval_start: int,
    interval_end: int,
    channel_idx: int,
    comparison: str,
    threshold: float,
    sustain_minutes: int,
    minute_seconds: int = 60,
) -> tuple[bool, bool]:
    """
    Return `(has_valid_data, event_detected)` for a sustained minute-level event.

    The interval is first collapsed into contiguous minute bins using the median
    over finite 1 Hz numerics values in each minute, mirroring the minute-level
    labeling pattern used in PhysioJEPA. A positive event requires
    `sustain_minutes` consecutive valid minute bins beyond threshold.
    """
    interval_len = interval_end - interval_start
    if interval_len < minute_seconds:
        return False, False

    value_series = np.full(interval_len, np.nan, dtype=np.float32)
    valid_mask = np.zeros(interval_len, dtype=bool)

    for center_time, window in zip(candidate_times.tolist(), candidate_windows):
        window_start, window_end = _window_interval(center_time, n_samples=window.shape[1])
        overlap_start = max(interval_start, window_start)
        overlap_end = min(interval_end, window_end)
        if overlap_start >= overlap_end:
            continue

        sample_start = overlap_start - window_start
        sample_end = overlap_end - window_start
        values = np.asarray(window[channel_idx, sample_start:sample_end], dtype=np.float32)
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            continue

        offset_start = overlap_start - interval_start
        offset_end = overlap_end - interval_start
        current_values = value_series[offset_start:offset_end]
        current_values[finite_mask] = values[finite_mask]
        value_series[offset_start:offset_end] = current_values
        valid_mask[offset_start:offset_end] |= finite_mask

    n_full_minutes = interval_len // minute_seconds
    has_valid_data = False
    run_length = 0

    for minute_idx in range(n_full_minutes):
        start = minute_idx * minute_seconds
        end = start + minute_seconds
        minute_valid = valid_mask[start:end]
        if not np.any(minute_valid):
            run_length = 0
            continue

        has_valid_data = True
        minute_values = value_series[start:end][minute_valid]
        minute_summary = float(np.median(minute_values))

        if comparison == "lt":
            is_event_minute = minute_summary < threshold
        elif comparison == "le":
            is_event_minute = minute_summary <= threshold
        elif comparison == "gt":
            is_event_minute = minute_summary > threshold
        elif comparison == "ge":
            is_event_minute = minute_summary >= threshold
        else:
            raise ValueError(f"Unsupported comparison: {comparison}")

        if is_event_minute:
            run_length += 1
            if run_length >= sustain_minutes:
                return True, True
        else:
            run_length = 0

    return has_valid_data, False


def _patient_second_series(
    patient_times: np.ndarray,
    patient_windows: np.ndarray,
    channel_idx: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    if len(patient_times) == 0:
        return 0, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

    window_len = int(patient_windows.shape[2])
    window_starts = np.asarray(np.round(patient_times), dtype=np.int64) - (window_len // 2)
    global_start = int(window_starts.min())
    global_end = int((window_starts + window_len).max())
    series_len = global_end - global_start

    values = np.full(series_len, np.nan, dtype=np.float32)
    valid = np.zeros(series_len, dtype=bool)

    for center_time, window in zip(patient_times.tolist(), patient_windows):
        window_start, _ = _window_interval(center_time, n_samples=window.shape[1])
        channel_values = np.asarray(window[channel_idx], dtype=np.float32)
        finite_idx = np.flatnonzero(np.isfinite(channel_values))
        if finite_idx.size == 0:
            continue
        absolute_idx = (window_start - global_start) + finite_idx
        values[absolute_idx] = channel_values[finite_idx]
        valid[absolute_idx] = True

    return global_start, values, valid


def _summarize_second_series_to_minutes(
    start_time: int,
    values: np.ndarray,
    valid: np.ndarray,
    minute_seconds: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=bool),
        )

    observed = np.flatnonzero(valid)
    if observed.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=bool),
        )
    first_observed = int(observed[0])
    last_observed = int(observed[-1]) + 1
    start_time += first_observed
    values = values[first_observed:last_observed]
    valid = valid[first_observed:last_observed]

    n_full_minutes = values.size // minute_seconds
    if n_full_minutes <= 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=bool),
        )

    minute_times = start_time + np.arange(n_full_minutes, dtype=np.int64) * minute_seconds
    minute_values = np.full(n_full_minutes, np.nan, dtype=np.float32)
    minute_valid = np.zeros(n_full_minutes, dtype=bool)

    for minute_idx in range(n_full_minutes):
        sec_start = minute_idx * minute_seconds
        sec_end = sec_start + minute_seconds
        minute_mask = valid[sec_start:sec_end]
        if not np.any(minute_mask):
            continue
        minute_valid[minute_idx] = True
        minute_values[minute_idx] = float(np.median(values[sec_start:sec_end][minute_mask]))

    return minute_times, minute_values, minute_valid


def _build_minute_event_labels(
    patient_times: np.ndarray,
    patient_windows: np.ndarray,
    channel_idx: int,
    threshold: float,
    comparison: str,
    minute_seconds: int = 60,
) -> dict[str, np.ndarray]:
    second_start, value_series, valid_series = _patient_second_series(
        patient_times,
        patient_windows,
        channel_idx=channel_idx,
    )
    minute_times, minute_values, minute_valid = _summarize_second_series_to_minutes(
        second_start,
        value_series,
        valid_series,
        minute_seconds=minute_seconds,
    )
    if comparison == "lt":
        event_minutes = minute_valid & (minute_values < threshold)
    elif comparison == "le":
        event_minutes = minute_valid & (minute_values <= threshold)
    elif comparison == "gt":
        event_minutes = minute_valid & (minute_values > threshold)
    elif comparison == "ge":
        event_minutes = minute_valid & (minute_values >= threshold)
    else:
        raise ValueError(f"Unsupported comparison: {comparison}")
    return {
        "minute_time": minute_times,
        "minute_value": minute_values,
        "minute_valid": minute_valid,
        "event_minutes": event_minutes,
    }


def _detect_sustained_event_starts(
    minute_times: np.ndarray,
    minute_valid: np.ndarray,
    event_minutes: np.ndarray,
    sustain_minutes: int,
    minute_seconds: int = 60,
) -> np.ndarray:
    if len(minute_times) < sustain_minutes:
        return np.zeros(0, dtype=np.int64)

    qualifying_starts: list[int] = []
    for start_idx in range(len(minute_times) - sustain_minutes + 1):
        window_times = minute_times[start_idx : start_idx + sustain_minutes]
        expected = minute_times[start_idx] + np.arange(sustain_minutes, dtype=np.int64) * minute_seconds
        if not np.array_equal(window_times, expected):
            continue
        if not np.all(minute_valid[start_idx : start_idx + sustain_minutes]):
            continue
        if not np.all(event_minutes[start_idx : start_idx + sustain_minutes]):
            continue
        qualifying_starts.append(int(minute_times[start_idx]))

    if not qualifying_starts:
        return np.zeros(0, dtype=np.int64)

    collapsed = [qualifying_starts[0]]
    for event_time in qualifying_starts[1:]:
        if event_time - collapsed[-1] < sustain_minutes * minute_seconds:
            continue
        collapsed.append(event_time)
    return np.asarray(collapsed, dtype=np.int64)


def _window_has_only_clean_minutes(
    minute_times: np.ndarray,
    minute_valid: np.ndarray,
    event_minutes: np.ndarray,
    window_start: int,
    sustain_minutes: int,
    minute_seconds: int = 60,
) -> bool:
    expected_times = window_start + np.arange(sustain_minutes, dtype=np.int64) * minute_seconds
    idx = np.searchsorted(minute_times, expected_times)
    if np.any(idx >= len(minute_times)):
        return False
    if not np.array_equal(minute_times[idx], expected_times):
        return False
    if not np.all(minute_valid[idx]):
        return False
    return not bool(np.any(event_minutes[idx]))


def build_event_classification_targets(
    anchors: pd.DataFrame,
    output_dir: str | Path,
    task_spec: EventTaskSpec = DEFAULT_EVENT_TASK,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build binary future event labels from 1 Hz numerics windows.

    `anchors` must contain:
    - `patient_id`
    - `anchor_time`: reference timestamp for the aligned window
    - optional `input_start_time`: start time of the model input window
    - optional `input_end_time`: end time of the model input window

    If `input_start_time` is provided, labels are masked out for anchors where
    hypotension or tachycardia is already active during the input interval.
    """
    required_cols = {"patient_id", "anchor_time"}
    missing = required_cols - set(anchors.columns)
    if missing:
        raise ValueError(f"Anchor table missing columns: {sorted(missing)}")

    numerics_patient_ids, numerics_times, numerics = load_numerics_windows(output_dir)
    patient_groups = _group_patient_rows(numerics_patient_ids)

    anchor_patient_ids = anchors["patient_id"].astype(str).to_numpy()
    if "input_end_time" in anchors.columns:
        anchor_end_times = anchors["input_end_time"].to_numpy(dtype=np.float64)
    else:
        anchor_end_times = anchors["anchor_time"].to_numpy(dtype=np.float64)
    input_start_times = None
    if "input_start_time" in anchors.columns:
        input_start_times = anchors["input_start_time"].to_numpy(dtype=np.float64)

    n_rows = len(anchors)
    n_horizons = len(task_spec.horizons_min)
    n_events = len(task_spec.event_names)
    target_dim = n_horizons * n_events
    targets = np.full((n_rows, target_dim), -1, dtype=np.int8)
    mask = np.zeros((n_rows, target_dim), dtype=bool)

    for row_idx, patient_id in enumerate(anchor_patient_ids):
        patient_rows = patient_groups.get(patient_id)
        if patient_rows is None or len(patient_rows) == 0:
            continue

        patient_times = numerics_times[patient_rows]
        patient_windows = numerics[patient_rows]

        input_start = None if input_start_times is None else int(round(input_start_times[row_idx]))
        anchor_end = int(round(anchor_end_times[row_idx]))

        is_stable = True
        if input_start is not None:
            _, hypotension_active = _range_has_sustained_event(
                patient_times,
                patient_windows,
                input_start,
                anchor_end,
                channel_idx=task_spec.hypotension_channel,
                comparison="le",
                threshold=task_spec.hypotension_threshold,
                sustain_minutes=task_spec.sustain_minutes,
            )
            _, tachy_active = _range_has_sustained_event(
                patient_times,
                patient_windows,
                input_start,
                anchor_end,
                channel_idx=task_spec.tachycardia_channel,
                comparison="gt",
                threshold=task_spec.tachycardia_threshold,
                sustain_minutes=task_spec.sustain_minutes,
            )
            is_stable = not (hypotension_active or tachy_active)

        if not is_stable:
            continue

        for horizon_idx, horizon_min in enumerate(task_spec.horizons_min):
            future_start = anchor_end
            future_end = anchor_end + horizon_min * 60

            valid_hypo, has_hypo = _range_has_sustained_event(
                patient_times,
                patient_windows,
                future_start,
                future_end,
                channel_idx=task_spec.hypotension_channel,
                comparison="le",
                threshold=task_spec.hypotension_threshold,
                sustain_minutes=task_spec.sustain_minutes,
            )
            valid_tachy, has_tachy = _range_has_sustained_event(
                patient_times,
                patient_windows,
                future_start,
                future_end,
                channel_idx=task_spec.tachycardia_channel,
                comparison="gt",
                threshold=task_spec.tachycardia_threshold,
                sustain_minutes=task_spec.sustain_minutes,
            )

            base_col = horizon_idx * n_events
            if valid_hypo:
                targets[row_idx, base_col] = int(has_hypo)
                mask[row_idx, base_col] = True
            if valid_tachy:
                targets[row_idx, base_col + 1] = int(has_tachy)
                mask[row_idx, base_col + 1] = True

    return targets, mask


def build_filtered_anchor_horizon_event_targets(
    anchors: pd.DataFrame,
    output_dir: str | Path,
    task_spec: EventTaskSpec = DEFAULT_EVENT_TASK,
) -> EventTargetBundleResult:
    baseline_targets, baseline_mask = build_event_classification_targets(
        anchors,
        output_dir,
        task_spec=EventTaskSpec(
            horizons_min=task_spec.horizons_min,
            target_generation_mode="anchor_horizon",
            hypotension_threshold=task_spec.hypotension_threshold,
            tachycardia_threshold=task_spec.tachycardia_threshold,
            sustain_minutes=task_spec.sustain_minutes,
            hypotension_channel=task_spec.hypotension_channel,
            tachycardia_channel=task_spec.tachycardia_channel,
            event_names=task_spec.event_names,
        ),
    )

    targets = baseline_targets.copy()
    mask = baseline_mask.copy()

    numerics_patient_ids, numerics_times, numerics = load_numerics_windows(output_dir)
    patient_groups = _group_patient_rows(numerics_patient_ids)
    anchor_patient_ids = anchors["patient_id"].astype(str).to_numpy()
    if "input_end_time" in anchors.columns:
        prediction_times = anchors["input_end_time"].to_numpy(dtype=np.float64)
    else:
        prediction_times = anchors["anchor_time"].to_numpy(dtype=np.float64)

    patient_hypotension_labels: dict[str, dict[str, np.ndarray]] = {}
    patient_has_sustained_hypotension: dict[str, bool] = {}
    positive_record_last_event_times: list[float] = []

    for patient_id in np.unique(anchor_patient_ids):
        patient_rows = patient_groups.get(patient_id)
        if patient_rows is None or len(patient_rows) == 0:
            patient_has_sustained_hypotension[patient_id] = False
            continue

        patient_times = numerics_times[patient_rows]
        patient_windows = numerics[patient_rows]
        labels = _build_minute_event_labels(
            patient_times,
            patient_windows,
            channel_idx=task_spec.hypotension_channel,
            threshold=task_spec.hypotension_threshold,
            comparison="le",
        )
        patient_hypotension_labels[patient_id] = labels
        event_starts = _detect_sustained_event_starts(
            labels["minute_time"],
            labels["minute_valid"],
            labels["event_minutes"],
            sustain_minutes=task_spec.sustain_minutes,
        )
        has_event = bool(len(event_starts) > 0)
        patient_has_sustained_hypotension[patient_id] = has_event
        if has_event:
            positive_record_last_event_times.append(float(event_starts[-1]))

    mean_last_positive_event_time = (
        float(np.mean(np.asarray(positive_record_last_event_times, dtype=np.float64)))
        if positive_record_last_event_times
        else None
    )

    n_events = len(task_spec.event_names)
    diagnostics: dict[str, object] = {
        "mode": "anchor_horizon_filtered",
        "negative_filter_rules": {
            "clean_outcome_window_minutes": int(task_spec.sustain_minutes),
            "require_valid_outcome_minute": True,
            "exclude_negative_samples_from_positive_recordings": True,
            "late_negative_cutoff": mean_last_positive_event_time,
        },
        "per_horizon": {},
    }

    for horizon_idx, horizon_min in enumerate(task_spec.horizons_min):
        hypo_col = horizon_idx * n_events
        positives_kept = int(((targets[:, hypo_col] == 1) & mask[:, hypo_col]).sum())
        baseline_negatives = int(((targets[:, hypo_col] == 0) & mask[:, hypo_col]).sum())
        dropped_positive_record = 0
        dropped_late_cutoff = 0
        dropped_outcome_window = 0

        for row_idx, patient_id in enumerate(anchor_patient_ids):
            if not mask[row_idx, hypo_col] or targets[row_idx, hypo_col] != 0:
                continue

            labels = patient_hypotension_labels.get(patient_id)
            if labels is None:
                targets[row_idx, hypo_col] = -1
                mask[row_idx, hypo_col] = False
                dropped_outcome_window += 1
                continue

            if patient_has_sustained_hypotension.get(patient_id, False):
                targets[row_idx, hypo_col] = -1
                mask[row_idx, hypo_col] = False
                dropped_positive_record += 1
                continue

            outcome_time = int(round(prediction_times[row_idx])) + horizon_min * 60
            if mean_last_positive_event_time is not None and outcome_time > mean_last_positive_event_time:
                targets[row_idx, hypo_col] = -1
                mask[row_idx, hypo_col] = False
                dropped_late_cutoff += 1
                continue

            if not _window_has_only_clean_minutes(
                labels["minute_time"],
                labels["minute_valid"],
                labels["event_minutes"],
                window_start=outcome_time,
                sustain_minutes=task_spec.sustain_minutes,
            ):
                targets[row_idx, hypo_col] = -1
                mask[row_idx, hypo_col] = False
                dropped_outcome_window += 1
                continue

        retained_negatives = int(((targets[:, hypo_col] == 0) & mask[:, hypo_col]).sum())
        retained_total = int(mask[:, hypo_col].sum())
        diagnostics["per_horizon"][str(horizon_min)] = {
            "positives_kept": positives_kept,
            "baseline_negative_candidates": baseline_negatives,
            "retained_negative_candidates": retained_negatives,
            "dropped_negative_positive_recording": dropped_positive_record,
            "dropped_negative_after_mean_last_positive_event": dropped_late_cutoff,
            "dropped_negative_outcome_window": dropped_outcome_window,
            "retained_prevalence": float(positives_kept / retained_total) if retained_total > 0 else 0.0,
        }

    return EventTargetBundleResult(
        targets=targets,
        mask=mask,
        auxiliary_arrays={},
        diagnostics=diagnostics,
    )


def build_event_targets(
    anchors: pd.DataFrame,
    output_dir: str | Path,
    task_spec: EventTaskSpec = DEFAULT_EVENT_TASK,
) -> EventTargetBundleResult:
    if task_spec.target_generation_mode == "anchor_horizon":
        targets, mask = build_event_classification_targets(
            anchors,
            output_dir,
            task_spec=task_spec,
        )
        return EventTargetBundleResult(
            targets=targets,
            mask=mask,
            auxiliary_arrays={},
            diagnostics={"mode": "anchor_horizon"},
        )

    if task_spec.target_generation_mode == "anchor_horizon_filtered":
        return build_filtered_anchor_horizon_event_targets(
            anchors,
            output_dir,
            task_spec=task_spec,
        )

    raise ValueError(f"Unsupported event target_generation_mode: {task_spec.target_generation_mode}")


def save_target_bundle(
    output_path: str | Path,
    anchors: pd.DataFrame,
    feature_targets: np.ndarray | None,
    feature_mask: np.ndarray | None,
    event_targets: np.ndarray | None,
    event_mask: np.ndarray | None,
    feature_spec: FeatureRegressionTaskSpec = DEFAULT_FEATURE_TASK,
    event_spec: EventTaskSpec = DEFAULT_EVENT_TASK,
    event_auxiliary_arrays: dict[str, np.ndarray] | None = None,
    event_diagnostics: dict[str, object] | None = None,
) -> None:
    """Persist targets and metadata to a compressed `.npz` bundle."""
    output_path = _as_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "anchor_patient_ids": anchors["patient_id"].astype(str).to_numpy(),
        "anchor_times": anchors["anchor_time"].to_numpy(dtype=np.float64),
    }
    if "anchor_id" in anchors.columns:
        arrays["anchor_ids"] = anchors["anchor_id"].to_numpy(dtype=np.int64)
    if "segment_id" in anchors.columns:
        arrays["segment_ids"] = anchors["segment_id"].astype(str).to_numpy()
    if "seg_name" in anchors.columns:
        arrays["segment_names"] = anchors["seg_name"].astype(str).to_numpy()
    if "input_start_time" in anchors.columns:
        arrays["input_start_times"] = anchors["input_start_time"].to_numpy(dtype=np.float64)
    if "input_end_time" in anchors.columns:
        arrays["input_end_times"] = anchors["input_end_time"].to_numpy(dtype=np.float64)
    if "split_label" in anchors.columns:
        arrays["split_labels"] = anchors["split_label"].astype(str).to_numpy()
    if feature_targets is not None and feature_mask is not None:
        arrays["feature_targets"] = feature_targets
        arrays["feature_mask"] = feature_mask
    if event_targets is not None and event_mask is not None:
        arrays["event_targets"] = event_targets
        arrays["event_mask"] = event_mask
    if event_auxiliary_arrays:
        arrays.update(event_auxiliary_arrays)

    np.savez_compressed(output_path, **arrays)

    metadata = {
        "feature_spec": asdict(feature_spec),
        "feature_target_names": list(feature_spec.target_names),
        "event_spec": asdict(event_spec),
        "event_target_names": list(event_spec.target_names),
        "n_anchors": int(len(anchors)),
    }
    if event_diagnostics is not None:
        metadata["event_diagnostics"] = event_diagnostics
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_anchor_table(csv_path: str | Path) -> pd.DataFrame:
    """Load anchors from CSV and preserve the minimal required schema."""
    csv_path = _as_path(csv_path)
    anchors = pd.read_csv(csv_path)
    required_cols = {"patient_id", "anchor_time"}
    missing = required_cols - set(anchors.columns)
    if missing:
        raise ValueError(f"Anchor CSV missing columns: {sorted(missing)}")
    keep_cols = ["patient_id", "anchor_time"]
    if "input_start_time" in anchors.columns:
        keep_cols.append("input_start_time")
    if "input_end_time" in anchors.columns:
        keep_cols.append("input_end_time")
    return anchors[keep_cols].copy()
