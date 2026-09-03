from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .event_episodes import (
    EventEpisode,
    any_episode_overlaps,
    any_episode_start_in_interval,
    episode_starts_at,
)


class LabelState(IntEnum):
    INVALID = -1
    NEGATIVE = 0
    POSITIVE = 1


class InvalidReason(IntEnum):
    KEPT = 0
    TIME_ALIGNMENT = 1
    INSUFFICIENT_INPUT_COVERAGE = 2
    INSUFFICIENT_FORECAST_COVERAGE = 3
    INSUFFICIENT_CONFIRMATION_COVERAGE = 4
    OUTCOME_MISSING = 5
    OUTCOME_MIXED = 6
    ACTIVE_BEFORE_OUTCOME = 7
    NO_NUMERICS = 8
    ACTIVE_EVENT_IN_INPUT = 9
    INPUT_COVERAGE_INSUFFICIENT = INSUFFICIENT_INPUT_COVERAGE
    FUTURE_COVERAGE_INSUFFICIENT = INSUFFICIENT_CONFIRMATION_COVERAGE
    EVENT_ACTIVE_IN_INPUT = ACTIVE_EVENT_IN_INPUT


class NegativeFilterReason(IntEnum):
    KEPT = 0
    BASE_INVALID = 1
    POSITIVE = 2
    EXCLUDED_EVENT_GROUP = 3
    AFTER_LATE_CUTOFF = 4
    NOT_STRICTLY_CLEAN = 5
    INSUFFICIENT_COVERAGE = 6
    CROSSES_SEGMENT_BOUNDARY = 7
    NO_CONFIRMED_ONSET_CUTOFF_GROUP = 8
    NO_POSITIVE_CUTOFF_GROUP = NO_CONFIRMED_ONSET_CUTOFF_GROUP


@dataclass(frozen=True)
class AnchorMinuteBounds:
    input_start_minute: int
    input_end_minute: int


def slice_is_covered(minute_valid: np.ndarray, start: int, end: int) -> bool:
    if start < 0 or end > len(minute_valid) or end < start:
        return False
    return bool(np.all(minute_valid[start:end]))


def label_onset_within_horizon(
    *,
    bounds: AnchorMinuteBounds,
    minute_valid: np.ndarray,
    event_minutes: np.ndarray,
    episodes: list[EventEpisode],
    horizon_minutes: int,
    sustain_minutes: int,
    exclude_active_input: bool = True,
    negative_policy: str = "observable-no-onset",
) -> tuple[LabelState, InvalidReason]:
    input_start = bounds.input_start_minute
    input_end = bounds.input_end_minute
    horizon_end = input_end + horizon_minutes
    required_end = input_end + horizon_minutes + sustain_minutes - 1

    if not slice_is_covered(minute_valid, input_start, input_end):
        return LabelState.INVALID, InvalidReason.INSUFFICIENT_INPUT_COVERAGE
    if exclude_active_input and any_episode_overlaps(episodes, input_start, input_end):
        return LabelState.INVALID, InvalidReason.ACTIVE_EVENT_IN_INPUT
    if horizon_end > len(minute_valid) or not slice_is_covered(minute_valid, input_end, horizon_end):
        return LabelState.INVALID, InvalidReason.INSUFFICIENT_FORECAST_COVERAGE
    if required_end > len(minute_valid):
        return LabelState.INVALID, InvalidReason.INSUFFICIENT_CONFIRMATION_COVERAGE

    has_onset = any_episode_start_in_interval(episodes, input_end, horizon_end)
    if has_onset:
        return LabelState.POSITIVE, InvalidReason.KEPT

    if not slice_is_covered(minute_valid, input_end, required_end):
        return LabelState.INVALID, InvalidReason.INSUFFICIENT_CONFIRMATION_COVERAGE

    if negative_policy == "observable-no-onset":
        return LabelState.NEGATIVE, InvalidReason.KEPT
    if negative_policy == "strict-clean-horizon":
        if not slice_is_covered(minute_valid, input_end, horizon_end):
            return LabelState.INVALID, InvalidReason.OUTCOME_MISSING
        if bool(np.any(np.asarray(event_minutes, dtype=bool)[input_end:horizon_end])):
            return LabelState.INVALID, InvalidReason.OUTCOME_MIXED
        return LabelState.NEGATIVE, InvalidReason.KEPT
    raise ValueError(f"Unsupported negative_policy: {negative_policy}")


def label_fixed_forecast_window(
    *,
    bounds: AnchorMinuteBounds,
    minute_valid: np.ndarray,
    event_minutes: np.ndarray,
    episodes: list[EventEpisode],
    forecast_gap_minutes: int,
    sustain_minutes: int,
    exclude_active_input: bool = True,
) -> tuple[LabelState, InvalidReason]:
    input_start = bounds.input_start_minute
    input_end = bounds.input_end_minute
    outcome_start = input_end + forecast_gap_minutes
    outcome_end = outcome_start + sustain_minutes

    if not slice_is_covered(minute_valid, input_start, input_end):
        return LabelState.INVALID, InvalidReason.INSUFFICIENT_INPUT_COVERAGE
    if exclude_active_input and any_episode_overlaps(episodes, input_start, input_end):
        return LabelState.INVALID, InvalidReason.ACTIVE_EVENT_IN_INPUT
    if outcome_end > len(minute_valid):
        return LabelState.INVALID, InvalidReason.INSUFFICIENT_CONFIRMATION_COVERAGE
    if not slice_is_covered(minute_valid, outcome_start, outcome_end):
        return LabelState.INVALID, InvalidReason.OUTCOME_MISSING

    if episode_starts_at(episodes, outcome_start):
        return LabelState.POSITIVE, InvalidReason.KEPT
    if any_episode_overlaps(episodes, outcome_start, outcome_start + 1):
        return LabelState.INVALID, InvalidReason.ACTIVE_BEFORE_OUTCOME
    if not bool(np.any(np.asarray(event_minutes, dtype=bool)[outcome_start:outcome_end])):
        return LabelState.NEGATIVE, InvalidReason.KEPT
    return LabelState.INVALID, InvalidReason.OUTCOME_MIXED
