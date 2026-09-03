from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventEpisode:
    start_minute_index: int
    end_minute_index_exclusive: int
    duration_minutes: int


def detect_maximal_episodes(
    minute_valid: np.ndarray,
    event_minutes: np.ndarray,
    sustain_minutes: int,
) -> list[EventEpisode]:
    if sustain_minutes <= 0:
        raise ValueError("sustain_minutes must be positive")
    if len(minute_valid) != len(event_minutes):
        raise ValueError("minute_valid and event_minutes must have the same length")

    valid = np.asarray(minute_valid, dtype=bool)
    events = np.asarray(event_minutes, dtype=bool)
    episodes: list[EventEpisode] = []
    run_start: int | None = None

    for idx in range(len(events) + 1):
        in_event = idx < len(events) and bool(valid[idx]) and bool(events[idx])
        if in_event and run_start is None:
            run_start = idx
        if not in_event and run_start is not None:
            run_end = idx
            duration = run_end - run_start
            if duration >= sustain_minutes:
                episodes.append(
                    EventEpisode(
                        start_minute_index=int(run_start),
                        end_minute_index_exclusive=int(run_end),
                        duration_minutes=int(duration),
                    )
                )
            run_start = None

    return episodes


def episode_overlaps_interval(
    episode: EventEpisode,
    interval_start: int,
    interval_end: int,
) -> bool:
    return (
        episode.start_minute_index < interval_end
        and episode.end_minute_index_exclusive > interval_start
    )


def episode_start_in_interval(
    episode: EventEpisode,
    interval_start: int,
    interval_end: int,
) -> bool:
    return interval_start <= episode.start_minute_index < interval_end


def episode_starts_at(episodes: list[EventEpisode], minute_index: int) -> bool:
    return any(ep.start_minute_index == minute_index for ep in episodes)


def any_episode_overlaps(
    episodes: list[EventEpisode],
    interval_start: int,
    interval_end: int,
) -> bool:
    return any(episode_overlaps_interval(ep, interval_start, interval_end) for ep in episodes)


def any_episode_start_in_interval(
    episodes: list[EventEpisode],
    interval_start: int,
    interval_end: int,
) -> bool:
    return any(episode_start_in_interval(ep, interval_start, interval_end) for ep in episodes)
