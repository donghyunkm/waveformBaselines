from __future__ import annotations

import numpy as np
import pandas as pd

from waveform_baselines.target_builders import (
    build_event_classification_targets,
    build_event_targets,
)
from waveform_baselines.task_specs import EventTaskSpec


def _make_numerics_windows(
    minute_values: list[float | None],
    patient_id: str,
    center_time: int = 600,
    context_seconds: int = 1200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerics = np.full((1, 5, context_seconds), np.nan, dtype=np.float32)
    for minute_idx, value in enumerate(minute_values):
        if value is None:
            continue
        start = minute_idx * 60
        end = start + 60
        numerics[0, 0, start:end] = np.float32(value)
    return (
        np.asarray([patient_id]),
        np.asarray([float(center_time)], dtype=np.float64),
        numerics,
    )


def _patch_numerics(monkeypatch, rows):
    patient_ids = np.concatenate([row[0] for row in rows])
    patient_times = np.concatenate([row[1] for row in rows])
    numerics = np.concatenate([row[2] for row in rows], axis=0)
    monkeypatch.setattr(
        "waveform_baselines.target_builders.load_numerics_windows",
        lambda _: (patient_ids, patient_times, numerics),
    )


def test_anchor_horizon_requires_stable_input(monkeypatch):
    anchors = pd.DataFrame(
        {
            "patient_id": ["stay_a"],
            "anchor_time": [1200],
            "input_start_time": [0],
            "input_end_time": [300],
        }
    )
    row = _make_numerics_windows([64.0, 64.0, 64.0, 64.0, 64.0], patient_id="stay_a")
    _patch_numerics(monkeypatch, [row])

    targets, mask = build_event_classification_targets(anchors, "unused")
    assert mask[:, 0].tolist() == [False]
    assert targets[:, 0].tolist() == [-1]


def test_anchor_horizon_labels_future_hypotension_and_tachycardia(monkeypatch):
    anchors = pd.DataFrame(
        {
            "patient_id": ["stay_a"],
            "anchor_time": [1200],
            "input_start_time": [0],
            "input_end_time": [0],
        }
    )
    _, patient_times, numerics = _make_numerics_windows([80.0] * 10, patient_id="stay_a")
    numerics[0, 0, :300] = 64.0
    numerics[0, 4, :300] = 150.0
    _patch_numerics(monkeypatch, [(np.asarray(["stay_a"]), patient_times, numerics)])

    task_spec = EventTaskSpec(horizons_min=(5,))
    targets, mask = build_event_classification_targets(anchors, "unused", task_spec=task_spec)
    assert mask[:, :2].tolist() == [[True, True]]
    assert targets[:, :2].tolist() == [[1, 1]]


def test_anchor_horizon_filtered_drops_negatives_from_positive_recordings_and_late_times(monkeypatch):
    anchors = pd.DataFrame(
        {
            "patient_id": ["stay_pos_a", "stay_pos_b", "stay_neg_keep", "stay_neg_late"],
            "anchor_time": [1200, 1200, 1200, 1200],
            "input_start_time": [0, 0, 0, 0],
            "input_end_time": [0, 0, 0, 480],
        }
    )
    rows = [
        _make_numerics_windows([80.0] * 10 + [64.0] * 5, patient_id="stay_pos_a"),
        _make_numerics_windows([80.0] * 15 + [64.0] * 5, patient_id="stay_pos_b"),
        _make_numerics_windows([80.0] * 20, patient_id="stay_neg_keep"),
        _make_numerics_windows([80.0] * 20, patient_id="stay_neg_late"),
    ]
    _patch_numerics(monkeypatch, rows)

    task_spec = EventTaskSpec(horizons_min=(5,), target_generation_mode="anchor_horizon_filtered")
    result = build_event_targets(anchors, "unused", task_spec=task_spec)

    assert result.targets[:, 0].tolist() == [-1, -1, 0, -1]
    assert result.mask[:, 0].tolist() == [False, False, True, False]
    assert result.diagnostics["per_horizon"]["5"]["retained_negative_candidates"] == 1
    assert result.diagnostics["per_horizon"]["5"]["dropped_negative_positive_recording"] == 2
    assert result.diagnostics["per_horizon"]["5"]["dropped_negative_after_mean_last_positive_event"] == 1


def test_anchor_horizon_filtered_requires_clean_valid_outcome_window(monkeypatch):
    anchors = pd.DataFrame(
        {
            "patient_id": ["stay_dirty", "stay_missing", "stay_clean"],
            "anchor_time": [1200, 1200, 1200],
            "input_start_time": [0, 0, 0],
            "input_end_time": [0, 0, 0],
        }
    )
    rows = [
        _make_numerics_windows([80.0] * 5 + [64.0] + [80.0] * 14, patient_id="stay_dirty"),
        _make_numerics_windows([80.0] * 5 + [None] + [80.0] * 14, patient_id="stay_missing"),
        _make_numerics_windows([80.0] * 20, patient_id="stay_clean"),
    ]
    _patch_numerics(monkeypatch, rows)

    task_spec = EventTaskSpec(horizons_min=(5,), target_generation_mode="anchor_horizon_filtered")
    result = build_event_targets(anchors, "unused", task_spec=task_spec)

    assert result.targets[:, 0].tolist() == [-1, -1, 0]
    assert result.mask[:, 0].tolist() == [False, False, True]
    assert result.diagnostics["per_horizon"]["5"]["dropped_negative_outcome_window"] == 2


def test_build_event_targets_rejects_unknown_mode(monkeypatch):
    anchors = pd.DataFrame(
        {
            "patient_id": ["stay_a"],
            "anchor_time": [1200],
            "input_start_time": [0],
            "input_end_time": [0],
        }
    )
    row = _make_numerics_windows([80.0] * 10, patient_id="stay_a")
    _patch_numerics(monkeypatch, [row])

    task_spec = EventTaskSpec(horizons_min=(5,), target_generation_mode="unsupported")
    try:
        build_event_targets(anchors, "unused", task_spec=task_spec)
    except ValueError as exc:
        assert "Unsupported event target_generation_mode" in str(exc)
    else:
        raise AssertionError("Expected unsupported event target_generation_mode to raise")
