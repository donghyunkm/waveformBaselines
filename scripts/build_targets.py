#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waveform_baselines.target_builders import (
    ICUExtractionPaths,
    build_event_targets,
    build_feature_regression_targets,
    load_anchor_table,
    save_target_bundle,
)
from waveform_baselines.task_specs import (
    DEFAULT_EVENT_TASK,
    DEFAULT_FEATURE_TASK,
    EventTaskSpec,
    FeatureRegressionTaskSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supervision targets for waveform baselines.")
    parser.add_argument("--anchors-csv", type=Path, help="CSV with patient_id, anchor_time, and optional input_start_time.")
    parser.add_argument(
        "--use-waveform-anchors",
        action="store_true",
        help="Use icuDataExtraction 20-minute waveform anchors directly. Useful for aligned feature-target smoke tests.",
    )
    parser.add_argument(
        "--icu-output-dir",
        type=Path,
        default=ICUExtractionPaths().output_dir,
        help="Path to icuDataExtraction/output_v2.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .npz bundle path.")
    parser.add_argument("--skip-feature-targets", action="store_true")
    parser.add_argument("--skip-event-targets", action="store_true")
    parser.add_argument(
        "--feature-horizon-mode",
        type=str,
        default=DEFAULT_FEATURE_TASK.horizon_mode,
        choices=["center", "gap"],
        help="Regression horizon semantics: center-to-center or gap after input window.",
    )
    parser.add_argument(
        "--event-horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVENT_TASK.horizons_min),
        help="Event-classification horizons in minutes. Default uses the repository event spec.",
    )
    parser.add_argument(
        "--event-target-generation-mode",
        type=str,
        default=DEFAULT_EVENT_TASK.target_generation_mode,
        choices=["anchor_horizon", "anchor_horizon_filtered"],
        help="Anchor-horizon baseline or anchor-horizon with filtered hypotension negatives.",
    )
    return parser.parse_args()


def _waveform_anchor_table(icu_output_dir: Path) -> pd.DataFrame:
    import numpy as np

    patient_ids = np.load(icu_output_dir / "patient_ids.npy", allow_pickle=True).astype(str)
    anchor_times = np.load(icu_output_dir / "window_times.npy").astype(float)
    return pd.DataFrame({"patient_id": patient_ids, "anchor_time": anchor_times})


def main() -> None:
    args = parse_args()
    if args.use_waveform_anchors == bool(args.anchors_csv):
        raise ValueError("Specify exactly one of --anchors-csv or --use-waveform-anchors.")

    if args.use_waveform_anchors:
        anchors = _waveform_anchor_table(args.icu_output_dir)
    else:
        anchors = load_anchor_table(args.anchors_csv)

    feature_spec = FeatureRegressionTaskSpec(
        horizons_min=DEFAULT_FEATURE_TASK.horizons_min,
        horizon_mode=args.feature_horizon_mode,
        input_window_minutes=DEFAULT_FEATURE_TASK.input_window_minutes,
        feature_names=DEFAULT_FEATURE_TASK.feature_names,
        correlation_names=DEFAULT_FEATURE_TASK.correlation_names,
        aggregation=DEFAULT_FEATURE_TASK.aggregation,
    )
    event_spec = EventTaskSpec(
        horizons_min=tuple(args.event_horizons),
        target_generation_mode=args.event_target_generation_mode,
        hypotension_threshold=DEFAULT_EVENT_TASK.hypotension_threshold,
        tachycardia_threshold=DEFAULT_EVENT_TASK.tachycardia_threshold,
        sustain_minutes=DEFAULT_EVENT_TASK.sustain_minutes,
        hypotension_channel=DEFAULT_EVENT_TASK.hypotension_channel,
        tachycardia_channel=DEFAULT_EVENT_TASK.tachycardia_channel,
        event_names=DEFAULT_EVENT_TASK.event_names,
    )

    feature_targets = None
    feature_mask = None
    if not args.skip_feature_targets:
        feature_targets, feature_mask = build_feature_regression_targets(
            anchors,
            args.icu_output_dir,
            task_spec=feature_spec,
        )

    event_targets = None
    event_mask = None
    event_auxiliary_arrays = None
    event_diagnostics = None
    if not args.skip_event_targets:
        event_result = build_event_targets(
            anchors,
            args.icu_output_dir,
            task_spec=event_spec,
        )
        event_targets = event_result.targets
        event_mask = event_result.mask
        event_auxiliary_arrays = event_result.auxiliary_arrays
        event_diagnostics = event_result.diagnostics

    save_target_bundle(
        output_path=args.output,
        anchors=anchors,
        feature_targets=feature_targets,
        feature_mask=feature_mask,
        event_targets=event_targets,
        event_mask=event_mask,
        feature_spec=feature_spec,
        event_spec=event_spec,
        event_auxiliary_arrays=event_auxiliary_arrays,
        event_diagnostics=event_diagnostics,
    )


if __name__ == "__main__":
    main()
