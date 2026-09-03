#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.extract_full_data_vasofree_waveform_features import (  # noqa: E402
    DEFAULT_FULL_DATA_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_WAVEFORM_ROOT,
    SegmentReader,
    apply_splits,
    build_full_data_anchor_table,
    extract_window,
    shard_anchor_table,
)
from waveform_baselines.wf_features.cache import load_feature_cache  # noqa: E402
from waveform_baselines.wf_features.definitions import feature_names as v7_feature_names  # noqa: E402
from waveform_baselines.wf_features_v8.cache import feature_stats, metadata_payload, subset_feature_cache_by_anchor_ids, validate_v7_v8_alignment  # noqa: E402
from waveform_baselines.wf_features_v8.config import DEFAULT_V8_EXTRACTION_CONFIG, V8_CACHE_ROOT  # noqa: E402
from waveform_baselines.wf_features_v8.definitions import feature_names  # noqa: E402
from waveform_baselines.wf_features_v8.pipeline import (  # noqa: E402
    extract_v8_feature_sequence,
    extract_v8_feature_sequence_cached_global,
    extract_v8_feature_sequence_reference_cached_components,
    extract_v8_reference_component_plan_for_segment,
)


DEFAULT_OUTPUT_NAME = "full_data_vasopressor_free_waveform_features_v8"
DEFAULT_V7_CACHE = V8_CACHE_ROOT / "full" / "v7" / "full_data_vasopressor_free_waveform_features_v7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract supplemental v8 waveform features aligned to frozen v7 caches.")
    parser.add_argument("--full-data-root", type=Path, default=DEFAULT_FULL_DATA_ROOT)
    parser.add_argument("--vasopressor-free-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--waveform-root", type=Path, default=DEFAULT_WAVEFORM_ROOT)
    parser.add_argument("--cache-root", type=Path, default=V8_CACHE_ROOT)
    parser.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--v7-cache", type=Path, default=DEFAULT_V7_CACHE)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--enable-cross-signal-timing", action="store_true")
    parser.add_argument("--enable-abp-advanced-morphology", action="store_true")
    parser.add_argument("--enable-pulse-deficit-features", action="store_true")
    parser.add_argument("--enable-pleth-fiducials", action="store_true")
    parser.add_argument("--enable-pleth-derivative-fiducials", action="store_true")
    parser.add_argument("--enable-systolic-time-features", action="store_true")
    parser.add_argument("--allow-prefix-v7-alignment", action="store_true", help="Permit v8 to match the first N rows of v7 for smoke caches only.")
    parser.add_argument("--use-global-event-cache", action="store_true", help="Use experimental full-input event detection/cache slicing instead of the causal reference extractor.")
    parser.add_argument("--use-reference-component-cache", action="store_true", help="Use exact causal minute/history component caching across anchors in this process.")
    parser.add_argument("--use-reference-segment-component-plan", action="store_true", help="Group anchors by segment and compute each exact causal minute/history component once per segment.")
    return parser.parse_args()


def count_labels(values: np.ndarray) -> dict[str, int]:
    counts = Counter(np.asarray(values, dtype=object).tolist())
    return {str(k): int(v) for k, v in sorted(counts.items())}


def _segment_component_work(group: pd.DataFrame, config, minute_weight_s: float = 0.0240, history_weight_s: float = 0.1190) -> tuple[int, int, float]:
    starts = np.rint(group["window_time"].to_numpy(dtype=np.float64) * config.sampling_rate_hz).astype(np.int64) - config.input_samples // 2
    fw = int(config.feature_window_samples)
    hw = int(config.rolling_history_samples)
    nwin = int(config.n_feature_windows)
    minute_starts = (starts[:, None] + np.arange(nwin, dtype=np.int64)[None, :] * fw).reshape(-1)
    history_offsets = (np.arange(4, nwin, dtype=np.int64) + 1) * fw - hw
    history_starts = (starts[:, None] + history_offsets[None, :]).reshape(-1)
    unique_minute = int(np.unique(minute_starts).size)
    unique_history = int(np.unique(history_starts).size)
    return unique_minute, unique_history, float(unique_minute * minute_weight_s + unique_history * history_weight_s)


def shard_anchor_table_by_segment(anchors: pd.DataFrame, shard_index: int | None, shard_count: int | None, config) -> pd.DataFrame:
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard-index and shard-count must be provided together")
    output_ordered = anchors.sort_values("anchor_id", kind="stable")
    ordered = anchors.sort_values(["segment_id", "window_time", "anchor_id"], kind="stable").reset_index(drop=True)
    if shard_count is None:
        return output_ordered.reset_index(drop=True)
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard {shard_index}/{shard_count}")
    groups = []
    for segment_id, group in ordered.groupby("segment_id", sort=False):
        unique_minute, unique_history, work = _segment_component_work(group, config)
        groups.append((str(segment_id), int(len(group)), unique_minute, unique_history, work))
    if not groups:
        raise ValueError("No segments available for segment-aware sharding")
    bins: list[list[str]] = [[] for _ in range(int(shard_count))]
    loads = np.zeros(int(shard_count), dtype=np.float64)
    for segment_id, _, _, _, work in sorted(groups, key=lambda row: row[4], reverse=True):
        idx = int(np.argmin(loads))
        bins[idx].append(segment_id)
        loads[idx] += float(work)
    selected = set(bins[int(shard_index)])
    shard = output_ordered[output_ordered["segment_id"].astype(str).isin(selected)].copy()
    if shard.empty:
        raise ValueError(f"Segment-aware shard {shard_index}/{shard_count} has no anchors")
    shard.attrs["segment_shard_predicted_work_s"] = float(loads[int(shard_index)])
    shard.attrs["segment_shard_predicted_min_work_s"] = float(np.min(loads))
    shard.attrs["segment_shard_predicted_max_work_s"] = float(np.max(loads))
    shard.attrs["segment_shard_count"] = int(len(selected))
    return shard.reset_index(drop=True)


def validate_no_v7_overlap() -> None:
    overlap = sorted(set(v7_feature_names()).intersection(feature_names()))
    if overlap:
        raise ValueError(f"v8 feature names overlap frozen v7 names: {overlap}")


def validate_against_v7_arrays(cache_dir: Path, v7_cache_path: Path, allow_prefix: bool) -> dict[str, object]:
    v7 = load_feature_cache(v7_cache_path, require_success=True)
    from waveform_baselines.wf_features_v8.cache import load_v8_feature_cache

    v8 = load_v8_feature_cache(cache_dir, require_success=False)
    if v8.metadata.get("shard_index") is not None:
        v7 = subset_feature_cache_by_anchor_ids(v7, v8.anchor_ids)
    elif allow_prefix and v8.values.shape[0] < v7.values.shape[0]:
        from waveform_baselines.wf_features.cache import FeatureCache

        v7 = FeatureCache(
            values=v7.values[: v8.values.shape[0]],
            mask=v7.mask[: v8.values.shape[0]],
            patient_ids=v7.patient_ids[: v8.values.shape[0]],
            anchor_times=v7.anchor_times[: v8.values.shape[0]],
            anchor_ids=v7.anchor_ids[: v8.values.shape[0]],
            split_labels=v7.split_labels[: v8.values.shape[0]],
            feature_names=v7.feature_names,
            metadata=v7.metadata,
            cache_dir=v7.cache_dir,
            segment_ids=v7.segment_ids[: v8.values.shape[0]] if v7.segment_ids is not None else None,
            segment_names=v7.segment_names[: v8.values.shape[0]] if v7.segment_names is not None else None,
        )
    return validate_v7_v8_alignment(v7, v8)


def main() -> None:
    args = parse_args()
    selected_modes = int(args.use_global_event_cache) + int(args.use_reference_component_cache) + int(args.use_reference_segment_component_plan)
    if selected_modes > 1:
        raise ValueError("--use-global-event-cache, --use-reference-component-cache, and --use-reference-segment-component-plan are mutually exclusive")
    validate_no_v7_overlap()
    config = DEFAULT_V8_EXTRACTION_CONFIG
    if args.enable_cross_signal_timing:
        config = config.__class__(**{**config.to_dict(), "enable_cross_signal_timing": True})
    if args.enable_abp_advanced_morphology:
        config = config.__class__(**{**config.to_dict(), "enable_abp_advanced_morphology": True})
    if args.enable_pulse_deficit_features:
        config = config.__class__(**{**config.to_dict(), "enable_pulse_deficit_features": True})
    if args.enable_pleth_fiducials:
        config = config.__class__(**{**config.to_dict(), "enable_pleth_fiducials": True})
    if args.enable_pleth_derivative_fiducials:
        config = config.__class__(**{**config.to_dict(), "enable_pleth_derivative_fiducials": True})
    if args.enable_systolic_time_features:
        config = config.__class__(**{**config.to_dict(), "enable_systolic_time_features": True})

    cache_dir = args.cache_root / config.feature_version / args.output_name
    if cache_dir.exists() and any(cache_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{cache_dir} exists and is nonempty; pass --overwrite")

    anchors = build_full_data_anchor_table(args.full_data_root, args.vasopressor_free_manifest)
    expected_full_n_samples = int(len(anchors))
    anchors = apply_splits(anchors, args.full_data_root / "patient_splits.json")
    if args.max_samples is not None:
        anchors = anchors.iloc[: args.max_samples].copy()
    if args.use_reference_segment_component_plan:
        anchors = shard_anchor_table_by_segment(anchors, args.shard_index, args.shard_count, config)
    else:
        anchors = shard_anchor_table(anchors, args.shard_index, args.shard_count)

    names = feature_names()
    n_samples = int(len(anchors))
    values = np.full((n_samples, config.n_feature_windows, len(names)), np.nan, dtype=np.float32)
    mask = np.zeros((n_samples, config.n_feature_windows, len(names)), dtype=bool)
    reader = SegmentReader(config.channel_order)
    extractor = extract_v8_feature_sequence_cached_global if args.use_global_event_cache else extract_v8_feature_sequence
    minute_component_cache: dict[tuple[object, ...], dict[str, float]] = {}
    history_component_cache: dict[tuple[object, ...], dict[str, float]] = {}
    component_cache_stats: dict[str, int] = {}

    print(json.dumps({
        "event": "waveform_feature_v8_extraction_start",
        "cache_dir": str(cache_dir),
        "n_samples": n_samples,
        "expected_full_n_samples": expected_full_n_samples,
        "feature_count": len(names),
        "enable_cross_signal_timing": config.enable_cross_signal_timing,
        "enable_abp_advanced_morphology": config.enable_abp_advanced_morphology,
        "enable_pulse_deficit_features": config.enable_pulse_deficit_features,
        "enable_pleth_fiducials": config.enable_pleth_fiducials,
        "enable_pleth_derivative_fiducials": config.enable_pleth_derivative_fiducials,
        "enable_systolic_time_features": config.enable_systolic_time_features,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "use_global_event_cache": bool(args.use_global_event_cache),
        "use_reference_component_cache": bool(args.use_reference_component_cache),
        "use_reference_segment_component_plan": bool(args.use_reference_segment_component_plan),
        "segment_shard_predicted_work_s": anchors.attrs.get("segment_shard_predicted_work_s"),
        "segment_shard_predicted_min_work_s": anchors.attrs.get("segment_shard_predicted_min_work_s"),
        "segment_shard_predicted_max_work_s": anchors.attrs.get("segment_shard_predicted_max_work_s"),
        "segment_shard_count": anchors.attrs.get("segment_shard_count"),
    }, sort_keys=True), flush=True)

    if args.use_reference_segment_component_plan:
        processed = 0
        segment_plan_stats: Counter[str] = Counter()
        for segment_id, group in anchors.groupby("segment_id", sort=False):
            first = group.iloc[0]
            segment, segment_fs = reader.load(first)
            if int(round(segment_fs)) != config.sampling_rate_hz:
                raise ValueError(f"Segment {segment_id} fs={segment_fs} does not match expected {config.sampling_rate_hz}")
            input_starts = np.rint(group["window_time"].to_numpy(dtype=np.float64) * config.sampling_rate_hz).astype(np.int64) - config.input_samples // 2
            seq_values, seq_mask, _, stats = extract_v8_reference_component_plan_for_segment(segment, input_starts, config=config)
            group_positions = group.index.to_numpy(dtype=np.int64)
            values[group_positions] = seq_values
            mask[group_positions] = seq_mask
            segment_plan_stats.update({key: int(value) for key, value in stats.items() if key.endswith(("requested", "computed", "hits")) or key == "samples"})
            segment_plan_stats["segments"] += 1
            segment_plan_stats["minute_cache_value_bytes_peak"] = max(int(segment_plan_stats.get("minute_cache_value_bytes_peak", 0)), int(stats.get("minute_cache_value_bytes", 0)))
            segment_plan_stats["history_cache_value_bytes_peak"] = max(int(segment_plan_stats.get("history_cache_value_bytes_peak", 0)), int(stats.get("history_cache_value_bytes", 0)))
            processed += int(len(group))
            if processed % 100 <= len(group):
                print(json.dumps({"event": "progress", "processed": processed, "n_samples": n_samples, "segment_plan_stats": dict(segment_plan_stats)}, sort_keys=True), flush=True)
        component_cache_stats.update({f"segment_plan_{key}": int(value) for key, value in segment_plan_stats.items()})
    else:
        for out_idx, row in enumerate(anchors.itertuples(index=False)):
            row_series = pd.Series(row._asdict())
            waveform = extract_window(row_series, reader, config.input_samples, config.sampling_rate_hz)
            if args.use_reference_component_cache:
                anchor_sample = int(round(float(row_series["window_time"]) * float(config.sampling_rate_hz)))
                input_start_sample = anchor_sample - config.input_samples // 2
                seq_values, seq_mask, _ = extract_v8_feature_sequence_reference_cached_components(
                    waveform,
                    input_start_sample=input_start_sample,
                    config=config,
                    minute_cache=minute_component_cache,
                    history_cache=history_component_cache,
                    cache_key_prefix=str(row_series["segment_id"]),
                    cache_stats=component_cache_stats,
                )
            else:
                seq_values, seq_mask, _ = extractor(waveform, config=config)
            values[out_idx] = seq_values
            mask[out_idx] = seq_mask
            if (out_idx + 1) % 100 == 0:
                progress = {"event": "progress", "processed": out_idx + 1, "n_samples": n_samples}
                if args.use_reference_component_cache:
                    progress["component_cache_stats"] = dict(component_cache_stats)
                    progress["minute_cache_size"] = len(minute_component_cache)
                    progress["history_cache_size"] = len(history_component_cache)
                print(json.dumps(progress, sort_keys=True), flush=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "values.npy", values)
    np.save(cache_dir / "mask.npy", mask)
    np.save(cache_dir / "patient_ids.npy", anchors["patient_id"].astype(str).to_numpy())
    np.save(cache_dir / "anchor_times.npy", anchors["window_time"].to_numpy(dtype=np.float64))
    np.save(cache_dir / "anchor_ids.npy", anchors["anchor_id"].to_numpy(dtype=np.int64))
    np.save(cache_dir / "split_labels.npy", anchors["split_label"].astype(str).to_numpy())
    np.save(cache_dir / "segment_ids.npy", anchors["segment_id"].astype(str).to_numpy())
    np.save(cache_dir / "segment_names.npy", anchors["seg_name"].astype(str).to_numpy())
    anchors[["anchor_id", "patient_id", "segment_id", "seg_name", "window_time", "split_label", "segment_path"]].to_csv(cache_dir / "anchors.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    metadata = metadata_payload({
        "extraction_config": config.to_dict(),
        "channel_order": list(config.channel_order),
        "sampling_rate_hz": config.sampling_rate_hz,
        "input_window_seconds": config.input_window_seconds,
        "feature_window_seconds": config.feature_window_seconds,
        "rolling_history_seconds": config.rolling_history_seconds,
        "full_data_root": str(args.full_data_root),
        "vasopressor_free_manifest": str(args.vasopressor_free_manifest),
        "waveform_root": str(args.waveform_root),
        "v7_cache": str(args.v7_cache),
        "n_samples": n_samples,
        "expected_full_n_samples": expected_full_n_samples,
        "split_sample_counts": count_labels(anchors["split_label"].to_numpy()),
        "split_patient_counts": count_labels(anchors.drop_duplicates("patient_id")["split_label"].to_numpy()),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_assignment": "segment_aware_greedy_unique_component_work" if args.use_reference_segment_component_plan else "contiguous_sorted_anchor_id_chunks",
        "is_full_data_segment_window_cache": True,
        "tier2_status": "cross-signal timing, pulse deficits, basic PLETH fiducials, PLETH derivative fiducials, advanced ABP morphology, and systolic-time features disabled unless explicitly enabled after audit",
        "use_global_event_cache": bool(args.use_global_event_cache),
        "use_reference_component_cache": bool(args.use_reference_component_cache),
        "use_reference_segment_component_plan": bool(args.use_reference_segment_component_plan),
        "event_cache_status": "experimental full-input event detection/cache slicing" if args.use_global_event_cache else "causal reference detector-per-window extraction",
        "reference_component_cache_status": "planned segment-local exact causal minute/history component extraction" if args.use_reference_segment_component_plan else ("exact causal minute/history component caching" if args.use_reference_component_cache else "disabled"),
        "reference_component_cache_stats": dict(component_cache_stats),
        "reference_component_minute_cache_size": len(minute_component_cache),
        "reference_component_history_cache_size": len(history_component_cache),
    })
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    quality = feature_stats(values, mask, names)
    (cache_dir / "feature_quality_report.json").write_text(json.dumps(quality, indent=2))
    alignment = validate_against_v7_arrays(cache_dir, args.v7_cache, allow_prefix=args.allow_prefix_v7_alignment)
    (cache_dir / "alignment_report.json").write_text(json.dumps(alignment, indent=2))
    (cache_dir / "_SUCCESS").write_text(f"completed_at_unix={time.time():.6f}\n")
    if args.use_reference_component_cache or args.use_reference_segment_component_plan:
        print(json.dumps({
            "event": "waveform_feature_v8_component_cache_summary",
            "component_cache_stats": component_cache_stats,
            "minute_cache_size": len(minute_component_cache),
            "history_cache_size": len(history_component_cache),
        }, sort_keys=True), flush=True)
    print(json.dumps({"cache_dir": str(cache_dir), "shape": list(values.shape), "alignment": alignment}, indent=2), flush=True)


if __name__ == "__main__":
    main()
