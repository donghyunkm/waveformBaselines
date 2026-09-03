#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waveform_baselines.task_specs import DEFAULT_EVENT_TASK, EventTaskSpec
from waveform_baselines.anchor_labeling import (
    AnchorMinuteBounds,
    InvalidReason,
    LabelState,
    NegativeFilterReason,
    label_fixed_forecast_window,
    label_onset_within_horizon,
)
from waveform_baselines.event_episodes import (
    EventEpisode,
    any_episode_overlaps,
    detect_maximal_episodes,
)
from waveform_baselines.event_timeline import (
    MAX_ALIGNMENT_RESIDUAL_SECONDS,
    MinuteTimeline,
    build_minute_timeline_from_windows,
    resolve_channel,
    summarize_numeric,
    timestamp_to_minute_index,
)
from waveform_baselines.target_builders import (
    _build_minute_event_labels,
    _detect_sustained_event_starts,
    _window_has_only_clean_minutes,
    save_target_bundle,
)

DEFAULT_FULL_DATA_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/physionet_restricted/mimic_derived_data/data_m3_120s_prediction")
DEFAULT_CACHE_DIR = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/full/v7/full_data_vasopressor_free_waveform_features_v7")
DEFAULT_WAVEFORM_ROOT = Path("/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched")
DEFAULT_NUMERICS_DIR = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/numerics/full_data_v1")
DEFAULT_OUTPUT = Path("outputs/targets/event_targets_full_data_anchor_onset_v2_5m_10m_recording_complete_scan_filtered.npz")
REF_DT = dt.datetime(2000, 1, 1)
VITAL_NAMES = ["ABP Mean", "PULSE", "SpO2", "RESP", "HR"]
VITAL_ALIASES = {
    "ABP Mean": ["ABPMean", "ART Mean", "ARTMean"],
    "PULSE": [],
    "SpO2": ["%SpO2"],
    "RESP": [],
    "HR": [],
}
TIME_TOLERANCE_SECONDS = 1.0
ALIGNMENT_QUANTILES = (0, 50, 90, 95, 99, 100)



@dataclass(frozen=True)
class NumericsRecord:
    path: Path
    start_epoch: float
    end_epoch: float
    fs: float
    sig_len: int
    sig_indices: dict[int, int | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-data event target bundles aligned to the full-data v7 feature cache.")
    parser.add_argument("--full-data-root", type=Path, default=DEFAULT_FULL_DATA_ROOT)
    parser.add_argument("--feature-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--waveform-root", type=Path, default=DEFAULT_WAVEFORM_ROOT)
    parser.add_argument("--numerics-dir", type=Path, default=DEFAULT_NUMERICS_DIR, help="Directory containing row-aligned X_numerics.npy; used with the production aligned-array target path.")
    parser.add_argument("--numerics-source", choices=["waveform-records", "aligned-array"], default="aligned-array")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-horizons", type=int, nargs="+", default=[5, 10], help="Horizons for anchor_onset_within_horizon. Output columns follow this order.")
    parser.add_argument("--forecast-gaps", type=int, nargs="+", default=[5, 10], help="Forecast gaps for anchor_fixed_forecast_window.")
    parser.add_argument(
        "--target-mode",
        choices=["anchor_onset_within_horizon", "anchor_fixed_forecast_window", "legacy_anchor_horizon_filtered", "anchor_horizon", "anchor_horizon_filtered"],
        default="anchor_onset_within_horizon",
    )
    parser.add_argument("--event-target-generation-mode", choices=["anchor_horizon", "anchor_horizon_filtered"], default=None, help="Deprecated legacy alias.")
    parser.add_argument("--events", nargs="+", choices=["hypotension", "tachycardia", "hypoxia"], default=["hypotension"])
    parser.add_argument("--numerics-window-time-basis", choices=["absolute", "segment-relative"], default="absolute")
    parser.add_argument("--aligned-time-basis", choices=["segment-relative", "absolute"], default="absolute")
    parser.add_argument("--timestamp-alignment-tolerance-seconds", type=float, default=1.0)
    parser.add_argument("--validate-only", action="store_true", help="Run timestamp/numeric preflight checks and exit without saving labels.")
    parser.add_argument("--hypotension-definition", choices=["map-only"], default="map-only")
    parser.add_argument("--sustain-minutes", type=int, default=5)
    parser.add_argument("--negative-policy", choices=["observable-no-onset", "strict-clean-horizon", "clean-fixed-window"], default="strict-clean-horizon")
    parser.add_argument("--negative-exclusion-scope", choices=["none", "segment", "recording", "icu-stay", "patient"], default="none")
    parser.add_argument("--apply-late-negative-cutoff", action="store_true", default=True)
    parser.add_argument("--no-late-negative-cutoff", action="store_false", dest="apply_late_negative_cutoff")
    parser.add_argument("--late-cutoff-candidate", choices=["input_end", "forecast_endpoint"], default="forecast_endpoint")
    parser.add_argument("--late-cutoff-group-scope", choices=["segment", "recording", "icu-stay", "patient"], default="recording", help="Grouping used only for late negative cutoff; defaults to --negative-exclusion-scope when omitted.")
    parser.add_argument("--late-cutoff-strategy", choices=["mean-last-positive", "group-last-positive"], default="group-last-positive", help="mean-last-positive preserves the legacy averaged cutoff; group-last-positive filters negatives after the last positive within each cutoff group.")
    parser.add_argument("--exclude-late-cutoff-groups-without-positives", action="store_true", default=True, help="With group-last-positive late cutoff, remove base negatives from cutoff groups that have no positive episode. Enabled by default for the current filtered target build.")
    parser.add_argument("--include-late-cutoff-groups-without-positives", action="store_false", dest="exclude_late_cutoff_groups_without_positives", help="Opt out of excluding cutoff groups that have no positive episode.")
    parser.add_argument("--min-valid-fraction-per-minute", type=float, default=1.0 / 60.0)
    parser.add_argument("--allow-partial-output", action="store_true")
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--legacy-target-bundle", type=Path, default=Path("outputs/targets/event_targets_full_data_anchor_horizon_filtered_5m_10m.npz"))
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--max-segments", type=int, default=None, help="Debug/smoke-test limit; omit for the full target bundle.")
    return parser.parse_args()


def source_record_name_from_segment_name(seg_name: object) -> str:
    name = str(seg_name)
    prefix, sep, suffix = name.rpartition("_")
    if sep and suffix.isdigit() and prefix:
        return prefix
    return name


def waveform_record_id_from_anchor(row: pd.Series) -> str:
    patient_id = str(row["patient_id"])
    return f"{patient_id}/{source_record_name_from_segment_name(row['seg_name'])}"


def waveform_record_id_from_segment_id(segment_id: object) -> str:
    value = str(segment_id)
    if "/" not in value:
        return source_record_name_from_segment_name(value)
    patient_id, seg_name = value.split("/", 1)
    return f"{patient_id}/{source_record_name_from_segment_name(seg_name)}"


def load_segment_start_seconds_by_id(full_data_root: Path) -> dict[str, float]:
    metadata = pd.DataFrame(json.loads((full_data_root / "segment_metadata.json").read_text()))
    required = {"patient_id", "seg_name", "seg_start_secs"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"segment_metadata.json missing columns: {sorted(missing)}")
    metadata = metadata.drop_duplicates(["patient_id", "seg_name"], keep="first")
    return {
        f"{str(row.patient_id)}/{str(row.seg_name)}": float(row.seg_start_secs)
        for row in metadata.itertuples(index=False)
    }


def load_cache_anchors(cache_dir: Path, full_data_root: Path) -> pd.DataFrame:
    anchors_path = cache_dir / "anchors.csv"
    anchor_ids_path = cache_dir / "anchor_ids.npy"
    anchor_times_path = cache_dir / "anchor_times.npy"
    values_path = cache_dir / "values.npy"
    for path in (anchors_path, anchor_ids_path, anchor_times_path, values_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing full-data feature-cache file: {path}")

    anchors = pd.read_csv(anchors_path)
    required = {"anchor_id", "patient_id", "segment_id", "seg_name", "window_time", "split_label"}
    missing = required.difference(anchors.columns)
    if missing:
        raise ValueError(f"Feature-cache anchors missing columns: {sorted(missing)}")
    if anchors["anchor_id"].duplicated().any():
        dupes = anchors.loc[anchors["anchor_id"].duplicated(), "anchor_id"].head(5).tolist()
        raise ValueError(f"anchors.csv contains duplicate anchor_id values, examples={dupes}")

    cache_anchor_ids = np.asarray(np.load(anchor_ids_path, mmap_mode="r"), dtype=np.int64)
    cache_anchor_times = np.asarray(np.load(anchor_times_path, mmap_mode="r"), dtype=np.float64)
    values = np.load(values_path, mmap_mode="r")
    if len(cache_anchor_ids) != len(cache_anchor_times):
        raise ValueError("len(anchor_ids.npy) does not match len(anchor_times.npy)")
    if values.shape[0] != len(cache_anchor_ids):
        raise ValueError("values.npy row count does not match anchor_ids.npy row count")
    if len(anchors) != len(cache_anchor_ids):
        raise ValueError("anchors.csv row count does not match feature-cache anchor_ids.npy row count")
    if pd.Index(cache_anchor_ids).duplicated().any():
        raise ValueError("anchor_ids.npy contains duplicate anchor IDs")
    if not np.all(np.isfinite(cache_anchor_times)):
        raise ValueError("Feature-cache anchor timestamps contain non-finite values")

    metadata = pd.DataFrame(json.loads((full_data_root / "segment_metadata.json").read_text()))
    metadata = metadata.drop_duplicates(["patient_id", "seg_name"], keep="first")
    anchors = anchors.merge(metadata[["patient_id", "seg_name", "seg_start_secs"]], on=["patient_id", "seg_name"], how="left")
    if anchors["seg_start_secs"].isna().any():
        raise ValueError("Some feature-cache anchors did not match segment_metadata.json")

    anchor_time_by_id = pd.Series(cache_anchor_times, index=cache_anchor_ids)
    anchors = anchors.copy()
    anchors["feature_cache_anchor_time_raw"] = anchors["anchor_id"].map(anchor_time_by_id)
    if anchors["feature_cache_anchor_time_raw"].isna().any():
        missing_ids = anchors.loc[anchors["feature_cache_anchor_time_raw"].isna(), "anchor_id"].head(10).tolist()
        raise ValueError(f"anchors.csv anchor_id values missing from anchor_ids.npy, examples={missing_ids}")

    window_time = anchors["window_time"].to_numpy(dtype=np.float64)
    seg_start = anchors["seg_start_secs"].to_numpy(dtype=np.float64)
    raw_cache_time = anchors["feature_cache_anchor_time_raw"].to_numpy(dtype=np.float64)
    cache_time_absolute_plausible = raw_cache_time >= (seg_start - TIME_TOLERANCE_SECONDS)
    cache_time_local_plausible = raw_cache_time < (seg_start - TIME_TOLERANCE_SECONDS)
    absolute_cache_fraction = float(cache_time_absolute_plausible.mean()) if len(raw_cache_time) else 0.0
    local_cache_fraction = float(cache_time_local_plausible.mean()) if len(raw_cache_time) else 0.0
    if absolute_cache_fraction >= 0.999:
        feature_cache_anchor_time_basis = "absolute"
        canonical = raw_cache_time
    elif local_cache_fraction >= 0.999:
        feature_cache_anchor_time_basis = "segment-relative"
        canonical = raw_cache_time + seg_start
    else:
        examples = anchors[["anchor_id", "window_time", "seg_start_secs", "feature_cache_anchor_time_raw"]].head(10).to_dict("records")
        raise ValueError(
            "feature-cache anchor_times.npy basis is ambiguous relative to segment starts; "
            f"absolute_plausible_fraction={absolute_cache_fraction:.6f}, local_plausible_fraction={local_cache_fraction:.6f}, examples={examples}"
        )

    anchors["canonical_anchor_time"] = canonical
    matches_absolute = np.isclose(window_time, canonical, atol=TIME_TOLERANCE_SECONDS, rtol=0.0)
    matches_local = np.isclose(window_time + seg_start, canonical, atol=TIME_TOLERANCE_SECONDS, rtol=0.0)
    absolute_fraction = float(matches_absolute.mean()) if len(matches_absolute) else 0.0
    local_fraction = float(matches_local.mean()) if len(matches_local) else 0.0
    if absolute_fraction >= 0.999:
        window_time_basis = "absolute"
    elif local_fraction >= 0.999:
        window_time_basis = "segment-relative"
    else:
        examples = anchors.loc[~(matches_absolute | matches_local), ["anchor_id", "window_time", "seg_start_secs", "canonical_anchor_time", "feature_cache_anchor_time_raw"]].head(10).to_dict("records")
        raise ValueError(
            "anchors.csv window_time matches neither absolute nor segment-relative canonical feature-cache anchor time; "
            f"absolute_match_fraction={absolute_fraction:.6f}, local_match_fraction={local_fraction:.6f}, examples={examples}"
        )

    anchors = anchors.set_index("anchor_id", drop=False).loc[cache_anchor_ids].reset_index(drop=True)
    anchors["anchor_time_absolute"] = anchors["canonical_anchor_time"].astype(np.float64)
    anchors["anchor_time_local"] = anchors["anchor_time_absolute"] - anchors["seg_start_secs"].astype(np.float64)
    anchors["input_start_time_absolute"] = anchors["anchor_time_absolute"] - 600.0
    anchors["input_end_time_absolute"] = anchors["anchor_time_absolute"] + 600.0
    anchors["input_start_time_local"] = anchors["anchor_time_local"] - 600.0
    anchors["input_end_time_local"] = anchors["anchor_time_local"] + 600.0
    anchors["anchor_time"] = anchors["anchor_time_absolute"]
    anchors["input_start_time"] = anchors["input_start_time_absolute"]
    anchors["input_end_time"] = anchors["input_end_time_absolute"]
    anchors["absolute_anchor_time"] = anchors["anchor_time_absolute"]
    duration = anchors["input_end_time_absolute"] - anchors["input_start_time_absolute"]
    if not np.allclose(duration.to_numpy(dtype=np.float64), 1200.0, atol=1e-6, rtol=0.0):
        raise ValueError("Anchor input intervals are not all exactly 20 minutes")
    anchors["source_record_name"] = anchors["seg_name"].map(source_record_name_from_segment_name).astype(str)
    anchors["waveform_record_id"] = anchors["patient_id"].astype(str) + "/" + anchors["source_record_name"]
    if anchors["waveform_record_id"].isna().any() or (anchors["waveform_record_id"].astype(str).str.len() == 0).any():
        raise ValueError("Could not derive waveform_record_id for all anchors")
    anchors["negative_group_id"] = anchors["waveform_record_id"].astype(str)
    anchors.attrs["timestamp_metadata"] = {
        "feature_cache_anchor_time_basis": feature_cache_anchor_time_basis,
        "feature_cache_anchor_time_absolute_plausible_fraction": absolute_cache_fraction,
        "feature_cache_anchor_time_local_plausible_fraction": local_cache_fraction,
        "anchors_csv_window_time_basis": window_time_basis,
        "anchors_csv_window_time_absolute_match_fraction": absolute_fraction,
        "anchors_csv_window_time_local_match_fraction": local_fraction,
        "canonical_bundle_time_basis": "absolute",
        "feature_cache_dir": str(cache_dir),
        "feature_cache_anchors": str(anchors_path),
        "feature_cache_anchor_ids": str(anchor_ids_path),
        "feature_cache_anchor_times": str(anchor_times_path),
        "feature_cache_values": str(values_path),
        "feature_cache_row_count": int(values.shape[0]),
        "waveform_record_id_rule": "patient_id/source_record_name, where source_record_name strips the final numeric chunk suffix from seg_name",
        "n_waveform_records": int(anchors["waveform_record_id"].nunique()),
        "n_waveform_segments": int(anchors["segment_id"].nunique()),
    }
    return anchors


def load_full_data_identity(full_data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    patient_ids = np.load(full_data_root / "patient_ids.npy", mmap_mode="r", allow_pickle=True).astype(str)
    seg_names = np.load(full_data_root / "seg_names.npy", mmap_mode="r", allow_pickle=True).astype(str)
    segment_ids = np.char.add(np.char.add(patient_ids.astype(str), "/"), seg_names.astype(str))
    window_times = np.asarray(np.load(full_data_root / "window_times.npy", mmap_mode="r"), dtype=np.float64)
    return segment_ids, window_times


def load_aligned_numerics(numerics_dir: Path, full_data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerics_path = numerics_dir / "X_numerics.npy"
    if not numerics_path.exists():
        raise FileNotFoundError(f"Missing row-aligned full-data numerics array: {numerics_path}")
    numerics = np.load(numerics_path, mmap_mode="r")
    if (numerics_dir / "numerics_seg_names.npy").exists() and (numerics_dir / "numerics_window_times.npy").exists():
        patient_ids = np.load(numerics_dir / "numerics_patient_ids.npy", mmap_mode="r", allow_pickle=True).astype(str)
        seg_names = np.load(numerics_dir / "numerics_seg_names.npy", mmap_mode="r", allow_pickle=True).astype(str)
        segment_ids = np.char.add(np.char.add(patient_ids.astype(str), "/"), seg_names.astype(str))
        window_times = np.asarray(np.load(numerics_dir / "numerics_window_times.npy", mmap_mode="r"), dtype=np.float64)
    else:
        segment_ids, window_times = load_full_data_identity(full_data_root)
    if numerics.shape[0] != len(segment_ids) or len(segment_ids) != len(window_times):
        raise ValueError("Numerics rows are not aligned with full-data segment/window identity arrays")
    if numerics.ndim != 3:
        raise ValueError(f"Expected X_numerics with shape (N, channels, samples), got {numerics.shape}")
    return segment_ids, window_times, numerics


def window_interval(center_time: float, n_samples: int) -> tuple[int, int]:
    start = int(round(center_time - n_samples / 2.0))
    return start, start + int(n_samples)


def interval_values(candidate_times: np.ndarray, candidate_windows: np.ndarray, start: int, end: int, channel_idx: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(end - start, np.nan, dtype=np.float32)
    valid = np.zeros(end - start, dtype=bool)
    for center_time, window in zip(candidate_times.tolist(), candidate_windows):
        w_start, w_end = window_interval(float(center_time), window.shape[1])
        overlap_start = max(start, w_start)
        overlap_end = min(end, w_end)
        if overlap_start >= overlap_end:
            continue
        sample_start = overlap_start - w_start
        sample_end = overlap_end - w_start
        out_start = overlap_start - start
        out_end = overlap_end - start
        arr = np.asarray(window[channel_idx, sample_start:sample_end], dtype=np.float32)
        finite = np.isfinite(arr)
        values[out_start:out_end][finite] = arr[finite]
        valid[out_start:out_end] |= finite
    return values, valid


def sustained_event(candidate_times: np.ndarray, candidate_windows: np.ndarray, start: int, end: int, channel_idx: int, threshold: float, comparison: str, sustain_minutes: int) -> tuple[bool, bool]:
    if end - start < 60:
        return False, False
    values, valid = interval_values(candidate_times, candidate_windows, start, end, channel_idx)
    return sustained_event_from_second_values(values, valid, threshold, comparison, sustain_minutes)


def sustained_event_from_second_values(values: np.ndarray, valid: np.ndarray, threshold: float, comparison: str, sustain_minutes: int) -> tuple[bool, bool]:
    n_minutes = values.size // 60
    has_valid = False
    run = 0
    for minute in range(n_minutes):
        lo = minute * 60
        hi = lo + 60
        minute_valid = valid[lo:hi]
        if not np.any(minute_valid):
            run = 0
            continue
        has_valid = True
        med = float(np.median(values[lo:hi][minute_valid]))
        if comparison == "le":
            event_minute = med <= threshold
        elif comparison == "lt":
            event_minute = med < threshold
        elif comparison == "gt":
            event_minute = med > threshold
        else:
            raise ValueError(f"Unsupported comparison: {comparison}")
        run = run + 1 if event_minute else 0
        if run >= sustain_minutes:
            return True, True
    return has_valid, False


def build_anchor_horizon_targets_from_aligned_array(anchors: pd.DataFrame, segment_rows: dict[str, np.ndarray], numerics_times: np.ndarray, numerics: np.ndarray, spec: EventTaskSpec) -> tuple[np.ndarray, np.ndarray]:
    n_events = len(spec.event_names)
    targets = np.full((len(anchors), len(spec.horizons_min) * n_events), -1, dtype=np.int8)
    mask = np.zeros_like(targets, dtype=bool)
    for row_idx, row in enumerate(anchors.itertuples(index=False)):
        rows = segment_rows.get(str(row.segment_id))
        if rows is None:
            continue
        times = numerics_times[rows]
        windows = numerics[rows]
        input_start = int(round(float(row.input_start_time)))
        input_end = int(round(float(row.input_end_time)))
        _, active_hypo = sustained_event(times, windows, input_start, input_end, spec.hypotension_channel, spec.hypotension_threshold, "le", spec.sustain_minutes)
        _, active_tachy = sustained_event(times, windows, input_start, input_end, spec.tachycardia_channel, spec.tachycardia_threshold, "gt", spec.sustain_minutes)
        if active_hypo or active_tachy:
            continue
        for horizon_idx, horizon_min in enumerate(spec.horizons_min):
            start = input_end
            end = input_end + horizon_min * 60
            hypo_valid, hypo = sustained_event(times, windows, start, end, spec.hypotension_channel, spec.hypotension_threshold, "le", spec.sustain_minutes)
            tachy_valid, tachy = sustained_event(times, windows, start, end, spec.tachycardia_channel, spec.tachycardia_threshold, "gt", spec.sustain_minutes)
            col = horizon_idx * n_events
            if hypo_valid:
                targets[row_idx, col] = int(hypo)
                mask[row_idx, col] = True
            if tachy_valid:
                targets[row_idx, col + 1] = int(tachy)
                mask[row_idx, col + 1] = True
    return targets, mask


def build_numerics_index(waveform_root: Path, patient_filter: set[str] | None = None) -> dict[str, list[NumericsRecord]]:
    records_path = waveform_root / "RECORDS-numerics"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing RECORDS-numerics at {records_path}")
    out: dict[str, list[NumericsRecord]] = {}
    for line in records_path.read_text().splitlines():
        record_name = line.strip()
        if not record_name:
            continue
        patient_id = record_name.split("/")[1]
        if patient_filter is not None and patient_id not in patient_filter:
            continue
        try:
            header = wfdb.rdheader(str(waveform_root / record_name))
        except Exception:
            continue
        if not header.base_date or not header.base_time:
            continue
        start = (dt.datetime.combine(header.base_date, header.base_time) - REF_DT).total_seconds()
        sig_indices: dict[int, int | None] = {}
        for vital_idx, vital_name in enumerate(VITAL_NAMES):
            col = header.sig_name.index(vital_name) if vital_name in header.sig_name else None
            if col is None:
                for alias in VITAL_ALIASES[vital_name]:
                    if alias in header.sig_name:
                        col = header.sig_name.index(alias)
                        break
            sig_indices[vital_idx] = col
        out.setdefault(patient_id, []).append(
            NumericsRecord(
                path=waveform_root / record_name,
                start_epoch=float(start),
                end_epoch=float(start + header.sig_len / header.fs),
                fs=float(header.fs),
                sig_len=int(header.sig_len),
                sig_indices=sig_indices,
            )
        )
    return out


def find_overlapping_record(records: list[NumericsRecord], start: float, end: float) -> NumericsRecord | None:
    best = None
    best_overlap = 0.0
    for record in records:
        overlap = min(end, record.end_epoch) - max(start, record.start_epoch)
        if overlap > best_overlap:
            best = record
            best_overlap = overlap
    return best if best_overlap > 0 else None


def empty_labels(start: int, end: int) -> dict[str, np.ndarray]:
    minute_times = np.arange(int(np.floor(start / 60.0) * 60), int(np.ceil(end / 60.0) * 60), 60, dtype=np.int64)
    valid = np.zeros(len(minute_times), dtype=bool)
    return {"minute_time": minute_times, "minute_value": np.full(len(minute_times), np.nan, dtype=np.float32), "minute_valid": valid, "event_minutes": valid.copy()}


def minute_labels_from_record(record: NumericsRecord, channel_idx: int, start: int, end: int, threshold: float, comparison: str) -> dict[str, np.ndarray]:
    first_minute = int(np.floor(start / 60.0) * 60)
    last_minute = int(np.ceil(end / 60.0) * 60)
    minute_times = np.arange(first_minute, last_minute, 60, dtype=np.int64)
    values = np.full(len(minute_times), np.nan, dtype=np.float32)
    valid = np.zeros(len(minute_times), dtype=bool)

    col = record.sig_indices.get(channel_idx)
    if col is None or len(minute_times) == 0:
        return empty_labels(start, end)

    try:
        data = wfdb.rdrecord(str(record.path)).p_signal
    except Exception:
        return empty_labels(start, end)

    channel = np.asarray(data[:, col], dtype=np.float32)
    for idx, minute_start in enumerate(minute_times.tolist()):
        sample_start = int(round((minute_start - record.start_epoch) * record.fs))
        sample_end = int(round((minute_start + 60 - record.start_epoch) * record.fs))
        sample_start = max(0, sample_start)
        sample_end = min(record.sig_len, sample_end)
        if sample_end <= sample_start:
            continue
        minute_values = channel[sample_start:sample_end]
        finite = np.isfinite(minute_values)
        if not np.any(finite):
            continue
        valid[idx] = True
        values[idx] = float(np.median(minute_values[finite]))

    if comparison == "le":
        event_minutes = valid & (values <= threshold)
    elif comparison == "lt":
        event_minutes = valid & (values < threshold)
    elif comparison == "gt":
        event_minutes = valid & (values > threshold)
    else:
        raise ValueError(f"Unsupported comparison: {comparison}")
    return {"minute_time": minute_times, "minute_value": values, "minute_valid": valid, "event_minutes": event_minutes}


def range_has_sustained_event(labels: dict[str, np.ndarray], start: int, end: int, sustain_minutes: int) -> tuple[bool, bool]:
    minute_times = labels["minute_time"]
    if len(minute_times) == 0:
        return False, False
    expected = np.arange(int(np.ceil(start / 60.0) * 60), end - 59, 60, dtype=np.int64)
    if len(expected) == 0:
        return False, False
    idx = np.searchsorted(minute_times, expected)
    in_bounds = idx < len(minute_times)
    present = np.zeros(len(expected), dtype=bool)
    present[in_bounds] = minute_times[idx[in_bounds]] == expected[in_bounds]
    safe_idx = idx.clip(max=len(minute_times) - 1)
    valid = present & labels["minute_valid"][safe_idx]
    has_valid = bool(np.any(valid))
    run = 0
    for ok, event in zip(present.tolist(), labels["event_minutes"][safe_idx].tolist()):
        if not ok:
            run = 0
            continue
        run = run + 1 if event else 0
        if run >= sustain_minutes:
            return has_valid, True
    return has_valid, False


def detect_sustained_event_starts(labels: dict[str, np.ndarray], sustain_minutes: int) -> np.ndarray:
    minute_times = labels["minute_time"]
    minute_valid = labels["minute_valid"]
    event_minutes = labels["event_minutes"]
    starts: list[int] = []
    for start_idx in range(0, max(0, len(minute_times) - sustain_minutes + 1)):
        window_times = minute_times[start_idx : start_idx + sustain_minutes]
        expected = minute_times[start_idx] + np.arange(sustain_minutes, dtype=np.int64) * 60
        if np.array_equal(window_times, expected) and np.all(minute_valid[start_idx : start_idx + sustain_minutes]) and np.all(event_minutes[start_idx : start_idx + sustain_minutes]):
            starts.append(int(minute_times[start_idx]))
    return np.asarray(starts, dtype=np.int64)


def window_has_only_clean_minutes(labels: dict[str, np.ndarray], window_start: int, sustain_minutes: int) -> bool:
    minute_times = labels["minute_time"]
    if len(minute_times) == 0:
        return False
    expected = window_start + np.arange(sustain_minutes, dtype=np.int64) * 60
    idx = np.searchsorted(minute_times, expected)
    if np.any(idx >= len(minute_times)) or not np.array_equal(minute_times[idx], expected):
        return False
    return bool(np.all(labels["minute_valid"][idx]) and not np.any(labels["event_minutes"][idx]))


def build_targets_from_waveform_records(anchors: pd.DataFrame, waveform_root: Path, spec: EventTaskSpec, progress_every: int, max_segments: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    grouped = list(anchors.groupby("segment_id", sort=False))
    if max_segments is not None:
        grouped = grouped[:max_segments]
    active_anchors = pd.concat([group for _, group in grouped], axis=0) if grouped else anchors.iloc[:0]
    numerics_index = build_numerics_index(waveform_root, set(active_anchors["patient_id"].astype(str)))
    n_events = len(spec.event_names)
    targets = np.full((len(anchors), len(spec.horizons_min) * n_events), -1, dtype=np.int8)
    mask = np.zeros_like(targets, dtype=bool)
    segment_hypotension_labels: dict[str, dict[str, np.ndarray]] = {}
    patient_has_hypotension: dict[str, bool] = {}
    patient_last_event_times: dict[str, float] = {}
    counts = {"segments": 0, "segments_without_numerics": 0, "segments_without_required_channels": 0}

    if max_segments is not None:
        grouped = grouped[:max_segments]
    for group_idx, (segment_id, segment_anchors) in enumerate(grouped, start=1):
        if progress_every and (group_idx == 1 or group_idx % progress_every == 0):
            print(json.dumps({"event": "progress", "segments_done": group_idx - 1, "segments_total": len(grouped)}), flush=True)
        counts["segments"] += 1
        patient_id = str(segment_anchors["patient_id"].iloc[0])
        records = numerics_index.get(patient_id, [])
        needed_start = float(segment_anchors["input_start_time"].min())
        needed_end = float(segment_anchors["input_end_time"].max() + (max(spec.horizons_min) + spec.sustain_minutes) * 60)
        record = find_overlapping_record(records, needed_start, needed_end)
        if record is None:
            counts["segments_without_numerics"] += 1
            continue
        if record.sig_indices.get(spec.hypotension_channel) is None and record.sig_indices.get(spec.tachycardia_channel) is None:
            counts["segments_without_required_channels"] += 1
            continue

        hypo_labels = minute_labels_from_record(record, spec.hypotension_channel, int(np.floor(needed_start)), int(np.ceil(needed_end)), spec.hypotension_threshold, "le")
        tachy_labels = minute_labels_from_record(record, spec.tachycardia_channel, int(np.floor(needed_start)), int(np.ceil(needed_end)), spec.tachycardia_threshold, "gt")
        segment_hypotension_labels[str(segment_id)] = hypo_labels
        starts = detect_sustained_event_starts(hypo_labels, spec.sustain_minutes)
        if len(starts) > 0:
            patient_has_hypotension[patient_id] = True
            patient_last_event_times[patient_id] = max(float(starts[-1]), patient_last_event_times.get(patient_id, float("-inf")))
        else:
            patient_has_hypotension.setdefault(patient_id, False)

        for row in segment_anchors.itertuples():
            input_start = int(round(float(row.input_start_time)))
            input_end = int(round(float(row.input_end_time)))
            _, active_hypo = range_has_sustained_event(hypo_labels, input_start, input_end, spec.sustain_minutes)
            _, active_tachy = range_has_sustained_event(tachy_labels, input_start, input_end, spec.sustain_minutes)
            if active_hypo or active_tachy:
                continue
            for horizon_idx, horizon_min in enumerate(spec.horizons_min):
                future_start = input_end
                future_end = input_end + horizon_min * 60
                hypo_valid, hypo = range_has_sustained_event(hypo_labels, future_start, future_end, spec.sustain_minutes)
                tachy_valid, tachy = range_has_sustained_event(tachy_labels, future_start, future_end, spec.sustain_minutes)
                col = horizon_idx * n_events
                if hypo_valid:
                    targets[row.Index, col] = int(hypo)
                    mask[row.Index, col] = True
                if tachy_valid:
                    targets[row.Index, col + 1] = int(tachy)
                    mask[row.Index, col + 1] = True

    diagnostics: dict[str, object] = {"mode": "anchor_horizon", "source_counts": counts}
    if spec.target_generation_mode == "anchor_horizon_filtered":
        positive_patient_last_times = [value for value in patient_last_event_times.values() if np.isfinite(value)]
        mean_last_positive_event_time = float(np.mean(np.asarray(positive_patient_last_times, dtype=np.float64))) if positive_patient_last_times else None
        diagnostics = {
            "mode": "anchor_horizon_filtered",
            "negative_filter_scope": "patient",
            "negative_filter_rules": {
                "clean_outcome_window_minutes": int(spec.sustain_minutes),
                "require_valid_outcome_minute": True,
                "exclude_negative_samples_from_positive_recordings": True,
                "late_negative_cutoff": mean_last_positive_event_time,
            },
            "source_counts": counts,
            "per_horizon": {},
        }
        for horizon_idx, horizon_min in enumerate(spec.horizons_min):
            hypo_col = horizon_idx * n_events
            positives_kept = int(((targets[:, hypo_col] == 1) & mask[:, hypo_col]).sum())
            baseline_negatives = int(((targets[:, hypo_col] == 0) & mask[:, hypo_col]).sum())
            dropped_positive_recording = 0
            dropped_late_cutoff = 0
            dropped_outcome_window = 0
            for row in anchors.itertuples():
                row_idx = int(row.Index)
                if not mask[row_idx, hypo_col] or targets[row_idx, hypo_col] != 0:
                    continue
                patient_id = str(row.patient_id)
                segment_id = str(row.segment_id)
                if patient_has_hypotension.get(patient_id, False):
                    targets[row_idx, hypo_col] = -1
                    mask[row_idx, hypo_col] = False
                    dropped_positive_recording += 1
                    continue
                outcome_time = int(round(float(row.input_end_time))) + horizon_min * 60
                if mean_last_positive_event_time is not None and outcome_time > mean_last_positive_event_time:
                    targets[row_idx, hypo_col] = -1
                    mask[row_idx, hypo_col] = False
                    dropped_late_cutoff += 1
                    continue
                labels = segment_hypotension_labels.get(segment_id)
                if labels is None or not window_has_only_clean_minutes(labels, outcome_time, spec.sustain_minutes):
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
                "dropped_negative_positive_recording": dropped_positive_recording,
                "dropped_negative_after_mean_last_positive_event": dropped_late_cutoff,
                "dropped_negative_outcome_window": dropped_outcome_window,
                "retained_prevalence": float(positives_kept / retained_total) if retained_total > 0 else 0.0,
            }
    return targets, mask, diagnostics


def build_targets_from_aligned_array(anchors: pd.DataFrame, numerics_dir: Path, full_data_root: Path, spec: EventTaskSpec, progress_every: int = 250, max_segments: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    segment_ids, numerics_times, numerics = load_aligned_numerics(numerics_dir, full_data_root)
    segment_rows = pd.Series(np.arange(len(segment_ids), dtype=np.int64)).groupby(pd.Series(segment_ids), sort=False).agg(list).to_dict()
    segment_rows = {str(key): np.asarray(value, dtype=np.int64) for key, value in segment_rows.items()}

    n_events = len(spec.event_names)
    targets = np.full((len(anchors), len(spec.horizons_min) * n_events), -1, dtype=np.int8)
    mask = np.zeros_like(targets, dtype=bool)
    segment_hypotension_labels: dict[str, dict[str, np.ndarray]] = {}
    patient_has_sustained_hypotension: dict[str, bool] = {}
    patient_last_event_times: dict[str, float] = {}
    counts = {
        "segments": 0,
        "segments_without_numerics": 0,
        "anchors_without_numerics": 0,
        "max_segments": max_segments,
    }

    grouped = list(anchors.groupby("segment_id", sort=False))
    if max_segments is not None:
        grouped = grouped[:max_segments]
    for group_idx, (segment_id, segment_anchors) in enumerate(grouped, start=1):
        if progress_every and (group_idx == 1 or group_idx % progress_every == 0):
            print(json.dumps({"event": "progress", "segments_done": group_idx - 1, "segments_total": len(grouped)}), flush=True)
        counts["segments"] += 1
        rows = segment_rows.get(str(segment_id))
        if rows is None or len(rows) == 0:
            counts["segments_without_numerics"] += 1
            counts["anchors_without_numerics"] += int(len(segment_anchors))
            continue

        patient_id = str(segment_anchors["patient_id"].iloc[0])
        times = numerics_times[rows]
        windows = numerics[rows]
        hypo_labels = _build_minute_event_labels(
            times,
            windows,
            channel_idx=spec.hypotension_channel,
            threshold=spec.hypotension_threshold,
            comparison="le",
        )
        tachy_labels = _build_minute_event_labels(
            times,
            windows,
            channel_idx=spec.tachycardia_channel,
            threshold=spec.tachycardia_threshold,
            comparison="gt",
        )
        segment_hypotension_labels[str(segment_id)] = hypo_labels
        event_starts = _detect_sustained_event_starts(
            hypo_labels["minute_time"],
            hypo_labels["minute_valid"],
            hypo_labels["event_minutes"],
            sustain_minutes=spec.sustain_minutes,
        )
        has_event = bool(len(event_starts) > 0)
        patient_has_sustained_hypotension[patient_id] = patient_has_sustained_hypotension.get(patient_id, False) or has_event
        if has_event:
            patient_last_event_times[patient_id] = max(float(event_starts[-1]), patient_last_event_times.get(patient_id, float("-inf")))

        for row in segment_anchors.itertuples():
            row_idx = int(row.Index)
            input_start = int(round(float(row.input_start_time)))
            input_end = int(round(float(row.input_end_time)))
            _, active_hypo = range_has_sustained_event(hypo_labels, input_start, input_end, spec.sustain_minutes)
            _, active_tachy = range_has_sustained_event(tachy_labels, input_start, input_end, spec.sustain_minutes)
            if active_hypo or active_tachy:
                continue
            for horizon_idx, horizon_min in enumerate(spec.horizons_min):
                future_start = input_end
                future_end = input_end + horizon_min * 60
                hypo_valid, hypo = range_has_sustained_event(hypo_labels, future_start, future_end, spec.sustain_minutes)
                tachy_valid, tachy = range_has_sustained_event(tachy_labels, future_start, future_end, spec.sustain_minutes)
                col = horizon_idx * n_events
                if hypo_valid:
                    targets[row_idx, col] = int(hypo)
                    mask[row_idx, col] = True
                if tachy_valid:
                    targets[row_idx, col + 1] = int(tachy)
                    mask[row_idx, col + 1] = True

    diagnostics: dict[str, object] = {
        "mode": spec.target_generation_mode,
        "numerics_source": "aligned-array",
        "identity_check": "segment-aware aligned-array target generation using absolute full-data anchor times",
        "source_counts": counts,
    }
    if spec.target_generation_mode != "anchor_horizon_filtered":
        return targets, mask, diagnostics

    mean_last_positive_event_time = (
        float(np.mean(np.asarray(list(patient_last_event_times.values()), dtype=np.float64)))
        if patient_last_event_times
        else None
    )
    diagnostics.update(
        {
            "negative_filter_scope": "patient",
            "negative_filter_rules": {
                "clean_outcome_window_minutes": int(spec.sustain_minutes),
                "require_valid_outcome_minute": True,
                "exclude_negative_samples_from_positive_recordings": True,
                "late_negative_cutoff": mean_last_positive_event_time,
            },
            "per_horizon": {},
        }
    )
    anchor_patient_ids = anchors["patient_id"].astype(str).to_numpy()
    prediction_times = anchors["input_end_time"].to_numpy(dtype=np.float64)
    anchor_segment_ids = anchors["segment_id"].astype(str).to_numpy()
    for horizon_idx, horizon_min in enumerate(spec.horizons_min):
        hypo_col = horizon_idx * n_events
        positives_kept = int(((targets[:, hypo_col] == 1) & mask[:, hypo_col]).sum())
        baseline_negatives = int(((targets[:, hypo_col] == 0) & mask[:, hypo_col]).sum())
        dropped_positive_record = 0
        dropped_late_cutoff = 0
        dropped_outcome_window = 0
        negative_rows = np.flatnonzero((targets[:, hypo_col] == 0) & mask[:, hypo_col])

        for row_idx in negative_rows.tolist():
            patient_id = anchor_patient_ids[row_idx]
            if patient_has_sustained_hypotension.get(patient_id, False):
                targets[row_idx, hypo_col] = -1
                mask[row_idx, hypo_col] = False
                dropped_positive_record += 1
                continue

            outcome_time = int(round(float(prediction_times[row_idx]))) + horizon_min * 60
            if mean_last_positive_event_time is not None and outcome_time > mean_last_positive_event_time:
                targets[row_idx, hypo_col] = -1
                mask[row_idx, hypo_col] = False
                dropped_late_cutoff += 1
                continue

            labels = segment_hypotension_labels.get(anchor_segment_ids[row_idx])
            if labels is None or not _window_has_only_clean_minutes(
                labels["minute_time"],
                labels["minute_valid"],
                labels["event_minutes"],
                window_start=outcome_time,
                sustain_minutes=spec.sustain_minutes,
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
    return targets, mask, diagnostics


def load_aligned_channel_names(numerics_dir: Path) -> list[str]:
    metadata_path = numerics_dir / "numerics_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing aligned numerics metadata with channel names: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    names = metadata.get("vital_names")
    if not names:
        raise ValueError(f"Aligned numerics metadata does not contain vital_names: {metadata_path}")
    return [str(name) for name in names]


def load_numerics_metadata(numerics_dir: Path) -> dict[str, object]:
    metadata_path = numerics_dir / "numerics_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing aligned numerics metadata: {metadata_path}")
    return json.loads(metadata_path.read_text())


def numerics_sampling_rate_hz(metadata: dict[str, object]) -> float:
    for key in ("sampling_rate_hz", "fs", "sample_rate_hz"):
        if key in metadata:
            return float(metadata[key])
    if int(metadata.get("samples_per_window", -1)) == 1200:
        metadata["sampling_rate_hz"] = 1.0
        metadata["sampling_rate_hz_inferred_from_legacy_samples_per_window"] = True
        return 1.0
    raise ValueError("Aligned numerics metadata does not report sampling_rate_hz")


def validate_aligned_numerics_cache(numerics_dir: Path, numerics: np.ndarray, channel_names: list[str]) -> dict[str, object]:
    metadata = load_numerics_metadata(numerics_dir)
    sampling_rate = numerics_sampling_rate_hz(metadata)
    if not np.isclose(sampling_rate, 1.0):
        raise ValueError(
            "The corrected event target builder currently requires "
            f"1 Hz numerics, but metadata reports {sampling_rate} Hz"
        )
    if numerics.ndim != 3:
        raise ValueError(f"Expected X_numerics with shape (N, channels, samples), got {numerics.shape}")
    if numerics.shape[2] <= 0:
        raise ValueError("X_numerics has no time samples")
    if len(channel_names) != numerics.shape[1]:
        raise ValueError(
            f"Channel-name count {len(channel_names)} does not match numerics channel dimension {numerics.shape[1]}"
        )
    timestamp_reference = metadata.get("window_time_reference", metadata.get("timestamp_reference", "center"))
    if str(timestamp_reference) != "center":
        raise ValueError(
            "numerics_window_times.npy must contain numeric-window center timestamps; "
            f"metadata reports {timestamp_reference!r}"
        )
    return metadata


def convert_segment_times(
    raw_times: np.ndarray,
    *,
    seg_start_secs: float,
    source_basis: str,
    target_basis: str,
) -> np.ndarray:
    raw = np.asarray(raw_times, dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise ValueError("Numeric window timestamps contain non-finite values")
    if source_basis == "absolute":
        absolute = raw
        local = raw - float(seg_start_secs)
    elif source_basis == "segment-relative":
        local = raw
        absolute = float(seg_start_secs) + raw
    else:
        raise ValueError(f"Unsupported numeric source basis: {source_basis}")
    if target_basis == "absolute":
        return absolute
    if target_basis == "segment-relative":
        return local
    raise ValueError(f"Unsupported target time basis: {target_basis}")


def segment_start_seconds(segment_anchors: pd.DataFrame, segment_id: str) -> float:
    values = segment_anchors["seg_start_secs"].to_numpy(dtype=np.float64)
    if len(values) == 0:
        raise ValueError(f"Segment {segment_id} has no anchors")
    if not np.allclose(values, values[0], atol=1e-6, rtol=0.0):
        raise ValueError(f"Segment {segment_id} has inconsistent seg_start_secs")
    return float(values[0])


def nearest_distances(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.sort(np.asarray(candidates, dtype=np.float64))
    if cand.size == 0:
        return np.full(ref.shape, np.inf, dtype=np.float64)
    insert = np.searchsorted(cand, ref)
    left_idx = np.clip(insert - 1, 0, cand.size - 1)
    right_idx = np.clip(insert, 0, cand.size - 1)
    return np.minimum(np.abs(ref - cand[left_idx]), np.abs(ref - cand[right_idx]))


def reason_counts_by_target(values: np.ndarray, names: list[str], enum_cls) -> dict[str, dict[str, int]]:
    return {name: _reason_counts(values[:, col], enum_cls) for col, name in enumerate(names)}


def audit_anchor_numeric_alignment(
    *,
    anchors: pd.DataFrame,
    segment_ids: np.ndarray,
    numerics_times: np.ndarray,
    numerics: np.ndarray,
    rows_by_segment: dict[str, np.ndarray],
    aligned_time_basis: str,
    numerics_window_time_basis: str,
    axes: list[int],
    sustain_minutes: int,
    tolerance_seconds: float,
    max_segments: int | None = None,
) -> dict[str, object]:
    anchor_col, input_start_col, input_end_col = _time_cols_for_basis(aligned_time_basis)
    grouped = list(anchors.groupby("segment_id", sort=False))
    if max_segments is not None:
        grouped = grouped[:max_segments]
    center_distances = []
    input_end_distances = []
    forecast_distances = []
    matching_centers = 0
    input_covered = 0
    forecast_covered = 0
    confirmation_covered = 0
    total = 0
    by_segment: dict[str, object] = {}
    n_seconds = int(numerics.shape[2])
    for segment_id, segment_anchors in grouped:
        rows = rows_by_segment.get(str(segment_id))
        if rows is None or len(rows) == 0:
            total += int(len(segment_anchors))
            continue
        seg_start_secs = segment_start_seconds(segment_anchors, str(segment_id))
        times = convert_segment_times(
            numerics_times[rows],
            seg_start_secs=seg_start_secs,
            source_basis=numerics_window_time_basis,
            target_basis=aligned_time_basis,
        )
        starts = np.round(times - n_seconds / 2.0)
        ends = starts + n_seconds
        numeric_min = float(starts.min())
        numeric_max = float(ends.max())
        centers = segment_anchors[anchor_col].to_numpy(dtype=np.float64)
        input_starts = segment_anchors[input_start_col].to_numpy(dtype=np.float64)
        input_ends = segment_anchors[input_end_col].to_numpy(dtype=np.float64)
        max_axis = max(axes) if axes else 0
        forecast_ends = input_ends + float(max_axis) * 60.0
        required_ends = input_ends + float(max_axis + sustain_minutes - 1) * 60.0
        d_center = nearest_distances(centers, times)
        d_input_end = nearest_distances(input_ends, times)
        d_forecast = nearest_distances(forecast_ends, times)
        center_distances.append(d_center)
        input_end_distances.append(d_input_end)
        forecast_distances.append(d_forecast)
        matching_centers += int(np.sum(d_center <= tolerance_seconds))
        input_covered += int(np.sum((input_starts >= numeric_min) & (input_ends <= numeric_max)))
        forecast_covered += int(np.sum((input_ends >= numeric_min) & (forecast_ends <= numeric_max)))
        confirmation_covered += int(np.sum((input_ends >= numeric_min) & (required_ends <= numeric_max)))
        total += int(len(segment_anchors))
        by_segment[str(segment_id)] = {
            "numeric_min": numeric_min,
            "numeric_max": numeric_max,
            "anchor_center_distance": summarize_numeric(d_center),
            "input_end_distance": summarize_numeric(d_input_end),
            "forecast_endpoint_distance": summarize_numeric(d_forecast),
        }
    center_all = np.concatenate(center_distances) if center_distances else np.asarray([], dtype=np.float64)
    input_end_all = np.concatenate(input_end_distances) if input_end_distances else np.asarray([], dtype=np.float64)
    forecast_all = np.concatenate(forecast_distances) if forecast_distances else np.asarray([], dtype=np.float64)
    audit = {
        "total_anchors_audited": int(total),
        "anchor_center_distance_seconds": summarize_numeric(center_all),
        "input_end_distance_seconds": summarize_numeric(input_end_all),
        "forecast_endpoint_distance_seconds": summarize_numeric(forecast_all),
        "fraction_anchors_with_matching_numeric_center": float(matching_centers / total) if total else 0.0,
        "fraction_anchors_with_complete_input_coverage": float(input_covered / total) if total else 0.0,
        "fraction_anchors_with_complete_forecast_coverage": float(forecast_covered / total) if total else 0.0,
        "fraction_anchors_with_complete_confirmation_coverage": float(confirmation_covered / total) if total else 0.0,
        "by_segment_sample": dict(list(by_segment.items())[:50]),
    }
    p95 = audit["anchor_center_distance_seconds"].get("p95", float("inf"))
    if not np.isfinite(p95) or p95 > tolerance_seconds:
        raise ValueError(
            "Feature anchors and numeric windows do not appear to use the same time coordinate; "
            f"anchor-center p95 distance={p95}, tolerance={tolerance_seconds}"
        )
    return audit


def event_target_names(target_mode: str, axes: list[int], event_names: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    for value in axes:
        for event_name in event_names:
            if target_mode == "anchor_onset_within_horizon":
                names.append(f"{event_name}_onset_within_{value}m")
            elif target_mode == "anchor_fixed_forecast_window":
                names.append(f"{event_name}_onset_at_{value}m_gap")
            else:
                names.append(f"{event_name}_within_{value}m")
    return names


def _time_cols_for_basis(time_basis: str) -> tuple[str, str, str]:
    if time_basis == "absolute":
        return "anchor_time_absolute", "input_start_time_absolute", "input_end_time_absolute"
    if time_basis == "segment-relative":
        return "anchor_time_local", "input_start_time_local", "input_end_time_local"
    raise ValueError(f"Unsupported time basis: {time_basis}")


def _event_params(event_name: str, spec: EventTaskSpec, channel_lookup: dict[str, int]) -> tuple[int, float, str]:
    if event_name == "hypotension":
        return channel_lookup["hypotension"], float(spec.hypotension_threshold), "le"
    if event_name == "tachycardia":
        return channel_lookup["tachycardia"], float(spec.tachycardia_threshold), "gt"
    if event_name == "hypoxia":
        return channel_lookup["hypoxia"], float(spec.hypoxia_threshold), "lt"
    raise ValueError(f"Unsupported event name: {event_name}")


def _group_id(row: pd.Series, scope: str | None) -> str:
    if scope is None or scope == "none":
        return ""
    if scope == "segment":
        return str(row["segment_id"])
    if scope == "recording":
        value = row.get("waveform_record_id")
        if value is None or str(value) == "" or str(value).lower() == "nan":
            raise ValueError("recording scope requires waveform_record_id")
        return str(value)
    if scope == "icu-stay":
        return str(row.get("icustay_id", row.get("ICUSTAY_ID", row["segment_id"])))
    if scope == "patient":
        return str(row["patient_id"])
    raise ValueError(f"Unsupported negative exclusion scope: {scope}")


def _safe_file_sha256(path: Path, max_bytes: int = 4 * 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        h.update(handle.read(max_bytes))
    return h.hexdigest()


def _reason_counts(values: np.ndarray, enum_cls) -> dict[str, int]:
    flat = np.asarray(values).reshape(-1)
    out: dict[str, int] = {}
    for reason in enum_cls:
        count = int(np.sum(flat == int(reason)))
        if count:
            out[reason.name] = count
    return out


def resolve_late_cutoff_group_scope(
    *,
    apply_late_negative_cutoff: bool,
    negative_exclusion_scope: str,
    late_cutoff_group_scope: str | None,
) -> str | None:
    if not apply_late_negative_cutoff:
        return None
    if late_cutoff_group_scope is not None:
        return late_cutoff_group_scope
    if negative_exclusion_scope != "none":
        return negative_exclusion_scope
    raise ValueError("--apply-late-negative-cutoff requires --late-cutoff-group-scope when --negative-exclusion-scope none")


def candidate_time_for_late_cutoff(row: pd.Series, *, input_end_col: str, axis_minutes: int, late_cutoff_candidate: str) -> float:
    candidate_time = float(row[input_end_col])
    if late_cutoff_candidate == "forecast_endpoint":
        return candidate_time + int(axis_minutes) * 60.0
    if late_cutoff_candidate == "input_end":
        return candidate_time
    raise ValueError(f"Unsupported late cutoff candidate: {late_cutoff_candidate}")


def invalidate_filtered_negative(
    *,
    row_idx: int,
    col: int,
    reason: NegativeFilterReason,
    filtered_targets: np.ndarray,
    filtered_mask: np.ndarray,
    filter_reasons: np.ndarray,
) -> None:
    filtered_targets[row_idx, col] = int(LabelState.INVALID)
    filtered_mask[row_idx, col] = False
    filter_reasons[row_idx, col] = int(reason)


def compute_late_negative_cutoffs(
    *,
    positive_group_onsets: dict[tuple[str, str], list[float]],
    group_starts: dict[str, float],
    event_names: tuple[str, ...],
    strategy: str,
) -> tuple[dict[str, float], dict[tuple[str, str], float], dict[str, object]]:
    last_elapsed_by_event_group: dict[tuple[str, str], float] = {}
    elapsed_by_event: dict[str, list[float]] = {event: [] for event in event_names}
    for (event_name, group_id), starts_abs in positive_group_onsets.items():
        group_start = group_starts.get(group_id)
        if group_start is None or not starts_abs:
            continue
        elapsed = float(max(starts_abs) - group_start)
        last_elapsed_by_event_group[(event_name, group_id)] = elapsed
        elapsed_by_event.setdefault(event_name, []).append(elapsed)

    if strategy == "mean-last-positive":
        global_cutoffs = {
            event_name: float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")
            for event_name, values in elapsed_by_event.items()
        }
        group_cutoffs: dict[tuple[str, str], float] = {}
    elif strategy == "group-last-positive":
        global_cutoffs = {}
        group_cutoffs = dict(last_elapsed_by_event_group)
    else:
        raise ValueError(f"Unsupported late cutoff strategy: {strategy}")

    summary: dict[str, object] = {
        "strategy": strategy,
        "n_event_groups_with_positive_onsets": int(len(last_elapsed_by_event_group)),
        "by_event": {},
    }
    by_event_summary: dict[str, object] = {}
    for event_name in event_names:
        values = np.asarray(elapsed_by_event.get(event_name, []), dtype=np.float64)
        by_event_summary[event_name] = {
            "n_groups_with_positive_onsets": int(len(values)),
            "mean_last_positive_elapsed_seconds": float(np.mean(values)) if len(values) else float("nan"),
            "min_last_positive_elapsed_seconds": float(np.min(values)) if len(values) else float("nan"),
            "max_last_positive_elapsed_seconds": float(np.max(values)) if len(values) else float("nan"),
        }
    summary["by_event"] = by_event_summary
    return global_cutoffs, group_cutoffs, summary


def build_canonical_late_cutoff_group_onsets(
    *,
    anchors: pd.DataFrame,
    active_indices: np.ndarray,
    rows_by_segment: dict[str, np.ndarray],
    numerics_segment_ids: np.ndarray,
    segment_start_secs_by_id: dict[str, float],
    numerics_times: np.ndarray,
    numerics: np.ndarray,
    spec: EventTaskSpec,
    channel_lookup: dict[str, int],
    aligned_time_basis: str,
    numerics_window_time_basis: str,
    min_valid_fraction_per_minute: float,
    late_cutoff_group_scope: str,
    input_start_col: str,
) -> tuple[dict[tuple[str, str], list[float]], dict[str, float], dict[str, object]]:
    if late_cutoff_group_scope == "recording" and aligned_time_basis != "absolute":
        raise ValueError("recording-level late cutoff requires --aligned-time-basis absolute")

    active = anchors.loc[active_indices].copy()
    onsets: dict[tuple[str, str], list[float]] = {}
    group_starts: dict[str, float] = {}
    diagnostics: dict[str, object] = {
        "late_cutoff_onset_timeline": "canonical_complete_recording_level",
        "late_cutoff_group_scope": late_cutoff_group_scope,
        "n_late_cutoff_groups_audited": 0,
        "anchor_bearing_segments": 0,
        "numerics_bearing_segments_scanned": 0,
        "recordings_with_additional_no_anchor_segments": 0,
        "confirmed_onset_recordings_last_onset_from_no_anchor_segment": 0,
    }
    if active.empty:
        return onsets, group_starts, diagnostics

    selected_groups = {str(_group_id(row, late_cutoff_group_scope)) for _, row in active.iterrows()}
    selected_groups.discard("")
    anchor_segments_by_group: dict[str, set[str]] = {gid: set() for gid in selected_groups}
    for _, row in active.iterrows():
        gid = str(_group_id(row, late_cutoff_group_scope))
        if gid:
            anchor_segments_by_group.setdefault(gid, set()).add(str(row["segment_id"]))
            group_starts[gid] = min(group_starts.get(gid, float("inf")), float(row[input_start_col]))

    numerics_segments_by_group: dict[str, list[str]] = {gid: [] for gid in selected_groups}
    for segment_id in sorted(rows_by_segment):
        gid = waveform_record_id_from_segment_id(segment_id)
        if gid in selected_groups:
            numerics_segments_by_group.setdefault(gid, []).append(str(segment_id))

    diagnostics["anchor_bearing_segments"] = int(sum(len(v) for v in anchor_segments_by_group.values()))

    for group_id in sorted(selected_groups):
        anchor_segment_ids = anchor_segments_by_group.get(group_id, set())
        numerics_segment_ids_for_group = numerics_segments_by_group.get(group_id, [])
        if not numerics_segment_ids_for_group:
            continue
        no_anchor_segments = set(numerics_segment_ids_for_group).difference(anchor_segment_ids)
        if no_anchor_segments:
            diagnostics["recordings_with_additional_no_anchor_segments"] = int(diagnostics["recordings_with_additional_no_anchor_segments"]) + 1
        diagnostics["numerics_bearing_segments_scanned"] = int(diagnostics["numerics_bearing_segments_scanned"]) + len(numerics_segment_ids_for_group)

        all_times: list[np.ndarray] = []
        all_windows: list[np.ndarray] = []
        segment_intervals: dict[str, tuple[float, float]] = {}
        for segment_id in numerics_segment_ids_for_group:
            rows = rows_by_segment.get(segment_id)
            if rows is None or len(rows) == 0:
                continue
            seg_start_secs = segment_start_secs_by_id.get(segment_id)
            if seg_start_secs is None:
                raise ValueError(f"Missing seg_start_secs for numerics segment {segment_id}")
            converted = convert_segment_times(
                numerics_times[rows],
                seg_start_secs=float(seg_start_secs),
                source_basis=numerics_window_time_basis,
                target_basis=aligned_time_basis,
            )
            segment_windows = numerics[rows]
            all_times.append(converted)
            all_windows.append(segment_windows)
            half = segment_windows.shape[2] / 2.0
            segment_intervals[segment_id] = (float((converted - half).min()), float((converted + half).max()))
        if not all_times:
            continue
        times = np.concatenate(all_times)
        windows = np.concatenate(all_windows, axis=0)
        order = np.argsort(times, kind="mergesort")
        times = times[order]
        windows = windows[order]
        timeline_origin = float((times - windows.shape[2] / 2.0).min())
        group_has_onset = False
        group_last_onset = float("nan")
        for event_name in spec.event_names:
            channel_idx, threshold, comparison = _event_params(event_name, spec, channel_lookup)
            timeline = build_minute_timeline_from_windows(
                window_times=times,
                windows=windows,
                channel_idx=channel_idx,
                threshold=threshold,
                comparison=comparison,
                min_valid_fraction_per_minute=min_valid_fraction_per_minute,
                timeline_origin_seconds=timeline_origin,
            )
            episodes = detect_maximal_episodes(timeline.minute_valid, timeline.event_minutes, spec.sustain_minutes)
            starts = [timeline.origin_seconds + ep.start_minute_index * 60.0 for ep in episodes]
            if starts:
                onsets.setdefault((event_name, group_id), []).extend(starts)
                group_has_onset = True
                group_last_onset = max(group_last_onset, max(starts)) if np.isfinite(group_last_onset) else max(starts)
        if group_has_onset and no_anchor_segments:
            for segment_id in no_anchor_segments:
                start, end = segment_intervals.get(segment_id, (float("nan"), float("nan")))
                if np.isfinite(start) and start <= group_last_onset < end:
                    diagnostics["confirmed_onset_recordings_last_onset_from_no_anchor_segment"] = int(diagnostics["confirmed_onset_recordings_last_onset_from_no_anchor_segment"]) + 1
                    break
        diagnostics["n_late_cutoff_groups_audited"] = int(diagnostics["n_late_cutoff_groups_audited"]) + 1

    return onsets, group_starts, diagnostics


def assert_final_negative_filter_invariants(
    *,
    anchors: pd.DataFrame,
    active_indices: np.ndarray,
    base_targets: np.ndarray,
    base_mask: np.ndarray,
    filtered_targets: np.ndarray,
    filtered_mask: np.ndarray,
    axes: list[int],
    event_names: tuple[str, ...],
    apply_late_negative_cutoff: bool,
    late_cutoff_candidate: str,
    late_cutoff_group_scope: str | None,
    late_cutoff_strategy: str,
    exclude_late_cutoff_groups_without_positives: bool,
    late_cutoff_by_event_group: dict[tuple[str, str], float],
    late_cutoff_group_starts: dict[str, float],
    input_end_col: str,
) -> None:
    n_events = len(event_names)
    for row_idx in active_indices.tolist():
        row = anchors.loc[int(row_idx)]
        late_group_id = _group_id(row, late_cutoff_group_scope) if late_cutoff_group_scope is not None else ""
        group_start = late_cutoff_group_starts.get(late_group_id)
        for axis_idx, axis_min in enumerate(axes):
            for event_offset, event_name in enumerate(event_names):
                col = axis_idx * n_events + event_offset
                if not filtered_mask[row_idx, col] or filtered_targets[row_idx, col] != int(LabelState.NEGATIVE):
                    continue
                if not base_mask[row_idx, col] or base_targets[row_idx, col] != int(LabelState.NEGATIVE):
                    raise AssertionError("Final negative was not a valid base negative")
                if not apply_late_negative_cutoff:
                    continue
                if late_cutoff_strategy != "group-last-positive":
                    continue
                cutoff = late_cutoff_by_event_group.get((event_name, late_group_id), float("nan"))
                if exclude_late_cutoff_groups_without_positives and not np.isfinite(cutoff):
                    raise AssertionError("Final negative belongs to a cutoff group without a confirmed onset")
                if np.isfinite(cutoff):
                    if group_start is None:
                        raise AssertionError("Final negative cutoff group has no recorded group start")
                    candidate_elapsed = candidate_time_for_late_cutoff(
                        row,
                        input_end_col=input_end_col,
                        axis_minutes=int(axis_min),
                        late_cutoff_candidate=late_cutoff_candidate,
                    ) - group_start
                    if candidate_elapsed > cutoff:
                        raise AssertionError("Final negative occurs after the last confirmed onset in its cutoff group")


def apply_negative_filters(
    *,
    anchors: pd.DataFrame,
    active_indices: np.ndarray,
    base_targets: np.ndarray,
    base_mask: np.ndarray,
    filtered_targets: np.ndarray,
    filtered_mask: np.ndarray,
    filter_reasons: np.ndarray,
    axes: list[int],
    event_names: tuple[str, ...],
    negative_exclusion_scope: str,
    event_group_has_event: dict[tuple[str, str], bool],
    apply_late_negative_cutoff: bool,
    late_cutoff_candidate: str,
    late_cutoff_group_scope: str | None,
    late_cutoff_strategy: str,
    exclude_late_cutoff_groups_without_positives: bool,
    late_cutoff_global_by_event: dict[str, float],
    late_cutoff_by_event_group: dict[tuple[str, str], float],
    late_cutoff_group_starts: dict[str, float],
    input_end_col: str,
) -> None:
    if negative_exclusion_scope == "none" and not apply_late_negative_cutoff:
        return

    n_events = len(event_names)
    for row_idx in active_indices.tolist():
        row = anchors.loc[int(row_idx)]
        event_group_id = _group_id(row, negative_exclusion_scope)
        late_group_id = _group_id(row, late_cutoff_group_scope) if late_cutoff_group_scope is not None else ""
        for axis_idx, axis_min in enumerate(axes):
            for event_offset, event_name in enumerate(event_names):
                col = axis_idx * n_events + event_offset
                if not base_mask[row_idx, col] or base_targets[row_idx, col] == int(LabelState.POSITIVE):
                    continue

                if negative_exclusion_scope != "none" and event_group_has_event.get((event_name, event_group_id), False):
                    invalidate_filtered_negative(
                        row_idx=row_idx,
                        col=col,
                        reason=NegativeFilterReason.EXCLUDED_EVENT_GROUP,
                        filtered_targets=filtered_targets,
                        filtered_mask=filtered_mask,
                        filter_reasons=filter_reasons,
                    )
                    continue

                if not apply_late_negative_cutoff:
                    continue

                if late_cutoff_strategy == "mean-last-positive":
                    cutoff = late_cutoff_global_by_event.get(event_name, float("nan"))
                elif late_cutoff_strategy == "group-last-positive":
                    cutoff = late_cutoff_by_event_group.get((event_name, late_group_id), float("nan"))
                    if not np.isfinite(cutoff) and exclude_late_cutoff_groups_without_positives:
                        invalidate_filtered_negative(
                            row_idx=row_idx,
                            col=col,
                            reason=NegativeFilterReason.NO_POSITIVE_CUTOFF_GROUP,
                            filtered_targets=filtered_targets,
                            filtered_mask=filtered_mask,
                            filter_reasons=filter_reasons,
                        )
                        continue
                else:
                    raise ValueError(f"Unsupported late cutoff strategy: {late_cutoff_strategy}")

                if not np.isfinite(cutoff):
                    continue
                group_start = late_cutoff_group_starts.get(late_group_id)
                if group_start is None:
                    continue
                candidate_elapsed = candidate_time_for_late_cutoff(
                    row,
                    input_end_col=input_end_col,
                    axis_minutes=int(axis_min),
                    late_cutoff_candidate=late_cutoff_candidate,
                ) - group_start
                if candidate_elapsed > cutoff:
                    invalidate_filtered_negative(
                        row_idx=row_idx,
                        col=col,
                        reason=NegativeFilterReason.AFTER_LATE_CUTOFF,
                        filtered_targets=filtered_targets,
                        filtered_mask=filtered_mask,
                        filter_reasons=filter_reasons,
                    )


def _label_counts(targets: np.ndarray, mask: np.ndarray, names: list[str], split_labels: np.ndarray | None = None) -> dict[str, object]:
    summary: dict[str, object] = {}
    for col, name in enumerate(names):
        valid = mask[:, col].astype(bool)
        pos = valid & (targets[:, col] == int(LabelState.POSITIVE))
        neg = valid & (targets[:, col] == int(LabelState.NEGATIVE))
        item = {
            "total_anchor_rows": int(targets.shape[0]),
            "valid": int(valid.sum()),
            "positive": int(pos.sum()),
            "negative": int(neg.sum()),
            "invalid": int((~valid).sum()),
            "prevalence": float(pos.sum() / valid.sum()) if valid.any() else 0.0,
            "percent_valid": float(valid.mean()) if len(valid) else 0.0,
        }
        if split_labels is not None:
            by_split = {}
            for split in sorted(set(np.asarray(split_labels, dtype=str).tolist())):
                rows = np.asarray(split_labels, dtype=str) == split
                v = valid & rows
                p = pos & rows
                n = neg & rows
                by_split[split] = {
                    "valid": int(v.sum()),
                    "positive": int(p.sum()),
                    "negative": int(n.sum()),
                    "prevalence": float(p.sum() / v.sum()) if v.any() else 0.0,
                }
            item["by_split"] = by_split
        summary[name] = item
    summary["unique_anchors_with_at_least_one_valid_target"] = int(np.any(mask, axis=1).sum())
    return summary


def _compact_pattern(arr: np.ndarray, start: int, end: int, max_len: int = 80) -> str:
    lo = max(0, int(start))
    hi = min(len(arr), int(end))
    chars = "".join("1" if bool(x) else "0" for x in np.asarray(arr, dtype=bool)[lo:hi].tolist())
    return chars[:max_len]


def write_audit_csv(
    path: Path,
    anchors: pd.DataFrame,
    names: list[str],
    targets: np.ndarray,
    mask: np.ndarray,
    invalid_reasons: np.ndarray,
    filtered_targets: np.ndarray,
    filtered_mask: np.ndarray,
    filter_reasons: np.ndarray,
    audit_context: dict[tuple[str, str], dict[str, object]],
    *,
    seed: int = 20260831,
) -> None:
    rng = np.random.default_rng(seed)
    rows_out: list[dict[str, object]] = []
    for col, name in enumerate(names):
        categories: list[tuple[str, np.ndarray]] = [
            ("positive", np.flatnonzero(mask[:, col] & (targets[:, col] == 1))),
            ("negative", np.flatnonzero(mask[:, col] & (targets[:, col] == 0))),
        ]
        for reason in InvalidReason:
            if reason == InvalidReason.KEPT:
                continue
            categories.append((f"invalid_{reason.name}", np.flatnonzero(invalid_reasons[:, col] == int(reason))))
        for category, idxs in categories:
            if idxs.size == 0:
                continue
            take = idxs if idxs.size <= 50 else rng.choice(idxs, size=50, replace=False)
            for row_idx in sorted(int(x) for x in take.tolist()):
                anchor = anchors.iloc[row_idx]
                event_name = name.split("_onset_")[0] if "_onset_" in name else name.split("_within_")[0]
                ctx = audit_context.get((str(anchor["segment_id"]), event_name, row_idx), audit_context.get((str(anchor["segment_id"]), event_name), {}))
                input_start_min = ""
                input_end_min = ""
                forecast_start_time = ""
                forecast_end_time = ""
                required_end_time = ""
                nearest_episode_start = ""
                nearest_episode_end = ""
                episode_overlaps_input = ""
                audit_context_error = ""
                if ctx:
                    try:
                        input_start_col = str(ctx.get("input_start_col", "input_start_time"))
                        input_end_col = str(ctx.get("input_end_col", "input_end_time"))
                        origin = float(ctx["timeline_origin"])
                        input_start_min = timestamp_to_minute_index(float(anchor[input_start_col]), origin)[0]
                        input_end_min = timestamp_to_minute_index(float(anchor[input_end_col]), origin)[0]
                        axis = int(ctx.get("axes", [])[col // int(ctx.get("n_events", 1))])
                        if ctx.get("target_mode") == "anchor_fixed_forecast_window":
                            forecast_start_time = float(anchor[input_end_col]) + axis * 60.0
                            forecast_end_time = forecast_start_time + int(ctx.get("sustain_minutes", 5)) * 60.0
                            required_end_time = forecast_end_time
                        else:
                            forecast_start_time = float(anchor[input_end_col])
                            forecast_end_time = forecast_start_time + axis * 60.0
                            required_end_time = forecast_start_time + (axis + int(ctx.get("sustain_minutes", 5)) - 1) * 60.0
                        episodes = ctx.get("episodes", [])
                        episode_overlaps_input = any_episode_overlaps(episodes, int(input_start_min), int(input_end_min))
                        if episodes:
                            starts = np.asarray([ep.start_minute_index for ep in episodes], dtype=np.int64)
                            nearest_idx = int(np.argmin(np.abs(starts - int(input_end_min))))
                            ep = episodes[nearest_idx]
                            nearest_episode_start = origin + ep.start_minute_index * 60.0
                            nearest_episode_end = origin + ep.end_minute_index_exclusive * 60.0
                    except (KeyError, ValueError, IndexError) as exc:
                        audit_context_error = f"{type(exc).__name__}: {exc}"
                        input_start_col = str(ctx.get("input_start_col", "input_start_time"))
                        input_end_col = str(ctx.get("input_end_col", "input_end_time"))
                else:
                    input_start_col = "input_start_time"
                    input_end_col = "input_end_time"
                minute_valid = ctx.get("minute_valid")
                event_minutes = ctx.get("event_minutes")
                rows_out.append(
                    {
                        "target_name": name,
                        "sample_category": category,
                        "anchor_id": anchor.get("anchor_id", row_idx),
                        "patient_id": anchor.get("patient_id", ""),
                        "segment_id": anchor.get("segment_id", ""),
                        "recording_id": anchor.get("waveform_record_id", ""),
                        "icu_stay_id": anchor.get("icustay_id", anchor.get("ICUSTAY_ID", "")),
                        "negative_group_id": anchor.get("negative_group_id", ""),
                        "split_label": anchor.get("split_label", ""),
                        "canonical_anchor_time": anchor.get("canonical_anchor_time", ""),
                        "raw_anchors_csv_window_time": anchor.get("window_time", ""),
                        "seg_start_secs": anchor.get("seg_start_secs", ""),
                        "anchor_time_absolute": anchor.get("anchor_time_absolute", ""),
                        "anchor_time_local": anchor.get("anchor_time_local", ""),
                        "raw_numeric_time_basis": ctx.get("numerics_window_time_basis", ""),
                        "working_time_basis": ctx.get("time_basis", ""),
                        "time_basis": ctx.get("time_basis", ""),
                        "timeline_origin": ctx.get("timeline_origin", ""),
                        "input_start_time": anchor.get(ctx.get("input_start_col", "input_start_time"), ""),
                        "input_end_time": anchor.get(ctx.get("input_end_col", "input_end_time"), ""),
                        "input_start_time_absolute": anchor.get("input_start_time_absolute", ""),
                        "input_end_time_absolute": anchor.get("input_end_time_absolute", ""),
                        "input_start_time_local": anchor.get("input_start_time_local", ""),
                        "input_end_time_local": anchor.get("input_end_time_local", ""),
                        "forecast_start_time": forecast_start_time,
                        "forecast_end_time": forecast_end_time,
                        "required_coverage_end_time": required_end_time,
                        "nearest_episode_start": nearest_episode_start,
                        "nearest_episode_end": nearest_episode_end,
                        "episode_overlaps_input": episode_overlaps_input,
                        "minute_validity_pattern": "" if minute_valid is None or input_end_min == "" else _compact_pattern(minute_valid, int(input_end_min), int(input_end_min) + 20),
                        "minute_event_pattern": "" if event_minutes is None or input_end_min == "" else _compact_pattern(event_minutes, int(input_end_min), int(input_end_min) + 20),
                        "base_target": int(targets[row_idx, col]),
                        "base_mask": bool(mask[row_idx, col]),
                        "base_invalid_reason": InvalidReason(int(invalid_reasons[row_idx, col])).name,
                        "filtered_target": int(filtered_targets[row_idx, col]),
                        "filtered_mask": bool(filtered_mask[row_idx, col]),
                        "negative_filter_reason": NegativeFilterReason(int(filter_reasons[row_idx, col])).name,
                        "audit_context_error": audit_context_error,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(path, index=False)


def select_audit_row_indices(
    names: list[str],
    targets: np.ndarray,
    mask: np.ndarray,
    invalid_reasons: np.ndarray,
    *,
    candidate_rows: np.ndarray | None = None,
    seed: int = 20260831,
) -> set[int]:
    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    candidate_mask = None
    if candidate_rows is not None:
        candidate_mask = np.zeros(targets.shape[0], dtype=bool)
        candidate_mask[np.asarray(candidate_rows, dtype=np.int64)] = True
    for col, name in enumerate(names):
        valid_col = mask[:, col]
        if candidate_mask is not None:
            valid_col = valid_col & candidate_mask
        categories: list[np.ndarray] = [
            np.flatnonzero(valid_col & (targets[:, col] == 1)),
            np.flatnonzero(valid_col & (targets[:, col] == 0)),
        ]
        for reason in InvalidReason:
            if reason != InvalidReason.KEPT:
                reason_mask = invalid_reasons[:, col] == int(reason)
                if candidate_mask is not None:
                    reason_mask = reason_mask & candidate_mask
                categories.append(np.flatnonzero(reason_mask))
        for idxs in categories:
            if idxs.size == 0:
                continue
            take = idxs if idxs.size <= 50 else rng.choice(idxs, size=50, replace=False)
            for row_idx in take.tolist():
                selected.add(int(row_idx))
    return selected


def build_audit_context_for_rows(
    *,
    selected_rows: set[int],
    anchors: pd.DataFrame,
    segment_ids: np.ndarray,
    numerics_times: np.ndarray,
    numerics: np.ndarray,
    rows_by_segment: dict[str, np.ndarray],
    spec: EventTaskSpec,
    channel_lookup: dict[str, int],
    target_mode: str,
    aligned_time_basis: str,
    numerics_window_time_basis: str,
    axes: list[int],
    min_valid_fraction_per_minute: float,
) -> dict[tuple[object, ...], dict[str, object]]:
    if not selected_rows:
        return {}
    _, input_start_col, input_end_col = _time_cols_for_basis(aligned_time_basis)
    context: dict[tuple[object, ...], dict[str, object]] = {}
    for row_idx in sorted(selected_rows):
        anchor = anchors.iloc[int(row_idx)]
        segment_id = str(anchor["segment_id"])
        rows = rows_by_segment.get(segment_id)
        if rows is None or len(rows) == 0:
            continue
        segment_anchors = anchors[anchors["segment_id"].astype(str) == segment_id]
        seg_start_secs = segment_start_seconds(segment_anchors, segment_id)
        times = convert_segment_times(
            numerics_times[rows],
            seg_start_secs=seg_start_secs,
            source_basis=numerics_window_time_basis,
            target_basis=aligned_time_basis,
        )
        windows = numerics[rows]
        timeline_origin = float(anchor[input_start_col])
        for event_name in spec.event_names:
            channel_idx, threshold, comparison = _event_params(event_name, spec, channel_lookup)
            timeline = build_minute_timeline_from_windows(
                window_times=times,
                windows=windows,
                channel_idx=channel_idx,
                threshold=threshold,
                comparison=comparison,
                min_valid_fraction_per_minute=min_valid_fraction_per_minute,
                timeline_origin_seconds=timeline_origin,
            )
            episodes = detect_maximal_episodes(timeline.minute_valid, timeline.event_minutes, spec.sustain_minutes)
            context[(segment_id, event_name, int(row_idx))] = {
                "time_basis": aligned_time_basis,
                "numerics_window_time_basis": numerics_window_time_basis,
                "timeline_origin": timeline.origin_seconds,
                "input_start_col": input_start_col,
                "input_end_col": input_end_col,
                "target_mode": target_mode,
                "axes": axes,
                "n_events": len(spec.event_names),
                "sustain_minutes": spec.sustain_minutes,
                "minute_valid": timeline.minute_valid,
                "event_minutes": timeline.event_minutes,
                "episodes": episodes,
            }
    return context


def build_corrected_targets_from_aligned_array(
    anchors: pd.DataFrame,
    numerics_dir: Path,
    full_data_root: Path,
    spec: EventTaskSpec,
    *,
    target_mode: str,
    aligned_time_basis: str,
    negative_policy: str,
    negative_exclusion_scope: str,
    apply_late_negative_cutoff: bool,
    late_cutoff_candidate: str,
    late_cutoff_group_scope: str | None,
    late_cutoff_strategy: str,
    exclude_late_cutoff_groups_without_positives: bool,
    min_valid_fraction_per_minute: float,
    numerics_window_time_basis: str = "absolute",
    timestamp_alignment_tolerance_seconds: float = 1.0,
    progress_every: int = 250,
    max_segments: int | None = None,
    audit_csv: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, object], list[str]]:
    if target_mode not in {"anchor_onset_within_horizon", "anchor_fixed_forecast_window"}:
        raise ValueError(f"Unsupported corrected target mode: {target_mode}")
    if target_mode == "anchor_fixed_forecast_window" and negative_policy != "clean-fixed-window":
        raise ValueError("anchor_fixed_forecast_window requires --negative-policy clean-fixed-window")
    resolved_late_cutoff_group_scope = resolve_late_cutoff_group_scope(
        apply_late_negative_cutoff=apply_late_negative_cutoff,
        negative_exclusion_scope=negative_exclusion_scope,
        late_cutoff_group_scope=late_cutoff_group_scope,
    )

    segment_ids, numerics_times, numerics = load_aligned_numerics(numerics_dir, full_data_root)
    channel_names = load_aligned_channel_names(numerics_dir)
    numerics_metadata = validate_aligned_numerics_cache(numerics_dir, numerics, channel_names)
    hypo_channel = resolve_channel(
        channel_names,
        preferred_name="ABP Mean",
        aliases=["ABPMean", "ART Mean", "ARTMean", "MAP"],
    )
    tachy_channel = resolve_channel(channel_names, preferred_name="HR", aliases=[])
    hypoxia_channel = resolve_channel(channel_names, preferred_name="SpO2", aliases=["%SpO2"])
    channel_info = {"hypotension": hypo_channel, "tachycardia": tachy_channel, "hypoxia": hypoxia_channel}
    channel_lookup = {event_name: channel_info[event_name].index for event_name in spec.event_names}
    channel_resolution = {event_name: asdict(channel_info[event_name]) for event_name in spec.event_names}
    channel_quality = {
        event_name: {"valid_minutes": 0, "total_minutes": 0, "finite_sample_count": 0, "total_sample_count": 0}
        for event_name in spec.event_names
    }

    axes = list(spec.horizons_min)
    names = event_target_names(target_mode, axes, spec.event_names)
    n_events = len(spec.event_names)
    target_dim = len(axes) * n_events
    base_targets = np.full((len(anchors), target_dim), int(LabelState.INVALID), dtype=np.int8)
    base_mask = np.zeros_like(base_targets, dtype=bool)
    invalid_reasons = np.full_like(base_targets, int(InvalidReason.NO_NUMERICS), dtype=np.int16)
    filtered_targets = base_targets.copy()
    filtered_mask = base_mask.copy()
    filter_reasons = np.full_like(base_targets, int(NegativeFilterReason.BASE_INVALID), dtype=np.int16)

    anchor_col, input_start_col, input_end_col = _time_cols_for_basis(aligned_time_basis)
    grouped = list(anchors.groupby("segment_id", sort=False))
    if max_segments is not None:
        grouped = grouped[:max_segments]
    active_indices = np.concatenate([group.index.to_numpy(dtype=np.int64) for _, group in grouped]) if grouped else np.zeros(0, dtype=np.int64)

    rows_by_segment = pd.Series(np.arange(len(segment_ids), dtype=np.int64)).groupby(pd.Series(segment_ids), sort=False).agg(list).to_dict()
    rows_by_segment = {str(key): np.asarray(value, dtype=np.int64) for key, value in rows_by_segment.items()}
    segment_start_secs_by_id = load_segment_start_seconds_by_id(full_data_root)
    alignment_audit = audit_anchor_numeric_alignment(
        anchors=anchors,
        segment_ids=segment_ids,
        numerics_times=numerics_times,
        numerics=numerics,
        rows_by_segment=rows_by_segment,
        aligned_time_basis=aligned_time_basis,
        numerics_window_time_basis=numerics_window_time_basis,
        axes=axes,
        sustain_minutes=spec.sustain_minutes,
        tolerance_seconds=timestamp_alignment_tolerance_seconds,
        max_segments=max_segments,
    )
    event_group_has_event: dict[tuple[str, str], bool] = {}
    late_cutoff_group_onsets: dict[tuple[str, str], list[float]] = {}
    late_cutoff_group_starts: dict[str, float] = {}
    late_cutoff_group_diagnostics: dict[str, object] = {"late_cutoff_onset_timeline": "segment_phase_specific"}
    segment_diagnostics: dict[str, object] = {}
    time_alignment_failures = 0

    recording_group_cutoff = (
        apply_late_negative_cutoff
        and resolved_late_cutoff_group_scope == "recording"
        and late_cutoff_strategy == "group-last-positive"
    )
    if recording_group_cutoff:
        late_cutoff_group_onsets, late_cutoff_group_starts, late_cutoff_group_diagnostics = build_canonical_late_cutoff_group_onsets(
            anchors=anchors,
            active_indices=active_indices,
            rows_by_segment=rows_by_segment,
            numerics_segment_ids=segment_ids,
            segment_start_secs_by_id=segment_start_secs_by_id,
            numerics_times=numerics_times,
            numerics=numerics,
            spec=spec,
            channel_lookup=channel_lookup,
            aligned_time_basis=aligned_time_basis,
            numerics_window_time_basis=numerics_window_time_basis,
            min_valid_fraction_per_minute=min_valid_fraction_per_minute,
            late_cutoff_group_scope=resolved_late_cutoff_group_scope,
            input_start_col=input_start_col,
        )

    for group_idx, (segment_id, segment_anchors) in enumerate(grouped, start=1):
        if progress_every and (group_idx == 1 or group_idx % progress_every == 0):
            print(json.dumps({"event": "progress", "segments_done": group_idx - 1, "segments_total": len(grouped)}), flush=True)
        rows = rows_by_segment.get(str(segment_id))
        if rows is None or len(rows) == 0:
            invalid_reasons[segment_anchors.index.to_numpy(dtype=np.int64), :] = int(InvalidReason.NO_NUMERICS)
            continue
        seg_start_secs = segment_start_seconds(segment_anchors, str(segment_id))
        times = convert_segment_times(
            numerics_times[rows],
            seg_start_secs=seg_start_secs,
            source_basis=numerics_window_time_basis,
            target_basis=aligned_time_basis,
        )
        windows = numerics[rows]
        timeline_origin = float((times - windows.shape[2] / 2.0).min())

        min_anchor_start = float(segment_anchors[input_start_col].min())
        max_required = float(segment_anchors[input_end_col].max() + (max(axes) + spec.sustain_minutes) * 60.0)
        numeric_min = float((times - windows.shape[2] / 2.0).min())
        numeric_max = float((times + windows.shape[2] / 2.0).max())
        overlap = max(0.0, min(max_required, numeric_max) - max(min_anchor_start, numeric_min))
        required_span = max(1.0, max_required - min_anchor_start)
        overlap_fraction = overlap / required_span
        if overlap_fraction < 0.01:
            raise ValueError("Anchor and numeric timelines appear to use incompatible time bases")

        anchor_values = segment_anchors[anchor_col].to_numpy(dtype=np.float64)
        nearest = nearest_distances(anchor_values, times)
        segment_diagnostics[str(segment_id)] = {
            "minimum_numeric_timestamp": numeric_min,
            "maximum_numeric_timestamp": numeric_max,
            "minimum_anchor_input_start_timestamp": min_anchor_start,
            "maximum_anchor_input_end_timestamp": float(segment_anchors[input_end_col].max()),
            "maximum_required_outcome_timestamp": max_required,
            "overlap_fraction": float(overlap_fraction),
            "nearest_numeric_timestamp_distance_for_anchor_center": summarize_numeric(nearest),
        }

        input_start_values = segment_anchors[input_start_col].to_numpy(dtype=np.float64)
        residues = np.asarray(np.round(input_start_values), dtype=np.int64) % 60
        origin_by_residue = {
            int(residue): float(input_start_values[residues == residue].min())
            for residue in np.unique(residues)
        }
        timelines: dict[tuple[str, int], MinuteTimeline] = {}
        episodes_by_event: dict[tuple[str, int], list[EventEpisode]] = {}
        for event_name in spec.event_names:
            channel_idx, threshold, comparison = _event_params(event_name, spec, channel_lookup)
            quality_counted = False
            for residue, origin_seconds in origin_by_residue.items():
                timeline = build_minute_timeline_from_windows(
                    window_times=times,
                    windows=windows,
                    channel_idx=channel_idx,
                    threshold=threshold,
                    comparison=comparison,
                    min_valid_fraction_per_minute=min_valid_fraction_per_minute,
                    timeline_origin_seconds=origin_seconds,
                )
                episodes = detect_maximal_episodes(timeline.minute_valid, timeline.event_minutes, spec.sustain_minutes)
                timelines[(event_name, int(residue))] = timeline
                episodes_by_event[(event_name, int(residue))] = episodes
                if not quality_counted:
                    quality = channel_quality[event_name]
                    quality["valid_minutes"] += int(timeline.minute_valid.sum())
                    quality["total_minutes"] += int(len(timeline.minute_valid))
                    quality["finite_sample_count"] += int(timeline.finite_samples_per_minute.sum())
                    quality["total_sample_count"] += int(len(timeline.finite_samples_per_minute) * 60)
                    quality_counted = True

        if negative_exclusion_scope != "none" or resolved_late_cutoff_group_scope is not None:
            event_group_ids: list[str] = []
            if negative_exclusion_scope != "none":
                event_group_ids = sorted({_group_id(anchors.loc[int(idx)], negative_exclusion_scope) for idx in segment_anchors.index.to_numpy(dtype=np.int64)})
                event_group_ids = [gid for gid in event_group_ids if gid]
            late_group_ids: list[str] = []
            if resolved_late_cutoff_group_scope is not None and not recording_group_cutoff:
                late_group_ids = sorted({_group_id(anchors.loc[int(idx)], resolved_late_cutoff_group_scope) for idx in segment_anchors.index.to_numpy(dtype=np.int64)})
                late_group_ids = [gid for gid in late_group_ids if gid]
            for (event_name, residue), episodes_for_event in episodes_by_event.items():
                if not episodes_for_event:
                    continue
                timeline = timelines[(event_name, residue)]
                starts_abs = [timeline.origin_seconds + ep.start_minute_index * 60.0 for ep in episodes_for_event]
                for gid in event_group_ids:
                    event_group_has_event[(event_name, gid)] = True
                for gid in late_group_ids:
                    late_cutoff_group_onsets.setdefault((event_name, gid), []).extend(starts_abs)

        for row in segment_anchors.itertuples():
            row_idx = int(row.Index)
            row_series = anchors.loc[row_idx]
            late_group_id = _group_id(row_series, resolved_late_cutoff_group_scope) if resolved_late_cutoff_group_scope is not None else ""
            if late_group_id and not recording_group_cutoff:
                late_cutoff_group_starts[late_group_id] = min(late_cutoff_group_starts.get(late_group_id, float("inf")), float(row_series[input_start_col]))
            row_residue = int(round(float(row_series[input_start_col]))) % 60
            for event_offset, event_name in enumerate(spec.event_names):
                timeline = timelines[(event_name, row_residue)]
                episodes = episodes_by_event[(event_name, row_residue)]
                try:
                    input_start_min, _ = timestamp_to_minute_index(float(row_series[input_start_col]), timeline.origin_seconds)
                    input_end_min, _ = timestamp_to_minute_index(float(row_series[input_end_col]), timeline.origin_seconds)
                except ValueError:
                    time_alignment_failures += 1
                    for axis_idx in range(len(axes)):
                        col = axis_idx * n_events + event_offset
                        invalid_reasons[row_idx, col] = int(InvalidReason.TIME_ALIGNMENT)
                    continue
                bounds = AnchorMinuteBounds(input_start_min, input_end_min)
                for axis_idx, axis_min in enumerate(axes):
                    col = axis_idx * n_events + event_offset
                    if target_mode == "anchor_onset_within_horizon":
                        state, reason = label_onset_within_horizon(
                            bounds=bounds,
                            minute_valid=timeline.minute_valid,
                            event_minutes=timeline.event_minutes,
                            episodes=episodes,
                            horizon_minutes=int(axis_min),
                            sustain_minutes=spec.sustain_minutes,
                            negative_policy=negative_policy,
                        )
                    else:
                        state, reason = label_fixed_forecast_window(
                            bounds=bounds,
                            minute_valid=timeline.minute_valid,
                            event_minutes=timeline.event_minutes,
                            episodes=episodes,
                            forecast_gap_minutes=int(axis_min),
                            sustain_minutes=spec.sustain_minutes,
                        )
                    base_targets[row_idx, col] = int(state)
                    base_mask[row_idx, col] = state != LabelState.INVALID
                    invalid_reasons[row_idx, col] = int(reason)
                    filtered_targets[row_idx, col] = int(state)
                    filtered_mask[row_idx, col] = state != LabelState.INVALID
                    filter_reasons[row_idx, col] = (
                        int(NegativeFilterReason.KEPT)
                        if state == LabelState.NEGATIVE
                        else int(NegativeFilterReason.POSITIVE)
                        if state == LabelState.POSITIVE
                        else int(NegativeFilterReason.BASE_INVALID)
                    )

    late_cutoff_global_by_event: dict[str, float] = {}
    late_cutoff_by_event_group: dict[tuple[str, str], float] = {}
    late_cutoff_summary: dict[str, object] = {}
    if apply_late_negative_cutoff:
        late_cutoff_global_by_event, late_cutoff_by_event_group, late_cutoff_summary = compute_late_negative_cutoffs(
            positive_group_onsets=late_cutoff_group_onsets,
            group_starts=late_cutoff_group_starts,
            event_names=spec.event_names,
            strategy=late_cutoff_strategy,
        )
    apply_negative_filters(
        anchors=anchors,
        active_indices=active_indices,
        base_targets=base_targets,
        base_mask=base_mask,
        filtered_targets=filtered_targets,
        filtered_mask=filtered_mask,
        filter_reasons=filter_reasons,
        axes=axes,
        event_names=spec.event_names,
        negative_exclusion_scope=negative_exclusion_scope,
        event_group_has_event=event_group_has_event,
        apply_late_negative_cutoff=apply_late_negative_cutoff,
        late_cutoff_candidate=late_cutoff_candidate,
        late_cutoff_group_scope=resolved_late_cutoff_group_scope,
        late_cutoff_strategy=late_cutoff_strategy,
        exclude_late_cutoff_groups_without_positives=exclude_late_cutoff_groups_without_positives,
        late_cutoff_global_by_event=late_cutoff_global_by_event,
        late_cutoff_by_event_group=late_cutoff_by_event_group,
        late_cutoff_group_starts=late_cutoff_group_starts,
        input_end_col=input_end_col,
    )
    assert_final_negative_filter_invariants(
        anchors=anchors,
        active_indices=active_indices,
        base_targets=base_targets,
        base_mask=base_mask,
        filtered_targets=filtered_targets,
        filtered_mask=filtered_mask,
        axes=axes,
        event_names=spec.event_names,
        apply_late_negative_cutoff=apply_late_negative_cutoff,
        late_cutoff_candidate=late_cutoff_candidate,
        late_cutoff_group_scope=resolved_late_cutoff_group_scope,
        late_cutoff_strategy=late_cutoff_strategy,
        exclude_late_cutoff_groups_without_positives=exclude_late_cutoff_groups_without_positives,
        late_cutoff_by_event_group=late_cutoff_by_event_group,
        late_cutoff_group_starts=late_cutoff_group_starts,
        input_end_col=input_end_col,
    )

    split_labels = anchors["split_label"].astype(str).to_numpy() if "split_label" in anchors.columns else None
    for event_name, quality in channel_quality.items():
        resolution = channel_resolution.get(event_name)
        if resolution is None:
            continue
        total_samples = int(quality["total_sample_count"])
        total_minutes = int(quality["total_minutes"])
        resolution["finite_sample_fraction"] = float(quality["finite_sample_count"] / total_samples) if total_samples else float("nan")
        resolution["valid_minute_fraction"] = float(quality["valid_minutes"] / total_minutes) if total_minutes else float("nan")
        resolution["valid_minutes"] = int(quality["valid_minutes"])
        resolution["total_minutes"] = total_minutes
    diagnostics = {
        "label_semantics_version": 2,
        "target_mode": target_mode,
        "time_basis": aligned_time_basis,
        "canonical_bundle_time_basis": "absolute",
        "numeric_source_time_basis": numerics_window_time_basis,
        "label_working_time_basis": aligned_time_basis,
        "anchor_reference": "explicit_input_start_end_from_verified_window_center",
        "window_time_meaning": "center of the 20-minute extracted-feature sequence; feature rows cover [window_time-600, window_time+600) in 20 one-minute rows",
        "interval_endpoint_convention": "half-open [start, end)",
        "input_duration_minutes": 20,
        "sustain_minutes": int(spec.sustain_minutes),
        "negative_policy": negative_policy,
        "negative_exclusion_scope": negative_exclusion_scope,
        "event_history_scope_complete": False,
        "late_negative_cutoff_enabled": bool(apply_late_negative_cutoff),
        "late_cutoff_coordinate": "seconds_since_late_cutoff_group_start",
        "late_cutoff_candidate": late_cutoff_candidate,
        "late_cutoff_group_scope": resolved_late_cutoff_group_scope,
        "late_cutoff_strategy": late_cutoff_strategy,
        "exclude_late_cutoff_groups_without_positives": bool(exclude_late_cutoff_groups_without_positives),
        "late_cutoff_summary": late_cutoff_summary,
        "late_cutoff_group_diagnostics": late_cutoff_group_diagnostics,
        "hypotension_definition": "map-only",
        "minute_validity_rule": f"finite_sample_count >= ceil(60 * {min_valid_fraction_per_minute})",
        "legacy_compatible": False,
        "partial_debug_build": max_segments is not None,
        "max_segments": max_segments,
        "n_anchors": int(len(anchors)),
        "n_processed_segments": int(len(grouped)),
        "channel_resolution": channel_resolution,
        "time_alignment_failures": int(time_alignment_failures),
        "timestamp_alignment_tolerance_seconds": float(timestamp_alignment_tolerance_seconds),
        "numeric_metadata": numerics_metadata,
        "timestamp_metadata": anchors.attrs.get("timestamp_metadata", {}),
        "alignment_audit": alignment_audit,
        "valid_target_cells": int(filtered_mask.sum()),
        "unique_anchors_with_at_least_one_valid_target": int(np.any(filtered_mask, axis=1).sum()),
        "base_counts": _label_counts(base_targets, base_mask, names, split_labels),
        "filtered_counts": _label_counts(filtered_targets, filtered_mask, names, split_labels),
        "base_invalid_reason_counts": _reason_counts(invalid_reasons, InvalidReason),
        "base_invalid_reason_counts_by_target": reason_counts_by_target(invalid_reasons, names, InvalidReason),
        "negative_filter_reason_counts": _reason_counts(filter_reasons, NegativeFilterReason),
        "negative_filter_reason_counts_by_target": reason_counts_by_target(filter_reasons, names, NegativeFilterReason),
        "alignment_audit_by_segment_sample": dict(list(segment_diagnostics.items())[:50]),
        "input_paths": {
            "full_data_root": str(full_data_root),
            "feature_cache_dir": str(anchors.attrs.get("timestamp_metadata", {}).get("feature_cache_dir", "")),
            "feature_cache_anchors": str(anchors.attrs.get("timestamp_metadata", {}).get("feature_cache_anchors", "")),
            "feature_cache_anchor_ids": str(anchors.attrs.get("timestamp_metadata", {}).get("feature_cache_anchor_ids", "")),
            "feature_cache_anchor_times": str(anchors.attrs.get("timestamp_metadata", {}).get("feature_cache_anchor_times", "")),
            "feature_cache_values": str(anchors.attrs.get("timestamp_metadata", {}).get("feature_cache_values", "")),
            "numerics_dir": str(numerics_dir),
        },
        "input_file_hashes_first_4mb": {
            "numerics_metadata": _safe_file_sha256(numerics_dir / "numerics_metadata.json"),
            "anchors_csv": _safe_file_sha256(Path(str(anchors.attrs.get("timestamp_metadata", {}).get("feature_cache_anchors", "")))),
            "anchor_ids": _safe_file_sha256(Path(str(anchors.attrs.get("timestamp_metadata", {}).get("feature_cache_anchor_ids", "")))),
            "anchor_times": _safe_file_sha256(Path(str(anchors.attrs.get("timestamp_metadata", {}).get("feature_cache_anchor_times", "")))),
            "values": _safe_file_sha256(Path(str(anchors.attrs.get("timestamp_metadata", {}).get("feature_cache_values", "")))),
        },
    }
    auxiliary = {
        "base_event_targets": base_targets,
        "base_event_mask": base_mask,
        "base_invalid_reason": invalid_reasons,
        "filtered_event_targets": filtered_targets,
        "filtered_event_mask": filtered_mask,
        "negative_filter_reason": filter_reasons,
        "anchor_time_local": anchors["anchor_time_local"].to_numpy(dtype=np.float64),
        "input_start_time_local": anchors["input_start_time_local"].to_numpy(dtype=np.float64),
        "input_end_time_local": anchors["input_end_time_local"].to_numpy(dtype=np.float64),
        "anchor_time_absolute": anchors["anchor_time_absolute"].to_numpy(dtype=np.float64),
        "input_start_time_absolute": anchors["input_start_time_absolute"].to_numpy(dtype=np.float64),
        "input_end_time_absolute": anchors["input_end_time_absolute"].to_numpy(dtype=np.float64),
    }
    if negative_exclusion_scope == "none" and not apply_late_negative_cutoff and negative_policy == "observable-no-onset":
        if not np.array_equal(base_targets, filtered_targets):
            raise AssertionError("Disabled negative filters changed event targets")
        if not np.array_equal(base_mask, filtered_mask):
            raise AssertionError("Disabled negative filters changed event masks")

    if audit_csv is not None:
        selected_rows = select_audit_row_indices(
            names,
            base_targets,
            base_mask,
            invalid_reasons,
            candidate_rows=active_indices if max_segments is not None else None,
        )
        audit_context = build_audit_context_for_rows(
            selected_rows=selected_rows,
            anchors=anchors,
            segment_ids=segment_ids,
            numerics_times=numerics_times,
            numerics=numerics,
            rows_by_segment=rows_by_segment,
            spec=spec,
            channel_lookup=channel_lookup,
            target_mode=target_mode,
            aligned_time_basis=aligned_time_basis,
            numerics_window_time_basis=numerics_window_time_basis,
            axes=axes,
            min_valid_fraction_per_minute=min_valid_fraction_per_minute,
        )
        write_audit_csv(
            audit_csv,
            anchors,
            names,
            base_targets,
            base_mask,
            invalid_reasons,
            filtered_targets,
            filtered_mask,
            filter_reasons,
            audit_context,
        )
        if audit_csv.exists():
            audit_df = pd.read_csv(audit_csv)
            diagnostics["audit_context_failures"] = int((audit_df.get("audit_context_error", pd.Series(dtype=str)).fillna("").astype(str) != "").sum())
        diagnostics["audit_csv"] = str(audit_csv)
    return filtered_targets, filtered_mask, auxiliary, diagnostics, names



def validate_cli_args(args: argparse.Namespace) -> None:
    if args.sustain_minutes <= 0:
        raise ValueError("--sustain-minutes must be positive")
    if any(h <= 0 for h in args.event_horizons):
        raise ValueError("--event-horizons must all be positive")
    if len(set(args.event_horizons)) != len(args.event_horizons):
        raise ValueError("--event-horizons contains duplicates")
    if any(g <= 0 for g in args.forecast_gaps):
        raise ValueError("--forecast-gaps must all be positive")
    if len(set(args.forecast_gaps)) != len(args.forecast_gaps):
        raise ValueError("--forecast-gaps contains duplicates")
    if not 0.0 < args.min_valid_fraction_per_minute <= 1.0:
        raise ValueError("--min-valid-fraction-per-minute must be in (0, 1]")
    if args.timestamp_alignment_tolerance_seconds < 0:
        raise ValueError("--timestamp-alignment-tolerance-seconds must be nonnegative")
    if args.target_mode == "anchor_fixed_forecast_window" and args.negative_policy != "clean-fixed-window":
        raise ValueError("anchor_fixed_forecast_window requires --negative-policy clean-fixed-window")
    if args.exclude_late_cutoff_groups_without_positives and not args.apply_late_negative_cutoff:
        raise ValueError("--exclude-late-cutoff-groups-without-positives requires --apply-late-negative-cutoff")
    if args.exclude_late_cutoff_groups_without_positives and args.late_cutoff_strategy != "group-last-positive":
        raise ValueError("--exclude-late-cutoff-groups-without-positives requires --late-cutoff-strategy group-last-positive")
    resolve_late_cutoff_group_scope(
        apply_late_negative_cutoff=args.apply_late_negative_cutoff,
        negative_exclusion_scope=args.negative_exclusion_scope,
        late_cutoff_group_scope=args.late_cutoff_group_scope,
    )


def run_timestamp_preflight(anchors: pd.DataFrame, args: argparse.Namespace, spec: EventTaskSpec, axes: list[int]) -> dict[str, object]:
    if args.numerics_source != "aligned-array":
        raise ValueError("--validate-only currently requires --numerics-source aligned-array")
    segment_ids, numerics_times, numerics = load_aligned_numerics(args.numerics_dir, args.full_data_root)
    channel_names = load_aligned_channel_names(args.numerics_dir)
    numeric_metadata = validate_aligned_numerics_cache(args.numerics_dir, numerics, channel_names)
    rows_by_segment = pd.Series(np.arange(len(segment_ids), dtype=np.int64)).groupby(pd.Series(segment_ids), sort=False).agg(list).to_dict()
    rows_by_segment = {str(key): np.asarray(value, dtype=np.int64) for key, value in rows_by_segment.items()}
    audit = audit_anchor_numeric_alignment(
        anchors=anchors,
        segment_ids=segment_ids,
        numerics_times=numerics_times,
        numerics=numerics,
        rows_by_segment=rows_by_segment,
        aligned_time_basis=args.aligned_time_basis,
        numerics_window_time_basis=args.numerics_window_time_basis,
        axes=axes,
        sustain_minutes=spec.sustain_minutes,
        tolerance_seconds=args.timestamp_alignment_tolerance_seconds,
        max_segments=args.max_segments,
    )
    return {
        "validate_only": True,
        "timestamp_metadata": anchors.attrs.get("timestamp_metadata", {}),
        "numeric_source_time_basis": args.numerics_window_time_basis,
        "label_working_time_basis": args.aligned_time_basis,
        "numeric_metadata": numeric_metadata,
        "alignment_audit": audit,
    }


def validate_primary_output_bundle(output_path: Path, target_names: list[str], diagnostics: dict[str, object], *, primary_filters_disabled: bool) -> None:
    bundle = np.load(output_path, allow_pickle=False)
    if primary_filters_disabled:
        if not np.array_equal(bundle["base_event_targets"], bundle["filtered_event_targets"]):
            raise AssertionError("base_event_targets and filtered_event_targets differ with disabled filters")
        if not np.array_equal(bundle["base_event_mask"], bundle["filtered_event_mask"]):
            raise AssertionError("base_event_mask and filtered_event_mask differ with disabled filters")
        if not np.array_equal(bundle["event_targets"], bundle["filtered_event_targets"]):
            raise AssertionError("event_targets and filtered_event_targets differ with disabled filters")
        if not np.array_equal(bundle["event_mask"], bundle["filtered_event_mask"]):
            raise AssertionError("event_mask and filtered_event_mask differ with disabled filters")
    if int(diagnostics.get("time_alignment_failures", -1)) != 0:
        raise AssertionError("time_alignment_failures must equal 0")
    if len(target_names) >= 2 and all("_onset_within_" in name for name in target_names):
        targets = bundle["event_targets"]
        mask = bundle["event_mask"]
        by_event_horizon: dict[tuple[str, int], int] = {}
        for col, name in enumerate(target_names):
            event_name, horizon_text = name.split("_onset_within_", 1)
            by_event_horizon[(event_name, int(horizon_text.removesuffix("m")))] = col
        for event_name in {event for event, _ in by_event_horizon}:
            five_col = by_event_horizon.get((event_name, 5))
            ten_col = by_event_horizon.get((event_name, 10))
            if five_col is None or ten_col is None:
                continue
            violations = (
                mask[:, five_col]
                & mask[:, ten_col]
                & (targets[:, five_col] == int(LabelState.POSITIVE))
                & (targets[:, ten_col] != int(LabelState.POSITIVE))
            )
            if np.any(violations):
                raise AssertionError(f"Valid 5-minute {event_name} positives must also be 10-minute positives")
    if not np.allclose(bundle["anchor_times"], bundle["anchor_time_absolute"]):
        raise AssertionError("anchor_times are not canonical absolute anchor times")
    if not np.allclose(bundle["input_start_times"], bundle["input_start_time_absolute"]):
        raise AssertionError("input_start_times are not canonical absolute times")
    if not np.allclose(bundle["input_end_times"], bundle["input_end_time_absolute"]):
        raise AssertionError("input_end_times are not canonical absolute times")

def main() -> None:
    args = parse_args()
    if args.event_target_generation_mode is not None:
        args.target_mode = args.event_target_generation_mode
    if args.target_mode == "anchor_horizon_filtered":
        args.target_mode = "legacy_anchor_horizon_filtered"
    if args.target_mode == "anchor_horizon":
        args.target_mode = "anchor_horizon"
    validate_cli_args(args)
    if args.max_segments is not None and not args.allow_partial_output:
        output_text = str(args.output).lower()
        if not any(token in output_text for token in ("debug", "smoke", "subset", "partial", "/tmp/")):
            raise ValueError("Refusing to write a partial debug build to a production-looking output path; pass --allow-partial-output or choose a debug output path")

    axes = tuple(args.event_horizons if args.target_mode == "anchor_onset_within_horizon" else args.forecast_gaps)
    spec = EventTaskSpec(
        horizons_min=axes,
        target_generation_mode=args.target_mode,
        hypotension_threshold=DEFAULT_EVENT_TASK.hypotension_threshold,
        tachycardia_threshold=DEFAULT_EVENT_TASK.tachycardia_threshold,
        hypoxia_threshold=DEFAULT_EVENT_TASK.hypoxia_threshold,
        sustain_minutes=args.sustain_minutes,
        hypotension_channel=DEFAULT_EVENT_TASK.hypotension_channel,
        tachycardia_channel=DEFAULT_EVENT_TASK.tachycardia_channel,
        hypoxia_channel=DEFAULT_EVENT_TASK.hypoxia_channel,
        event_names=tuple(args.events),
    )
    anchors = load_cache_anchors(args.feature_cache_dir, args.full_data_root)
    if args.validate_only:
        diagnostics = run_timestamp_preflight(anchors, args, spec, list(axes))
        print(json.dumps(diagnostics, indent=2))
        return
    if args.target_mode in {"legacy_anchor_horizon_filtered", "anchor_horizon"}:
        legacy_mode = "anchor_horizon_filtered" if args.target_mode == "legacy_anchor_horizon_filtered" else "anchor_horizon"
        spec = EventTaskSpec(
            horizons_min=axes,
            target_generation_mode=legacy_mode,
            hypotension_threshold=DEFAULT_EVENT_TASK.hypotension_threshold,
            tachycardia_threshold=DEFAULT_EVENT_TASK.tachycardia_threshold,
            hypoxia_threshold=DEFAULT_EVENT_TASK.hypoxia_threshold,
            sustain_minutes=args.sustain_minutes,
            hypotension_channel=DEFAULT_EVENT_TASK.hypotension_channel,
            tachycardia_channel=DEFAULT_EVENT_TASK.tachycardia_channel,
            hypoxia_channel=DEFAULT_EVENT_TASK.hypoxia_channel,
            event_names=tuple(args.events),
        )
        if args.numerics_source == "waveform-records":
            targets, mask, diagnostics = build_targets_from_waveform_records(anchors, args.waveform_root, spec, args.progress_every, args.max_segments)
        else:
            targets, mask, diagnostics = build_targets_from_aligned_array(anchors, args.numerics_dir, args.full_data_root, spec, args.progress_every, args.max_segments)
        target_names = list(spec.target_names)
        auxiliary = None
    elif args.numerics_source == "waveform-records":
        raise ValueError("Corrected v2 modes currently require --numerics-source aligned-array for extracted-feature classification labels")
    else:
        targets, mask, auxiliary, diagnostics, target_names = build_corrected_targets_from_aligned_array(
            anchors,
            args.numerics_dir,
            args.full_data_root,
            spec,
            target_mode=args.target_mode,
            aligned_time_basis=args.aligned_time_basis,
            negative_policy=args.negative_policy,
            negative_exclusion_scope=args.negative_exclusion_scope,
            apply_late_negative_cutoff=args.apply_late_negative_cutoff,
            late_cutoff_candidate=args.late_cutoff_candidate,
            late_cutoff_group_scope=args.late_cutoff_group_scope,
            late_cutoff_strategy=args.late_cutoff_strategy,
            exclude_late_cutoff_groups_without_positives=args.exclude_late_cutoff_groups_without_positives,
            min_valid_fraction_per_minute=args.min_valid_fraction_per_minute,
            numerics_window_time_basis=args.numerics_window_time_basis,
            timestamp_alignment_tolerance_seconds=args.timestamp_alignment_tolerance_seconds,
            progress_every=args.progress_every,
            max_segments=args.max_segments,
            audit_csv=args.audit_csv or args.output.with_suffix(".audit.csv"),
        )
    if not np.allclose(anchors["anchor_time"], anchors["anchor_time_absolute"]):
        raise AssertionError("anchor_time must be canonical absolute time")
    if not np.allclose(anchors["input_start_time"], anchors["input_start_time_absolute"]):
        raise AssertionError("input_start_time must be canonical absolute time")
    if not np.allclose(anchors["input_end_time"], anchors["input_end_time_absolute"]):
        raise AssertionError("input_end_time must be canonical absolute time")

    save_target_bundle(
        output_path=args.output,
        anchors=anchors,
        feature_targets=None,
        feature_mask=None,
        event_targets=targets,
        event_mask=mask,
        event_spec=spec,
        event_auxiliary_arrays=auxiliary,
        event_diagnostics=diagnostics,
    )
    metadata_path = args.output.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["event_target_names"] = target_names
    metadata["source"] = {
        "full_data_root": str(args.full_data_root),
        "feature_cache_dir": str(args.feature_cache_dir),
        "numerics_source": args.numerics_source,
        "numerics_dir": str(args.numerics_dir),
        "waveform_root": str(args.waveform_root),
        "clinical_vitals_dir_note": "/gpfs/data/eh3828lab/datasets/mimic_clinical contains MIMIC clinical tables; this builder uses bedside monitor waveform numerics to preserve prior classification semantics.",
    }
    metadata["event_spec"] = asdict(spec)
    metadata["timestamp_metadata"] = anchors.attrs.get("timestamp_metadata", {})
    metadata["command_line_args"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata_path.write_text(json.dumps(metadata, indent=2))
    primary_filters_disabled = (
        args.target_mode == "anchor_onset_within_horizon"
        and args.negative_policy == "observable-no-onset"
        and args.negative_exclusion_scope == "none"
        and not args.apply_late_negative_cutoff
    )
    if auxiliary is not None:
        validate_primary_output_bundle(args.output, target_names, diagnostics, primary_filters_disabled=primary_filters_disabled)
    print(json.dumps({"output": str(args.output), "shape": list(targets.shape), "valid_values": int(mask.sum()), "target_names": target_names, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
