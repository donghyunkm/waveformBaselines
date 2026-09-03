#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT))


DEFAULT_WAVEFORM_ROOT = Path("/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched")
DEFAULT_MIMIC_CLINICAL_DIR = Path("/gpfs/data/eh3828lab/datasets/mimic_clinical")
DEFAULT_OUTPUT = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/"
    "segment_level_vasopressor_free_waveform_manifest.csv"
)
DEFAULT_QC_OUTPUT = DEFAULT_OUTPUT.with_suffix(".qc.json")
DEFAULT_FREE_SEGMENTS_OUTPUT = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_free_segments.txt")
DEFAULT_INTERVALS_OUTPUT = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_vasopressor_intervals.csv")
DEFAULT_UNCERTAIN_OUTPUT = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_uncertain_vasopressor_evidence.csv")

MV_VASOPRESSOR_ITEMIDS = [
    221906,
    221289,
    221749,
    222315,
    221662,
    221653,
    221986,
]

CV_VASOPRESSOR_ITEMIDS = [
    30047,
    30120,
    30044,
    30119,
    30309,
    30127,
    30128,
    30051,
    42273,
    42802,
    30043,
    30307,
    30042,
    30306,
    30125,
]

MV_ITEMID_TO_DRUG = {
    221906: "norepinephrine",
    221289: "epinephrine",
    221749: "phenylephrine",
    221662: "dopamine",
    221653: "dobutamine",
    221986: "milrinone",
    222315: "vasopressin",
}

CV_ITEMID_TO_DRUG = {
    30047: "norepinephrine",
    30120: "norepinephrine",
    30044: "epinephrine",
    30119: "epinephrine",
    30309: "epinephrine",
    30127: "phenylephrine",
    30128: "phenylephrine",
    30043: "dopamine",
    30307: "dopamine",
    30042: "dobutamine",
    30306: "dobutamine",
    30125: "milrinone",
    30051: "vasopressin",
    42273: "vasopressin",
    42802: "vasopressin",
}

assert set(MV_ITEMID_TO_DRUG) == set(MV_VASOPRESSOR_ITEMIDS)
assert set(CV_ITEMID_TO_DRUG) == set(CV_VASOPRESSOR_ITEMIDS)

AUTHORITATIVE_CV_REFERENCE = "vasopressorML.startendtime / MIMIC-III concepts/durations/vasopressor_durations.sql"
AUTHORITATIVE_MV_REFERENCE = "vasopressorML.mvextraction / MIMIC-III vasopressor concept"
SPECIAL_CV_RATE_AMOUNT_SWAPPED_ITEMIDS = {42273, 42802} & set(CV_VASOPRESSOR_ITEMIDS)
HEADER_RECORD_RE = re.compile(r"^(?P<subject>p\d{6})-(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-(?P<hour>\d{2})-(?P<minute>\d{2})(?P<numerics>n)?$")
SEGMENT_RE = re.compile(r"^\d+_\d+$")
HALF_OPEN_OVERLAP_RULE = "start_a < end_b and end_a > start_b"
METADATA_WFDB_TIMESTAMP_TOLERANCE_SECONDS = 1.0
ICU_CONTAINMENT_TOLERANCE_SECONDS = 1.0


@dataclass(frozen=True)
class WaveformSegment:
    segment_id: str
    segment_name: str
    segment_path: str
    subject_id: int
    subject_id_str: str
    parent_record: str | None
    segment_start_time: pd.Timestamp | pd.NaT
    segment_end_time: pd.Timestamp | pd.NaT
    segment_duration_seconds: float | None
    waveform_fs_hz: float | None
    waveform_samples: int | None
    waveform_channels: int | None
    timestamp_status: str
    wfdb_segment_start_time: pd.Timestamp | pd.NaT
    metadata_wfdb_timestamp_delta_seconds: float | None
    metadata_wfdb_timestamp_abs_delta_seconds: float | None
    metadata_wfdb_timestamp_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a segment-level vasopressor-free manifest for MIMIC-III matched waveform segments.")
    parser.add_argument("--waveform-root", type=Path, default=DEFAULT_WAVEFORM_ROOT)
    parser.add_argument("--mimic-clinical-dir", type=Path, default=DEFAULT_MIMIC_CLINICAL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qc-output", type=Path, default=DEFAULT_QC_OUTPUT)
    parser.add_argument("--free-segments-output", type=Path, default=DEFAULT_FREE_SEGMENTS_OUTPUT)
    parser.add_argument("--free-records-output", type=Path, default=None, help="Backward-compatible alias for --free-segments-output.")
    parser.add_argument("--vasopressor-intervals-output", type=Path, default=DEFAULT_INTERVALS_OUTPUT)
    parser.add_argument("--uncertain-vasopressor-output", type=Path, default=DEFAULT_UNCERTAIN_OUTPUT)
    parser.add_argument("--segment-metadata-json", type=Path, default=None)
    parser.add_argument("--max-patient-dirs", type=int, default=None)
    return parser.parse_args()


def _parse_header_first_line(header_path: Path) -> dict[str, object]:
    try:
        with header_path.open(errors="replace") as handle:
            first = handle.readline().strip()
    except OSError:
        first = ""
    if not first:
        return {"record_name": header_path.stem, "n_sig": None, "fs": None, "samples": None, "base_time": None, "base_date": None}
    parts = first.split()
    n_sig = fs = samples = None
    try:
        n_sig = int(parts[1].split("/", 1)[0]) if len(parts) > 1 else None
    except ValueError:
        pass
    try:
        fs = float(parts[2]) if len(parts) > 2 else None
    except ValueError:
        pass
    try:
        samples = int(float(parts[3])) if len(parts) > 3 else None
    except ValueError:
        pass
    return {
        "record_name": parts[0].split("/", 1)[0] if parts else header_path.stem,
        "n_sig": n_sig,
        "fs": fs,
        "samples": samples,
        "base_time": parts[4] if len(parts) > 4 else None,
        "base_date": parts[5] if len(parts) > 5 else None,
    }


def _timestamp_from_header_fields(base_time: object, base_date: object) -> pd.Timestamp | pd.NaT:
    if not base_time or not base_date:
        return pd.NaT
    parsed = pd.to_datetime(f"{base_date} {base_time}", dayfirst=True, errors="coerce")
    return pd.Timestamp(parsed) if pd.notna(parsed) else pd.NaT


def _parse_parent_start(record_name: str, header_path: Path | None = None) -> pd.Timestamp | pd.NaT:
    if header_path is not None and header_path.exists():
        info = _parse_header_first_line(header_path)
        start = _timestamp_from_header_fields(info["base_time"], info["base_date"])
        if pd.notna(start):
            return start
    match = HEADER_RECORD_RE.match(record_name)
    if not match or match.group("numerics"):
        return pd.NaT
    parts = match.groupdict()
    return pd.Timestamp(int(parts["year"]), int(parts["month"]), int(parts["day"]), int(parts["hour"]), int(parts["minute"]))


def _segments_from_parent_header(parent_header: Path) -> dict[str, tuple[str, int]]:
    parent_name = parent_header.stem
    out: dict[str, tuple[str, int]] = {}
    cumulative = 0
    for line in parent_header.read_text(errors="replace").splitlines()[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            segment_samples = int(float(parts[1]))
        except ValueError:
            continue
        if SEGMENT_RE.match(parts[0]):
            out[parts[0]] = (parent_name, cumulative)
        cumulative += max(segment_samples, 0)
    return out


def _segment_from_parts(
    subject_id_str: str,
    segment_name: str,
    segment_path: Path,
    parent_record: str | None,
    start: pd.Timestamp | pd.NaT,
    fs: float | None,
    samples: int | None,
    n_sig: int | None,
    timestamp_status: str,
    wfdb_segment_start_time: pd.Timestamp | pd.NaT = pd.NaT,
    metadata_wfdb_timestamp_delta_seconds: float | None = None,
    metadata_wfdb_timestamp_abs_delta_seconds: float | None = None,
    metadata_wfdb_timestamp_status: str = "not_compared",
) -> WaveformSegment:
    duration = float(samples) / float(fs) if fs and samples is not None and fs > 0 else None
    if pd.notna(start) and duration is not None and duration > 0:
        end = pd.Timestamp(start) + pd.to_timedelta(duration, unit="s")
    else:
        end = pd.NaT
        if timestamp_status == "valid":
            timestamp_status = "invalid_duration"
    return WaveformSegment(
        segment_id=f"{subject_id_str}/{segment_name}",
        segment_name=segment_name,
        segment_path=str(segment_path),
        subject_id=int(subject_id_str.lstrip("p")),
        subject_id_str=subject_id_str,
        parent_record=parent_record,
        segment_start_time=pd.Timestamp(start) if pd.notna(start) else pd.NaT,
        segment_end_time=end,
        segment_duration_seconds=duration,
        waveform_fs_hz=float(fs) if fs is not None else None,
        waveform_samples=int(samples) if samples is not None else None,
        waveform_channels=int(n_sig) if n_sig is not None else None,
        timestamp_status=timestamp_status,
        wfdb_segment_start_time=pd.Timestamp(wfdb_segment_start_time) if pd.notna(wfdb_segment_start_time) else pd.NaT,
        metadata_wfdb_timestamp_delta_seconds=metadata_wfdb_timestamp_delta_seconds,
        metadata_wfdb_timestamp_abs_delta_seconds=metadata_wfdb_timestamp_abs_delta_seconds,
        metadata_wfdb_timestamp_status=metadata_wfdb_timestamp_status,
    )


def load_waveform_segments(waveform_root: Path, max_patient_dirs: int | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    records: list[WaveformSegment] = []
    patient_dirs = sorted(path for path in waveform_root.glob("p??/p??????") if path.is_dir())
    if max_patient_dirs is not None:
        patient_dirs = patient_dirs[:max_patient_dirs]
    for patient_dir in patient_dirs:
        subject_id_str = patient_dir.name
        parent_starts: dict[str, pd.Timestamp | pd.NaT] = {}
        segment_offsets: dict[str, tuple[str, int]] = {}
        records_path = patient_dir / "RECORDS"
        for line in records_path.read_text(errors="replace").splitlines() if records_path.exists() else []:
            name = line.strip()
            if not name or name.endswith("n"):
                continue
            if HEADER_RECORD_RE.match(name):
                header_path = patient_dir / f"{name}.hea"
                parent_starts[name] = _parse_parent_start(name, header_path)
                if header_path.exists():
                    segment_offsets.update(_segments_from_parent_header(header_path))
        for header_path in sorted(patient_dir.glob("*.hea")):
            segment_name = header_path.stem
            if segment_name.endswith("_layout") or segment_name.endswith("n") or HEADER_RECORD_RE.match(segment_name) or not SEGMENT_RE.match(segment_name):
                continue
            info = _parse_header_first_line(header_path)
            parent_record = None
            start = pd.NaT
            timestamp_status = "missing_parent_segment_mapping"
            if segment_name in segment_offsets:
                parent_record, offset_samples = segment_offsets[segment_name]
                parent_start = parent_starts.get(parent_record, pd.NaT)
                fs = info["fs"]
                if pd.notna(parent_start) and fs and fs > 0:
                    start = parent_start + pd.to_timedelta(offset_samples / float(fs), unit="s")
                    timestamp_status = "valid"
                else:
                    timestamp_status = "missing_parent_start"
            records.append(_segment_from_parts(subject_id_str, segment_name, header_path.with_suffix(""), parent_record, start, info["fs"], info["samples"], info["n_sig"], timestamp_status))
    qc = {"segment_metadata_duplicate_group_size_counts": {}, "segment_metadata_duplicate_examples": []}
    return pd.DataFrame([record.__dict__ for record in records]), qc


def load_waveform_segments_from_segment_metadata(segment_metadata_json: Path, waveform_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = json.loads(segment_metadata_json.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{segment_metadata_json} must contain a list of segment metadata rows")
    metadata = pd.DataFrame(rows)
    group_sizes = metadata.groupby(["patient_id", "seg_name"]).size()
    duplicate_groups = group_sizes[group_sizes > 1]
    qc = {
        "segment_metadata_input_rows": int(len(metadata)),
        "segment_metadata_unique_segments": int(len(group_sizes)),
        "segment_metadata_duplicate_group_size_counts": {str(int(k)): int(v) for k, v in group_sizes.value_counts().sort_index().items()},
        "segment_metadata_duplicate_examples": [{"patient_id": str(idx[0]), "segment_name": str(idx[1]), "row_count": int(count)} for idx, count in duplicate_groups.head(10).items()],
        "segment_metadata_deduplication_rule": "segment-level manifest; duplicate patient_id/seg_name rows are treated as the same physical waveform segment",
    }
    metadata = metadata.drop_duplicates(["patient_id", "seg_name"], keep="first")
    parent_cache: dict[str, tuple[dict[str, pd.Timestamp | pd.NaT], dict[str, tuple[str, int]]]] = {}
    timestamp_deltas: list[dict[str, object]] = []
    records: list[WaveformSegment] = []
    for row in metadata.itertuples(index=False):
        patient_id = str(row.patient_id)
        segment_name = str(row.seg_name)
        record_base = waveform_root / patient_id[:3] / patient_id / segment_name
        header_path = record_base.with_suffix(".hea")
        info = _parse_header_first_line(header_path) if header_path.exists() else {"n_sig": None, "fs": None, "samples": None}
        start = pd.to_datetime(float(row.seg_start_secs), unit="s", origin=pd.Timestamp("2000-01-01"), errors="coerce")
        wfdb_start = pd.NaT
        delta_seconds = None
        abs_delta_seconds = None
        comparison_status = "not_compared"
        parent_record = None
        patient_dir = waveform_root / patient_id[:3] / patient_id
        if patient_id not in parent_cache:
            parent_starts: dict[str, pd.Timestamp | pd.NaT] = {}
            segment_offsets: dict[str, tuple[str, int]] = {}
            records_path = patient_dir / "RECORDS"
            for name in records_path.read_text(errors="replace").splitlines() if records_path.exists() else []:
                name = name.strip()
                if not name or name.endswith("n"):
                    continue
                if HEADER_RECORD_RE.match(name):
                    parent_header = patient_dir / f"{name}.hea"
                    parent_starts[name] = _parse_parent_start(name, parent_header)
                    if parent_header.exists():
                        segment_offsets.update(_segments_from_parent_header(parent_header))
            parent_cache[patient_id] = (parent_starts, segment_offsets)
        parent_starts, segment_offsets = parent_cache[patient_id]
        if segment_name in segment_offsets and info.get("fs"):
            parent_record, offset_samples = segment_offsets[segment_name]
            parent_start = parent_starts.get(parent_record, pd.NaT)
            if pd.notna(start) and pd.notna(parent_start):
                wfdb_start = parent_start + pd.to_timedelta(offset_samples / float(info["fs"]), unit="s")
                delta_seconds = float((pd.Timestamp(start) - pd.Timestamp(wfdb_start)).total_seconds())
                abs_delta_seconds = abs(delta_seconds)
                comparison_status = "matched" if abs_delta_seconds <= METADATA_WFDB_TIMESTAMP_TOLERANCE_SECONDS else "mismatch"
                timestamp_deltas.append({
                    "segment_id": f"{patient_id}/{segment_name}",
                    "metadata_segment_start": pd.Timestamp(start),
                    "wfdb_segment_start": pd.Timestamp(wfdb_start),
                    "timestamp_delta_seconds": delta_seconds,
                })
        timestamp_status = "valid" if pd.notna(start) else "invalid_segment_metadata_start"
        if timestamp_status == "valid" and comparison_status == "mismatch":
            timestamp_status = "metadata_wfdb_timestamp_mismatch"
        records.append(
            _segment_from_parts(
                patient_id,
                segment_name,
                record_base,
                parent_record,
                pd.Timestamp(start) if pd.notna(start) else pd.NaT,
                info["fs"],
                info["samples"],
                info["n_sig"],
                timestamp_status,
                pd.Timestamp(wfdb_start) if pd.notna(wfdb_start) else pd.NaT,
                delta_seconds,
                abs_delta_seconds,
                comparison_status,
            )
        )
    if timestamp_deltas:
        delta = pd.DataFrame(timestamp_deltas)
        abs_delta = delta["timestamp_delta_seconds"].abs()
        qc["metadata_vs_wfdb_timestamp_comparison"] = {
            "n_compared": int(len(delta)),
            "median_abs_delta_seconds": float(abs_delta.median()),
            "p95_abs_delta_seconds": float(abs_delta.quantile(0.95)),
            "p99_abs_delta_seconds": float(abs_delta.quantile(0.99)),
            "max_abs_delta_seconds": float(abs_delta.max()),
            "fraction_within_1_second": float((abs_delta <= 1.0).mean()),
            "fraction_within_1_minute": float((abs_delta <= 60.0).mean()),
            "tolerance_seconds": METADATA_WFDB_TIMESTAMP_TOLERANCE_SECONDS,
            "matched_count": int((abs_delta <= METADATA_WFDB_TIMESTAMP_TOLERANCE_SECONDS).sum()),
            "mismatch_count": int((abs_delta > METADATA_WFDB_TIMESTAMP_TOLERANCE_SECONDS).sum()),
            "worst_examples": delta.assign(abs_delta_seconds=abs_delta).sort_values("abs_delta_seconds", ascending=False).head(10).to_dict(orient="records"),
        }
    else:
        qc["metadata_vs_wfdb_timestamp_comparison"] = {"n_compared": 0}
    return pd.DataFrame([record.__dict__ for record in records]), qc


def validate_itemid_mappings(mimic_dir: Path) -> dict[str, object]:
    if set(MV_ITEMID_TO_DRUG) != set(MV_VASOPRESSOR_ITEMIDS):
        raise ValueError("MetaVision ITEMID mapping does not match imported vasopressor ITEMID set")
    if set(CV_ITEMID_TO_DRUG) != set(CV_VASOPRESSOR_ITEMIDS):
        raise ValueError("CareVue ITEMID mapping does not match imported vasopressor ITEMID set")
    itemids = sorted(set(MV_VASOPRESSOR_ITEMIDS) | set(CV_VASOPRESSOR_ITEMIDS))
    d_items = pd.read_csv(mimic_dir / "D_ITEMS.csv.gz", usecols=["ITEMID", "LABEL", "DBSOURCE", "LINKSTO"], dtype={"ITEMID": int, "LABEL": str, "DBSOURCE": str, "LINKSTO": str})
    found = d_items[d_items["ITEMID"].isin(itemids)].sort_values("ITEMID")
    missing = sorted(set(itemids) - set(found["ITEMID"].tolist()))
    if missing:
        raise ValueError(f"Configured vasopressor ITEMIDs missing from D_ITEMS: {missing}")
    duplicate_mapped_ids = [itemid for itemid, count in Counter(list(MV_ITEMID_TO_DRUG) + list(CV_ITEMID_TO_DRUG)).items() if count != 1]
    if duplicate_mapped_ids:
        raise ValueError(f"Configured ITEMIDs do not map exactly once: {duplicate_mapped_ids}")
    return {
        "missing_itemids_in_d_items": missing,
        "itemid_labels": found.to_dict(orient="records"),
        "carevue_authoritative_reference": AUTHORITATIVE_CV_REFERENCE,
        "metavision_authoritative_reference": AUTHORITATIVE_MV_REFERENCE,
        "carevue_contains_30309_epinephrine": 30309 in set(CV_VASOPRESSOR_ITEMIDS) and CV_ITEMID_TO_DRUG.get(30309) == "epinephrine",
    }


def _uncertainty_reasons(frame: pd.DataFrame) -> pd.Series:
    reasons = pd.Series("", index=frame.index, dtype=object)
    checks = {
        "missing_icustay_id": frame.get("ICUSTAY_ID", pd.Series(index=frame.index, dtype=object)).isna(),
        "missing_linkorderid": frame.get("LINKORDERID", pd.Series(index=frame.index, dtype=object)).isna(),
        "missing_starttime": frame.get("STARTTIME", pd.Series(index=frame.index, dtype=object)).isna(),
        "missing_endtime": frame.get("ENDTIME", pd.Series(index=frame.index, dtype=object)).isna(),
        "non_positive_duration": frame.get("STARTTIME", pd.Series(index=frame.index, dtype=object)).notna()
        & frame.get("ENDTIME", pd.Series(index=frame.index, dtype=object)).notna()
        & (frame.get("ENDTIME") <= frame.get("STARTTIME")),
    }
    for label, mask in checks.items():
        reasons = reasons.mask(mask, reasons.where(reasons.eq(""), reasons + ";") + label)
    return reasons.mask(reasons.eq(""), "other_unresolved_timing")


def load_mv_vasopressor_intervals(mimic_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    usecols = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID", "LINKORDERID", "STATUSDESCRIPTION", "AMOUNT", "RATE", "CANCELREASON"]
    mv = pd.read_csv(mimic_dir / "INPUTEVENTS_MV.csv.gz", usecols=usecols, dtype={"ROW_ID": int, "SUBJECT_ID": int, "HADM_ID": "Int64", "ICUSTAY_ID": "Int64", "ITEMID": int, "LINKORDERID": "Int64", "STATUSDESCRIPTION": str, "CANCELREASON": "Int64"}, parse_dates=["STARTTIME", "ENDTIME"])
    mv = mv[mv["ITEMID"].isin(MV_VASOPRESSOR_ITEMIDS)].copy()
    qc: dict[str, object] = {
        "metavision_raw_rows": int(len(mv)),
        "metavision_status_counts_before_filter": {str(k): int(v) for k, v in mv["STATUSDESCRIPTION"].value_counts(dropna=False).items()},
        "metavision_status_filter_rule": "retain all configured vasopressor rows except STATUSDESCRIPTION == 'Rewritten'; valid intervals require ICUSTAY_ID, LINKORDERID, STARTTIME, and ENDTIME; unresolved retained rows are propagated as uncertain evidence",
    }
    retained = mv[mv["STATUSDESCRIPTION"].ne("Rewritten")].copy()
    qc["metavision_rows_excluded_by_status"] = int(len(mv) - len(retained))
    qc["metavision_status_counts_after_filter"] = {str(k): int(v) for k, v in retained["STATUSDESCRIPTION"].value_counts(dropna=False).items()}
    retained["vasopressor_name"] = retained["ITEMID"].map(MV_ITEMID_TO_DRUG)
    missing_required = retained[["ICUSTAY_ID", "LINKORDERID", "STARTTIME", "ENDTIME"]].isna().any(axis=1)
    non_positive = retained["STARTTIME"].notna() & retained["ENDTIME"].notna() & (retained["ENDTIME"] <= retained["STARTTIME"])
    qc["metavision_rows_missing_required_fields"] = int(missing_required.sum())
    qc["metavision_rows_zero_or_negative_duration"] = int(non_positive.sum())
    uncertain = retained.loc[missing_required | non_positive].copy()
    if not uncertain.empty:
        uncertain["vasopressor_source"] = "metavision"
        uncertain["uncertainty_reason"] = _uncertainty_reasons(uncertain)
        uncertain["evidence_start"] = uncertain["STARTTIME"]
        uncertain["evidence_end"] = uncertain["ENDTIME"]
        uncertain["evidence_time"] = pd.NaT
        uncertain = uncertain[["ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID", "vasopressor_name", "STATUSDESCRIPTION", "vasopressor_source", "uncertainty_reason", "evidence_start", "evidence_end", "evidence_time"]]
    else:
        uncertain = pd.DataFrame(columns=["ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID", "vasopressor_name", "STATUSDESCRIPTION", "vasopressor_source", "uncertainty_reason", "evidence_start", "evidence_end", "evidence_time"])
    valid = retained.loc[~missing_required & ~non_positive].copy()
    rows = []
    for (icustay_id, linkorderid), group in valid.groupby(["ICUSTAY_ID", "LINKORDERID"], sort=False):
        rows.append({
            "SUBJECT_ID": int(group["SUBJECT_ID"].iloc[0]),
            "HADM_ID": int(group["HADM_ID"].iloc[0]) if pd.notna(group["HADM_ID"].iloc[0]) else pd.NA,
            "ICUSTAY_ID": int(icustay_id),
            "ITEMID": ",".join(str(x) for x in sorted(group["ITEMID"].astype(int).unique())),
            "vasopressor_name": ",".join(sorted(group["vasopressor_name"].dropna().unique())),
            "vasopressor_source": "metavision",
            "vaso_start": group["STARTTIME"].min(),
            "vaso_end": group["ENDTIME"].max(),
            "source_order_id": int(linkorderid),
            "source_row_count": int(len(group)),
            "episode_uncertain_end": False,
        })
    episodes = pd.DataFrame(rows)
    if not episodes.empty:
        episodes["duration_minutes"] = (episodes["vaso_end"] - episodes["vaso_start"]).dt.total_seconds() / 60.0
        episodes = episodes.sort_values(["ICUSTAY_ID", "vaso_start", "vaso_end"]).reset_index(drop=True)
    qc["metavision_retained_intervals"] = int(len(episodes))
    qc["uncertain_metavision_rows"] = int(len(uncertain))
    qc["uncertain_metavision_rows_by_reason"] = {str(k): int(v) for k, v in uncertain["uncertainty_reason"].value_counts(dropna=False).items()}
    return episodes, uncertain, qc


def load_cv_vasopressor_rows(mimic_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    usecols = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID", "RATE", "AMOUNT", "ORDERID", "LINKORDERID", "STOPPED"]
    cv = pd.read_csv(mimic_dir / "INPUTEVENTS_CV.csv.gz", usecols=usecols, dtype={"ROW_ID": int, "SUBJECT_ID": int, "HADM_ID": "Int64", "ICUSTAY_ID": "Int64", "ITEMID": int, "ORDERID": "Int64", "LINKORDERID": "Int64", "STOPPED": str}, parse_dates=["CHARTTIME"])
    cv = cv[cv["ITEMID"].isin(CV_VASOPRESSOR_ITEMIDS)].copy()
    cv["RATE"] = pd.to_numeric(cv["RATE"], errors="coerce")
    cv["AMOUNT"] = pd.to_numeric(cv["AMOUNT"], errors="coerce")
    if SPECIAL_CV_RATE_AMOUNT_SWAPPED_ITEMIDS:
        special = cv["ITEMID"].isin(SPECIAL_CV_RATE_AMOUNT_SWAPPED_ITEMIDS)
        old_rate = cv.loc[special, "RATE"].copy()
        cv.loc[special, "RATE"] = cv.loc[special, "AMOUNT"].values
        cv.loc[special, "AMOUNT"] = old_rate.values
    qc = {
        "carevue_raw_rows": int(len(cv)),
        "carevue_stop_status_counts": {str(k): int(v) for k, v in cv["STOPPED"].value_counts(dropna=False).items()},
        "carevue_rows_missing_icustay_id": int(cv["ICUSTAY_ID"].isna().sum()),
        "carevue_rows_missing_charttime": int(cv["CHARTTIME"].isna().sum()),
        "carevue_orderid_count": int(cv["ORDERID"].nunique(dropna=True)),
        "carevue_linkorderid_count": int(cv["LINKORDERID"].nunique(dropna=True)),
        "carevue_rows_with_orderid_ne_linkorderid": int((cv["ORDERID"] != cv["LINKORDERID"]).sum()),
        "carevue_interval_rule": "MIMIC-III vasopressor_durations CareVue logic adapted to clinical drug-level grouping: map ITEMID to vasopressor_name, group chart rows by ICUSTAY_ID/CHARTTIME/vasopressor_name, use max RATE as the drug state at duplicate timestamps, carry forward RATE, start on positive-rate transitions or after stop, end on explicit stop, zero rate, or final chart row",
    }
    return cv, qc


def _join_id_strings(series: pd.Series) -> str:
    values = []
    for value in series.dropna().tolist():
        for part in str(int(value)).split(","):
            if part:
                values.append(part)
    return ",".join(sorted(set(values)))


def reconstruct_cv_vasopressor_intervals(cv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    valid_rows = cv[cv["ICUSTAY_ID"].notna() & cv["CHARTTIME"].notna()].copy()
    uncertain_rows = cv[cv["ICUSTAY_ID"].isna() | cv["CHARTTIME"].isna()].copy()
    if valid_rows.empty:
        qc = {"carevue_rows_linked_into_episodes": 0, "carevue_reconstructed_itemid_episodes": 0, "carevue_reconstructed_any_vasopressor_episodes": 0}
        return pd.DataFrame(), pd.DataFrame(), qc, uncertain_rows
    x = valid_rows.copy()
    x["ICUSTAY_ID"] = x["ICUSTAY_ID"].astype(int)
    x["ITEMID"] = x["ITEMID"].astype(int)
    x["vasopressor_name"] = x["ITEMID"].map(CV_ITEMID_TO_DRUG)
    stopped_text = x["STOPPED"].fillna("").astype(str)
    x["_vaso_stopped_raw"] = (stopped_text.eq("Stopped") | stopped_text.str.startswith("D/C")).astype(int)
    x["_active_positive_rate"] = (x["RATE"].gt(0) & x["_vaso_stopped_raw"].eq(0)).astype(int)
    x["_any_positive_rate_raw"] = x["RATE"].gt(0).astype(int)
    x["_any_zero_rate_raw"] = x["RATE"].eq(0).astype(int)
    x["_vaso_null"] = x["RATE"].notna().astype(int)
    x = x.groupby(["ICUSTAY_ID", "CHARTTIME", "vasopressor_name"], as_index=False).agg(
        SUBJECT_ID=("SUBJECT_ID", "first"),
        HADM_ID=("HADM_ID", "first"),
        source_itemids=("ITEMID", lambda s: ",".join(str(v) for v in sorted(s.astype(int).unique()))),
        source_rates=("RATE", lambda s: ",".join("NA" if pd.isna(v) else str(float(v)) for v in s.tolist())),
        source_stopped_values=("STOPPED", lambda s: ",".join("NA" if pd.isna(v) else str(v) for v in s.tolist())),
        source_row_count=("ROW_ID", "count"),
        source_order_ids=("ORDERID", _join_id_strings),
        source_linkorder_ids=("LINKORDERID", _join_id_strings),
        any_stopped=("_vaso_stopped_raw", "max"),
        any_active_positive=("_active_positive_rate", "max"),
        any_positive_rate=("_any_positive_rate_raw", "max"),
        any_zero_rate=("_any_zero_rate_raw", "max"),
        vaso_null=("_vaso_null", "max"),
        vaso_rate=("RATE", "max"),
        vaso_amount=("AMOUNT", "max"),
    ).sort_values(["ICUSTAY_ID", "vasopressor_name", "CHARTTIME"]).reset_index(drop=True)
    x["vaso_stopped"] = (x["any_stopped"].eq(1) & x["any_active_positive"].eq(0)).astype(int)
    stop_active_conflicts = x[x["any_stopped"].eq(1) & x["any_active_positive"].eq(1)].copy()
    zero_positive_conflicts = x[x["any_zero_rate"].eq(1) & x["any_positive_rate"].eq(1)].copy()
    x["vaso_partition"] = x.groupby(["ICUSTAY_ID", "vasopressor_name"], sort=False)["vaso_null"].cumsum()
    x["vaso_prevrate_ifnull"] = x.groupby(["ICUSTAY_ID", "vasopressor_name", "vaso_partition"], sort=False, dropna=False)["vaso_rate"].transform(lambda s: s.iloc[0])
    x["_lag_rate_same_nullflag"] = x.groupby(["ICUSTAY_ID", "vasopressor_name", "vaso_null"], sort=False)["vaso_prevrate_ifnull"].shift(1)
    x["_lag_rate"] = x.groupby(["ICUSTAY_ID", "vasopressor_name"], sort=False)["vaso_prevrate_ifnull"].shift(1)
    x["_lag_stopped"] = x.groupby(["ICUSTAY_ID", "vasopressor_name"], sort=False)["vaso_stopped"].shift(1)
    x["vaso_start_flag"] = np.nan
    x.loc[x["vaso_rate"].gt(0) & x["_lag_rate_same_nullflag"].isna(), "vaso_start_flag"] = 1
    x.loc[x["vaso_start_flag"].isna() & x["vaso_rate"].eq(0) & x["_lag_rate"].eq(0), "vaso_start_flag"] = 0
    x.loc[x["vaso_start_flag"].isna() & x["vaso_prevrate_ifnull"].eq(0) & x["_lag_rate"].eq(0), "vaso_start_flag"] = 0
    x.loc[x["vaso_start_flag"].isna() & x["_lag_rate"].eq(0), "vaso_start_flag"] = 1
    x.loc[x["vaso_start_flag"].isna() & x["_lag_stopped"].eq(1), "vaso_start_flag"] = 1
    x["vaso_first"] = x["vaso_start_flag"].fillna(0).groupby([x["ICUSTAY_ID"], x["vasopressor_name"]]).cumsum()
    x["_next_charttime"] = x.groupby(["ICUSTAY_ID", "vasopressor_name"], sort=False)["CHARTTIME"].shift(-1)
    stop_condition = x["vaso_stopped"].eq(1) | x["vaso_rate"].eq(0) | x["_next_charttime"].isna()
    x["vaso_stop"] = np.where(stop_condition, x["vaso_first"], np.nan)
    episode_rows = x[x["vaso_first"].notna() & x["vaso_first"].ne(0)].copy()
    episode_rows["_start_candidate"] = episode_rows["CHARTTIME"].where(episode_rows["vaso_rate"].notna())
    episode_rows["_end_candidate"] = episode_rows["CHARTTIME"].where(episode_rows["vaso_first"].eq(episode_rows["vaso_stop"]))
    candidate_itemid = episode_rows.groupby(["ICUSTAY_ID", "vasopressor_name", "vaso_first"], as_index=False).agg(
        SUBJECT_ID=("SUBJECT_ID", "first"),
        HADM_ID=("HADM_ID", "first"),
        source_itemids=("source_itemids", lambda s: ",".join(sorted(set(",".join(v for v in s if v).split(",")) - {""}))),
        source_rates=("source_rates", lambda s: ";".join(str(v) for v in s if v)),
        source_stopped_values=("source_stopped_values", lambda s: ";".join(str(v) for v in s if v)),
        vaso_start=("_start_candidate", "min"),
        vaso_end=("_end_candidate", "min"),
        min_charttime=("CHARTTIME", "min"),
        max_rate=("vaso_rate", "max"),
        source_row_count=("source_row_count", "sum"),
        source_order_id=("source_order_ids", lambda s: ",".join(sorted(set(",".join(v for v in s if v).split(",")) - {""}))),
    )
    missing_bounds = candidate_itemid["vaso_start"].isna() | candidate_itemid["vaso_end"].isna()
    start_equals_end = candidate_itemid["min_charttime"].eq(candidate_itemid["vaso_end"])
    nonpositive_rate = candidate_itemid["max_rate"].le(0) | candidate_itemid["max_rate"].isna()
    keep = ~missing_bounds & ~start_equals_end & ~nonpositive_rate
    dropped_positive = candidate_itemid.loc[~keep & candidate_itemid["max_rate"].gt(0)].copy()
    if not dropped_positive.empty:
        dropped_uncertain = pd.DataFrame({
            "ROW_ID": pd.NA,
            "SUBJECT_ID": dropped_positive["SUBJECT_ID"],
            "HADM_ID": dropped_positive["HADM_ID"],
            "ICUSTAY_ID": dropped_positive["ICUSTAY_ID"],
            "CHARTTIME": dropped_positive["min_charttime"],
            "ITEMID": dropped_positive["source_itemids"],
            "vasopressor_name": dropped_positive["vasopressor_name"],
            "RATE": dropped_positive["max_rate"],
            "AMOUNT": pd.NA,
            "ORDERID": dropped_positive["source_order_id"],
            "LINKORDERID": pd.NA,
            "STOPPED": pd.NA,
            "uncertainty_reason": np.where(start_equals_end.loc[dropped_positive.index], "carevue_positive_rate_start_equals_end", "carevue_positive_rate_unresolved_duration"),
            "evidence_start": dropped_positive["vaso_start"].fillna(dropped_positive["min_charttime"]),
            "evidence_end": dropped_positive["vaso_end"],
            "evidence_time": dropped_positive["min_charttime"],
        })
        if uncertain_rows.empty:
            uncertain_rows = dropped_uncertain.copy()
        else:
            uncertain_rows = pd.DataFrame([*uncertain_rows.to_dict(orient="records"), *dropped_uncertain.to_dict(orient="records")])
    itemid = candidate_itemid.loc[keep].copy()
    itemid["vasopressor_source"] = "carevue"
    itemid["ITEMID"] = itemid["source_itemids"]
    itemid["episode_uncertain_end"] = False
    itemid["duration_minutes"] = (itemid["vaso_end"] - itemid["vaso_start"]).dt.total_seconds() / 60.0
    itemid = itemid[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "source_itemids", "vasopressor_name", "vasopressor_source", "vaso_start", "vaso_end", "source_order_id", "source_row_count", "episode_uncertain_end", "duration_minutes"]].sort_values(["ICUSTAY_ID", "vaso_start", "vaso_end"]).reset_index(drop=True)
    merged = merge_any_vasopressor_episodes(itemid)
    qc = {
        "carevue_candidate_episode_count_before_final_filter": int(len(candidate_itemid)),
        "carevue_episodes_dropped_missing_bounds": int(missing_bounds.sum()),
        "carevue_episodes_dropped_start_equals_end": int((~missing_bounds & start_equals_end).sum()),
        "carevue_episodes_dropped_nonpositive_rate": int((~missing_bounds & nonpositive_rate).sum()),
        "carevue_positive_rate_rows_in_dropped_episodes": int(dropped_positive["source_row_count"].sum()) if not dropped_positive.empty else 0,
        "carevue_stays_affected_by_dropped_positive_rate_episodes": int(dropped_positive["ICUSTAY_ID"].nunique()) if not dropped_positive.empty else 0,
        "carevue_dropped_positive_rate_episode_examples": dropped_positive.head(10).to_dict(orient="records"),
        "carevue_same_drug_timestamp_stop_active_conflicts": int(len(stop_active_conflicts)),
        "carevue_same_drug_timestamp_zero_positive_conflicts": int(len(zero_positive_conflicts)),
        "carevue_same_drug_timestamp_stop_active_conflict_examples": stop_active_conflicts[["ICUSTAY_ID", "CHARTTIME", "vasopressor_name", "source_itemids", "source_rates", "source_stopped_values", "any_stopped", "any_active_positive", "any_positive_rate", "any_zero_rate", "vaso_rate", "vaso_stopped"]].head(10).to_dict(orient="records"),
        "carevue_same_drug_timestamp_zero_positive_conflict_examples": zero_positive_conflicts[["ICUSTAY_ID", "CHARTTIME", "vasopressor_name", "source_itemids", "source_rates", "source_stopped_values", "any_stopped", "any_active_positive", "any_positive_rate", "any_zero_rate", "vaso_rate", "vaso_stopped"]].head(10).to_dict(orient="records"),
        "carevue_rows_linked_into_episodes": int(itemid["source_row_count"].sum()) if not itemid.empty else 0,
        "carevue_reconstructed_itemid_episodes": int(len(itemid)),
        "carevue_reconstructed_drug_episodes": int(len(itemid)),
        "carevue_reconstructed_any_vasopressor_episodes": int(len(merged)),
        "carevue_episodes_with_uncertain_end_times": int(itemid["episode_uncertain_end"].sum()) if not itemid.empty else 0,
        "carevue_episode_duration_minutes": describe_series(itemid["duration_minutes"]) if not itemid.empty else {},
    }
    return itemid, merged, qc, uncertain_rows


def merge_any_vasopressor_episodes(intervals: pd.DataFrame) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "vasopressor_source", "vasopressor_name", "vaso_start", "vaso_end", "source_row_count", "duration_minutes"])
    rows = []
    for icustay_id, group in intervals.sort_values(["ICUSTAY_ID", "vaso_start", "vaso_end"]).groupby("ICUSTAY_ID", sort=False):
        current = None
        for row in group.itertuples(index=False):
            row_dict = {
                "SUBJECT_ID": int(row.SUBJECT_ID),
                "HADM_ID": int(row.HADM_ID) if pd.notna(row.HADM_ID) else pd.NA,
                "ICUSTAY_ID": int(icustay_id),
                "vasopressor_source": row.vasopressor_source,
                "vasopressor_name": str(row.vasopressor_name),
                "vaso_start": row.vaso_start,
                "vaso_end": row.vaso_end,
                "source_row_count": int(row.source_row_count),
            }
            if current is None:
                current = row_dict
            elif row.vaso_start <= current["vaso_end"]:
                current["vaso_end"] = max(current["vaso_end"], row.vaso_end)
                current["vasopressor_source"] = ",".join(sorted(set(str(current["vasopressor_source"]).split(",") + [row.vasopressor_source])))
                current["vasopressor_name"] = ",".join(sorted(set(str(current["vasopressor_name"]).split(",") + str(row.vasopressor_name).split(","))))
                current["source_row_count"] += int(row.source_row_count)
            else:
                rows.append(current)
                current = row_dict
        if current is not None:
            rows.append(current)
    out = pd.DataFrame(rows)
    out["duration_minutes"] = (out["vaso_end"] - out["vaso_start"]).dt.total_seconds() / 60.0
    return out.reset_index(drop=True)


def load_vasopressor_intervals(mimic_dir: Path, icustays: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    qc: dict[str, object] = {"d_items_verification": validate_itemid_mappings(mimic_dir)}
    mv, uncertain_mv, mv_qc = load_mv_vasopressor_intervals(mimic_dir)
    cv_rows, cv_raw_qc = load_cv_vasopressor_rows(mimic_dir)
    cv_itemid, cv_merged, cv_episode_qc, uncertain_cv = reconstruct_cv_vasopressor_intervals(cv_rows)
    intervals = pd.concat([mv, cv_itemid], ignore_index=True)
    if not intervals.empty:
        intervals["HADM_ID"] = intervals["HADM_ID"].astype("Int64")
        intervals["ICUSTAY_ID"] = intervals["ICUSTAY_ID"].astype("Int64")
        intervals = intervals.sort_values(["ICUSTAY_ID", "vaso_start", "vaso_end", "vasopressor_source"]).reset_index(drop=True)
    continuous = pd.concat([merge_any_vasopressor_episodes(mv), cv_merged], ignore_index=True)
    if not continuous.empty:
        continuous = merge_any_vasopressor_episodes(continuous)
    uncertain = uncertain_cv.copy()
    for col in ["ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID", "STARTTIME", "ENDTIME", "STATUSDESCRIPTION", "evidence_start", "evidence_end", "evidence_time", "uncertainty_reason"]:
        if col not in uncertain.columns:
            uncertain[col] = pd.NA if col not in {"CHARTTIME", "STARTTIME", "ENDTIME", "evidence_start", "evidence_end", "evidence_time"} else pd.NaT
    uncertain["vasopressor_source"] = "carevue"
    if "vasopressor_name" not in uncertain.columns:
        uncertain["vasopressor_name"] = pd.NA
    missing_drug = uncertain["vasopressor_name"].isna()
    uncertain.loc[missing_drug, "vasopressor_name"] = uncertain.loc[missing_drug, "ITEMID"].map(CV_ITEMID_TO_DRUG)
    missing_reason = uncertain["uncertainty_reason"].isna()
    uncertain.loc[missing_reason, "uncertainty_reason"] = "missing_icustay_or_charttime_prevents_carevue_episode_reconstruction"
    uncertain.loc[uncertain["evidence_time"].isna(), "evidence_time"] = uncertain["CHARTTIME"]
    uncertain = uncertain[["ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "STARTTIME", "ENDTIME", "ITEMID", "vasopressor_name", "STATUSDESCRIPTION", "vasopressor_source", "uncertainty_reason", "evidence_start", "evidence_end", "evidence_time"]]
    if uncertain.empty:
        uncertain = uncertain_mv.copy()
    elif not uncertain_mv.empty:
        uncertain = pd.DataFrame([*uncertain.to_dict(orient="records"), *uncertain_mv.to_dict(orient="records")])
    qc.update(mv_qc)
    qc.update(cv_raw_qc)
    qc.update(cv_episode_qc)
    qc["raw_vasopressor_rows"] = {"metavision": int(qc["metavision_raw_rows"]), "carevue": int(qc["carevue_raw_rows"])}
    qc["vasopressor_intervals_by_icustay_availability"] = {str(k): int(v) for k, v in intervals["ICUSTAY_ID"].notna().value_counts(dropna=False).items()} if not intervals.empty else {}
    qc["uncertain_carevue_rows"] = int((uncertain["vasopressor_source"] == "carevue").sum())
    qc["uncertain_metavision_rows"] = int((uncertain["vasopressor_source"] == "metavision").sum())
    qc["uncertain_vasopressor_episode_rows"] = int(len(uncertain))
    qc["uncertain_vasopressor_rows_by_source"] = {str(k): int(v) for k, v in uncertain["vasopressor_source"].value_counts(dropna=False).items()}
    qc["uncertain_vasopressor_rows_by_reason"] = {str(k): int(v) for k, v in uncertain["uncertainty_reason"].value_counts(dropna=False).items()}
    icu_times = icustays[["ICUSTAY_ID", "INTIME", "OUTTIME"]].drop_duplicates("ICUSTAY_ID")
    icu_hadm = icustays[["ICUSTAY_ID", "HADM_ID"]].drop_duplicates("ICUSTAY_ID").rename(columns={"HADM_ID": "ICUSTAYS_HADM_ID"})
    uncertain_with_icu = uncertain.merge(icu_hadm, on="ICUSTAY_ID", how="left") if not uncertain.empty and "ICUSTAY_ID" in uncertain else uncertain
    if not uncertain_with_icu.empty:
        inconsistent = uncertain_with_icu["ICUSTAY_ID"].notna() & uncertain_with_icu["HADM_ID"].notna() & uncertain_with_icu["ICUSTAYS_HADM_ID"].notna() & uncertain_with_icu["HADM_ID"].ne(uncertain_with_icu["ICUSTAYS_HADM_ID"])
        qc["uncertain_rows_with_inconsistent_icustay_hadm"] = int(inconsistent.sum())
    else:
        qc["uncertain_rows_with_inconsistent_icustay_hadm"] = 0
    with_icu = intervals.merge(icu_times, on="ICUSTAY_ID", how="left") if not intervals.empty else intervals
    if not with_icu.empty:
        qc["valid_vasopressor_intervals_missing_icustay_timing"] = int((with_icu["INTIME"].isna() | with_icu["OUTTIME"].isna()).sum())
        qc["valid_vasopressor_intervals_outside_recorded_icu_interval"] = int((with_icu["INTIME"].notna() & with_icu["OUTTIME"].notna() & ((with_icu["vaso_start"] < with_icu["INTIME"]) | (with_icu["vaso_end"] > with_icu["OUTTIME"]))).sum())
    return intervals, continuous, uncertain, qc


def load_icustays(mimic_dir: Path) -> pd.DataFrame:
    return pd.read_csv(mimic_dir / "ICUSTAYS.csv.gz", usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "DBSOURCE", "INTIME", "OUTTIME"], dtype={"SUBJECT_ID": int, "HADM_ID": int, "ICUSTAY_ID": int, "DBSOURCE": str}, parse_dates=["INTIME", "OUTTIME"])


def overlap_seconds(start_a: pd.Timestamp, end_a: pd.Timestamp, start_b: pd.Timestamp, end_b: pd.Timestamp) -> float:
    return max(0.0, (min(end_a, end_b) - max(start_a, start_b)).total_seconds())


def fully_contained_with_tolerance(segment_start: pd.Timestamp, segment_end: pd.Timestamp, icu_start: pd.Timestamp, icu_end: pd.Timestamp) -> bool:
    tolerance = pd.Timedelta(seconds=ICU_CONTAINMENT_TOLERANCE_SECONDS)
    return bool(segment_start >= icu_start - tolerance and segment_end <= icu_end + tolerance)


def match_icu_stays(segments: pd.DataFrame, icustays: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    by_subject = {subject_id: group.copy() for subject_id, group in icustays.groupby("SUBJECT_ID", sort=False)}
    rows = []
    counts: Counter[str] = Counter()
    for row in segments.itertuples(index=False):
        out = row._asdict()
        if row.timestamp_status != "valid" or pd.isna(row.segment_start_time) or pd.isna(row.segment_end_time) or not row.segment_duration_seconds or row.segment_duration_seconds <= 0:
            status = "metadata_wfdb_timestamp_mismatch" if row.timestamp_status == "metadata_wfdb_timestamp_mismatch" else "unknown_segment_time"
            out.update({"hadm_id": pd.NA, "icustay_id": pd.NA, "icu_match_status": status, "icu_intime": pd.NaT, "icu_outtime": pd.NaT, "icu_dbsource": pd.NA, "icu_overlap_seconds": np.nan, "icu_overlap_fraction": np.nan})
            counts[status] += 1
            rows.append(out)
            continue
        candidates = by_subject.get(row.subject_id)
        if candidates is None:
            out.update({"hadm_id": pd.NA, "icustay_id": pd.NA, "icu_match_status": "unmatched", "icu_intime": pd.NaT, "icu_outtime": pd.NaT, "icu_dbsource": pd.NA, "icu_overlap_seconds": 0.0, "icu_overlap_fraction": 0.0})
            counts["unmatched"] += 1
            rows.append(out)
            continue
        overlaps = candidates[(candidates["INTIME"] < row.segment_end_time) & (candidates["OUTTIME"] > row.segment_start_time)]
        if len(overlaps) == 1:
            match = overlaps.iloc[0]
            seconds = overlap_seconds(row.segment_start_time, row.segment_end_time, match["INTIME"], match["OUTTIME"])
            frac = seconds / float(row.segment_duration_seconds)
            status = "matched_full" if fully_contained_with_tolerance(row.segment_start_time, row.segment_end_time, match["INTIME"], match["OUTTIME"]) else "matched_partial"
            out.update({"hadm_id": int(match["HADM_ID"]), "icustay_id": int(match["ICUSTAY_ID"]), "icu_match_status": status, "icu_intime": match["INTIME"], "icu_outtime": match["OUTTIME"], "icu_dbsource": match["DBSOURCE"], "icu_overlap_seconds": seconds, "icu_overlap_fraction": frac})
            counts[status] += 1
        elif len(overlaps) == 0:
            out.update({"hadm_id": pd.NA, "icustay_id": pd.NA, "icu_match_status": "unmatched", "icu_intime": pd.NaT, "icu_outtime": pd.NaT, "icu_dbsource": pd.NA, "icu_overlap_seconds": 0.0, "icu_overlap_fraction": 0.0})
            counts["unmatched"] += 1
        else:
            out.update({"hadm_id": pd.NA, "icustay_id": pd.NA, "icu_match_status": "ambiguous_multiple_icu_stays", "icu_intime": pd.NaT, "icu_outtime": pd.NaT, "icu_dbsource": pd.NA, "icu_overlap_seconds": np.nan, "icu_overlap_fraction": np.nan})
            counts["ambiguous_multiple_icu_stays"] += 1
        rows.append(out)
    return pd.DataFrame(rows), dict(counts)


def uncertain_event_overlap(row: object, uncertain_vaso: pd.DataFrame) -> tuple[list[str], list[str]]:
    if uncertain_vaso.empty:
        return [], []
    candidates = uncertain_vaso[uncertain_vaso["SUBJECT_ID"].eq(row.subject_id)]
    relevant = pd.Series(False, index=candidates.index)
    if pd.notna(row.icustay_id) and "ICUSTAY_ID" in candidates.columns:
        relevant = relevant | candidates["ICUSTAY_ID"].eq(row.icustay_id)
    if pd.notna(row.hadm_id) and {"ICUSTAY_ID", "HADM_ID"}.issubset(candidates.columns):
        relevant = relevant | (candidates["ICUSTAY_ID"].isna() & candidates["HADM_ID"].eq(row.hadm_id))
    if {"ICUSTAY_ID", "HADM_ID"}.issubset(candidates.columns):
        relevant = relevant | (candidates["ICUSTAY_ID"].isna() & candidates["HADM_ID"].isna())
    candidates = candidates[relevant]
    if candidates.empty:
        return [], []
    overlap = pd.Series(False, index=candidates.index)
    overlap_reason = pd.Series("", index=candidates.index, dtype=object)
    if {"evidence_start", "evidence_end"}.issubset(candidates.columns):
        interval_like = candidates["evidence_start"].notna() & candidates["evidence_end"].notna()
        interval_overlap = interval_like & (candidates["evidence_start"] < row.segment_end_time) & (candidates["evidence_end"] > row.segment_start_time)
        overlap = overlap | interval_overlap
        overlap_reason = overlap_reason.mask(interval_overlap, "uncertain_interval_overlap")
        start_only = candidates["evidence_start"].notna() & candidates["evidence_end"].isna()
        start_only_overlap = start_only & (candidates["evidence_start"] < row.segment_end_time)
        overlap = overlap | start_only_overlap
        overlap_reason = overlap_reason.mask(start_only_overlap, "uncertain_start_only")
        end_only = candidates["evidence_start"].isna() & candidates["evidence_end"].notna()
        end_only_overlap = end_only & (candidates["evidence_end"] > row.segment_start_time)
        overlap = overlap | end_only_overlap
        overlap_reason = overlap_reason.mask(end_only_overlap, "uncertain_end_only")
    time_col = "evidence_time" if "evidence_time" in candidates.columns else "CHARTTIME"
    if time_col in candidates.columns:
        point_like = candidates[time_col].notna()
        point_overlap = point_like & (candidates[time_col] >= row.segment_start_time) & (candidates[time_col] < row.segment_end_time)
        overlap = overlap | point_overlap
        overlap_reason = overlap_reason.mask(point_overlap, "uncertain_point_in_segment")
    no_known_time = pd.Series(True, index=candidates.index)
    for col in ["evidence_start", "evidence_end", "evidence_time", "CHARTTIME", "STARTTIME", "ENDTIME"]:
        if col in candidates.columns:
            no_known_time = no_known_time & candidates[col].isna()
    no_time_overlap = pd.Series(False, index=candidates.index)
    if "ICUSTAY_ID" in candidates.columns and pd.notna(row.icustay_id):
        same_icu = candidates["ICUSTAY_ID"].eq(row.icustay_id)
        no_time_overlap = no_time_overlap | same_icu
        overlap_reason = overlap_reason.mask(no_known_time & same_icu, "untimed_same_icustay")
    if {"ICUSTAY_ID", "HADM_ID"}.issubset(candidates.columns) and pd.notna(row.hadm_id):
        same_hadm = candidates["ICUSTAY_ID"].isna() & candidates["HADM_ID"].eq(row.hadm_id)
        no_time_overlap = no_time_overlap | same_hadm
        overlap_reason = overlap_reason.mask(no_known_time & same_hadm, "untimed_same_hadm")
    if {"ICUSTAY_ID", "HADM_ID"}.issubset(candidates.columns):
        patient_only = candidates["ICUSTAY_ID"].isna() & candidates["HADM_ID"].isna()
        no_time_overlap = no_time_overlap | patient_only
        overlap_reason = overlap_reason.mask(no_known_time & patient_only, "untimed_patient_only")
    overlap = overlap | (no_known_time & no_time_overlap)
    matched = candidates.loc[overlap].copy()
    sources = sorted(matched["vasopressor_source"].dropna().astype(str).unique().tolist())
    reasons = sorted(set(overlap_reason.loc[overlap].replace("", pd.NA).dropna().astype(str).tolist()))
    return sources, reasons


def classify_segments(segments: pd.DataFrame, vaso: pd.DataFrame, uncertain_vaso: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    vaso_by_icu = {int(k): g.copy() for k, g in vaso.groupby("ICUSTAY_ID", sort=False)} if not vaso.empty else {}
    rows = []
    detail_rows = []
    for row in segments.itertuples(index=False):
        out = row._asdict()
        if row.icu_match_status == "matched_partial":
            out.update({"has_vasopressor_overlap": pd.NA, "vasopressor_free": pd.NA, "overlapping_vasopressor_count": pd.NA, "first_overlapping_vasopressor_time": pd.NaT, "last_overlapping_vasopressor_time": pd.NaT, "first_overlapping_vasopressor_name": pd.NA, "vasopressor_source": pd.NA, "vasopressor_match_method": "none", "vasopressor_uncertainty_reason": pd.NA, "classification_status": "unknown_partial_icu_coverage"})
            rows.append(out)
            continue
        if row.icu_match_status != "matched_full":
            status = "unknown_timestamp_mismatch" if row.icu_match_status == "metadata_wfdb_timestamp_mismatch" else f"unknown_{row.icu_match_status}"
            out.update({"has_vasopressor_overlap": pd.NA, "vasopressor_free": pd.NA, "overlapping_vasopressor_count": pd.NA, "first_overlapping_vasopressor_time": pd.NaT, "last_overlapping_vasopressor_time": pd.NaT, "first_overlapping_vasopressor_name": pd.NA, "vasopressor_source": pd.NA, "vasopressor_match_method": "none", "vasopressor_uncertainty_reason": pd.NA, "classification_status": status})
            rows.append(out)
            continue
        candidates = vaso_by_icu.get(int(row.icustay_id), pd.DataFrame())
        overlaps = candidates[(candidates["vaso_start"] < row.segment_end_time) & (candidates["vaso_end"] > row.segment_start_time)].copy() if not candidates.empty else pd.DataFrame()
        if not overlaps.empty:
            overlaps = overlaps.sort_values(["vaso_start", "vaso_end"])
            out.update({"has_vasopressor_overlap": True, "vasopressor_free": False, "overlapping_vasopressor_count": int(len(overlaps)), "first_overlapping_vasopressor_time": overlaps["vaso_start"].iloc[0], "last_overlapping_vasopressor_time": overlaps["vaso_end"].max(), "first_overlapping_vasopressor_name": overlaps["vasopressor_name"].iloc[0], "vasopressor_source": ",".join(sorted(overlaps["vasopressor_source"].dropna().unique())), "vasopressor_match_method": "icustay_id", "vasopressor_uncertainty_reason": pd.NA, "classification_status": "classified_overlap"})
            for overlap in overlaps.itertuples(index=False):
                detail_rows.append({"segment_id": row.segment_id, "segment_name": row.segment_name, "subject_id": row.subject_id, "segment_start_time": row.segment_start_time, "segment_end_time": row.segment_end_time, "icustay_id": row.icustay_id, "vasopressor_match_method": "icustay_id", "vasopressor_source": overlap.vasopressor_source, "itemid": str(overlap.ITEMID), "vasopressor_name": overlap.vasopressor_name, "vaso_start": overlap.vaso_start, "vaso_end": overlap.vaso_end})
        else:
            uncertain_sources, uncertain_reasons = uncertain_event_overlap(row, uncertain_vaso)
            if uncertain_sources:
                out.update({"has_vasopressor_overlap": pd.NA, "vasopressor_free": pd.NA, "overlapping_vasopressor_count": pd.NA, "first_overlapping_vasopressor_time": pd.NaT, "last_overlapping_vasopressor_time": pd.NaT, "first_overlapping_vasopressor_name": pd.NA, "vasopressor_source": ",".join(uncertain_sources), "vasopressor_match_method": "unresolved_event_time", "vasopressor_uncertainty_reason": ",".join(uncertain_reasons), "classification_status": "unknown_unresolved_vasopressor_episode_timing"})
            else:
                out.update({"has_vasopressor_overlap": False, "vasopressor_free": True, "overlapping_vasopressor_count": 0, "first_overlapping_vasopressor_time": pd.NaT, "last_overlapping_vasopressor_time": pd.NaT, "first_overlapping_vasopressor_name": pd.NA, "vasopressor_source": pd.NA, "vasopressor_match_method": "none", "vasopressor_uncertainty_reason": pd.NA, "classification_status": "classified_no_overlap"})
        rows.append(out)
    details = pd.DataFrame(detail_rows)
    manifest = pd.DataFrame(rows)
    qc = {
        "vasopressor_match_method_counts": {str(k): int(v) for k, v in manifest["vasopressor_match_method"].value_counts(dropna=False).items()},
        "classification_status_counts": {str(k): int(v) for k, v in manifest["classification_status"].value_counts(dropna=False).items()},
        "fallback_match_count": int(manifest["vasopressor_match_method"].isin(["hadm_time_fallback", "subject_time_fallback"]).sum()),
        "segments_unknown_due_to_partial_icu_coverage": int(manifest["classification_status"].eq("unknown_partial_icu_coverage").sum()),
        "segments_unknown_due_to_uncertain_carevue": int((manifest["classification_status"].eq("unknown_unresolved_vasopressor_episode_timing") & manifest["vasopressor_source"].fillna("").astype(str).str.contains("carevue")).sum()),
        "segments_unknown_due_to_uncertain_metavision": int((manifest["classification_status"].eq("unknown_unresolved_vasopressor_episode_timing") & manifest["vasopressor_source"].fillna("").astype(str).str.contains("metavision")).sum()),
        "segments_unknown_due_to_untimed_same_hadm_evidence": int((manifest["classification_status"].eq("unknown_unresolved_vasopressor_episode_timing") & manifest["vasopressor_uncertainty_reason"].fillna("").astype(str).str.contains("untimed_same_hadm")).sum()),
        "segments_unknown_due_to_timestamp_mismatch": int(manifest["classification_status"].eq("unknown_timestamp_mismatch").sum()),
    }
    return manifest, qc, details


def describe_series(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    return {name: float(value) for name, value in {"min": values.min(), "p01": values.quantile(0.01), "p05": values.quantile(0.05), "median": values.median(), "p95": values.quantile(0.95), "p99": values.quantile(0.99), "max": values.max()}.items()}


def qc_examples(segments: pd.DataFrame, overlap_details: pd.DataFrame, n: int) -> list[dict[str, object]]:
    examples = []
    for row in segments.head(n).itertuples(index=False):
        details = overlap_details[overlap_details["segment_id"].eq(row.segment_id)].head(5) if not overlap_details.empty else pd.DataFrame()
        examples.append({
            "segment_id": row.segment_id,
            "segment_interval": [str(row.segment_start_time), str(row.segment_end_time)],
            "icu_interval": [str(row.icu_intime), str(row.icu_outtime)],
            "icu_overlap_fraction": None if pd.isna(row.icu_overlap_fraction) else float(row.icu_overlap_fraction),
            "overlapping_vasopressors": details[["vasopressor_name", "vasopressor_source", "vaso_start", "vaso_end", "vasopressor_match_method"]].astype(str).to_dict(orient="records") if not details.empty else [],
            "vasopressor_free": None if pd.isna(row.vasopressor_free) else bool(row.vasopressor_free),
            "classification_status": row.classification_status,
            "vasopressor_uncertainty_reason": getattr(row, "vasopressor_uncertainty_reason", None),
        })
    return examples


def build_qc(manifest: pd.DataFrame, vaso: pd.DataFrame, continuous_vaso: pd.DataFrame, overlap_details: pd.DataFrame, extra: dict[str, object]) -> dict[str, object]:
    classified = manifest["vasopressor_free"].notna()
    free = manifest["vasopressor_free"].eq(True)
    overlap = manifest["has_vasopressor_overlap"].eq(True)
    matched = manifest["icu_match_status"].astype(str).str.startswith("matched_")
    qc = {
        "total_waveform_segments": int(len(manifest)),
        "segments_with_valid_timestamps": int(manifest["timestamp_status"].eq("valid").sum()),
        "segments_matched_to_icu_stays": int(matched.sum()),
        "matched_full_segments": int(manifest["icu_match_status"].eq("matched_full").sum()),
        "matched_partial_segments": int(manifest["icu_match_status"].eq("matched_partial").sum()),
        "icu_match_status_counts": {str(k): int(v) for k, v in manifest["icu_match_status"].value_counts(dropna=False).items()},
        "timestamp_status_counts": {str(k): int(v) for k, v in manifest["timestamp_status"].value_counts(dropna=False).items()},
        "segments_with_metadata_wfdb_timestamp_match": int(manifest["metadata_wfdb_timestamp_status"].eq("matched").sum()) if "metadata_wfdb_timestamp_status" in manifest else 0,
        "segments_with_metadata_wfdb_timestamp_mismatch": int(manifest["metadata_wfdb_timestamp_status"].eq("mismatch").sum()) if "metadata_wfdb_timestamp_status" in manifest else 0,
        "icu_overlap_fraction_distribution": describe_series(manifest.loc[matched, "icu_overlap_fraction"]),
        "segments_with_vasopressor_overlap": int(overlap.sum()),
        "segments_classified_vasopressor_free": int(free.sum()),
        "segments_with_unknown_classification": int((~classified).sum()),
        "percentage_vasopressor_free_among_classified": float(free.sum() / classified.sum() * 100.0) if classified.any() else None,
        "vasopressor_exposures_by_source": {str(k): int(v) for k, v in vaso["vasopressor_source"].value_counts().items()} if not vaso.empty else {},
        "vasopressor_exposures_by_drug": {str(k): int(v) for k, v in vaso["vasopressor_name"].value_counts().items()} if not vaso.empty else {},
        "continuous_vasopressor_interval_count": int(len(continuous_vaso)),
        "overlapping_vasopressor_exposures_by_source": {str(k): int(v) for k, v in overlap_details["vasopressor_source"].value_counts().items()} if not overlap_details.empty else {},
        "overlapping_vasopressor_exposures_by_drug": {str(k): int(v) for k, v in overlap_details["vasopressor_name"].value_counts().items()} if not overlap_details.empty else {},
        "examples_overlapping_segments": qc_examples(manifest[overlap], overlap_details, 5),
        "examples_non_overlapping_segments": qc_examples(manifest[free], overlap_details, 5),
        "examples_unknown_segments": qc_examples(manifest[manifest["vasopressor_free"].isna()], overlap_details, 5),
        "examples_timestamp_mismatches": manifest.loc[manifest.get("metadata_wfdb_timestamp_status", pd.Series(index=manifest.index, dtype=object)).eq("mismatch"), ["segment_id", "segment_start_time", "wfdb_segment_start_time", "metadata_wfdb_timestamp_delta_seconds", "metadata_wfdb_timestamp_abs_delta_seconds", "timestamp_status"]].head(10).astype(str).to_dict(orient="records") if "metadata_wfdb_timestamp_status" in manifest else [],
        "examples_partial_icu_matches": manifest.loc[manifest["icu_match_status"].eq("matched_partial"), ["segment_id", "segment_start_time", "segment_end_time", "icu_intime", "icu_outtime", "icu_overlap_fraction", "classification_status"]].head(10).astype(str).to_dict(orient="records"),
        "examples_untimed_same_hadm_uncertainty": manifest.loc[manifest["vasopressor_uncertainty_reason"].fillna("").astype(str).str.contains("untimed_same_hadm"), ["segment_id", "subject_id", "hadm_id", "icustay_id", "classification_status", "vasopressor_source", "vasopressor_uncertainty_reason"]].head(10).astype(str).to_dict(orient="records") if "vasopressor_uncertainty_reason" in manifest else [],
    }
    qc.update(extra)
    return qc


def add_backward_compatibility_aliases(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    out["record_name"] = out["segment_name"]
    out["record_path"] = out["segment_path"]
    out["record_start_time"] = out["segment_start_time"]
    out["record_end_time"] = out["segment_end_time"]
    out["record_duration_seconds"] = out["segment_duration_seconds"]
    return out


def write_outputs(
    manifest: pd.DataFrame,
    qc: dict[str, object],
    continuous_vaso: pd.DataFrame,
    uncertain_vaso: pd.DataFrame,
    output: Path,
    qc_output: Path,
    free_segments_output: Path,
    intervals_output: Path,
    uncertain_output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    add_backward_compatibility_aliases(manifest).to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    qc_output.parent.mkdir(parents=True, exist_ok=True)
    qc_output.write_text(json.dumps(qc, indent=2, default=str))
    free_segments = manifest.loc[manifest["vasopressor_free"].eq(True), "segment_id"].astype(str)
    free_segments_output.parent.mkdir(parents=True, exist_ok=True)
    free_segments_output.write_text("\n".join(free_segments.tolist()) + ("\n" if len(free_segments) else ""))
    intervals_output.parent.mkdir(parents=True, exist_ok=True)
    continuous_vaso.to_csv(intervals_output, index=False)
    uncertain_output.parent.mkdir(parents=True, exist_ok=True)
    uncertain_vaso.to_csv(uncertain_output, index=False)


def print_manual_examples(qc: dict[str, object]) -> None:
    print("Manual sanity-check examples:", flush=True)
    for label, key in [("overlapping", "examples_overlapping_segments"), ("non_overlapping", "examples_non_overlapping_segments"), ("unknown", "examples_unknown_segments")]:
        print(f"  {label}:", flush=True)
        for example in qc.get(key, [])[:3]:
            print(
                "    "
                f"segment={example['segment_id']} "
                f"segment_interval={example['segment_interval']} "
                f"icu_interval={example['icu_interval']} "
                f"icu_overlap_fraction={example['icu_overlap_fraction']} "
                f"overlapping_vasopressors={example['overlapping_vasopressors']} "
                f"classification_status={example['classification_status']} "
                f"vasopressor_free={example['vasopressor_free']}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    free_segments_output = args.free_records_output or args.free_segments_output
    if args.segment_metadata_json is not None:
        print(f"Loading waveform segments from segment metadata {args.segment_metadata_json}", flush=True)
        segments, segment_qc = load_waveform_segments_from_segment_metadata(args.segment_metadata_json, args.waveform_root)
        if args.max_patient_dirs is not None:
            keep = sorted(segments["subject_id_str"].unique())[: args.max_patient_dirs]
            segments = segments[segments["subject_id_str"].isin(keep)].reset_index(drop=True)
    else:
        print(f"Loading waveform segments from {args.waveform_root}", flush=True)
        segments, segment_qc = load_waveform_segments(args.waveform_root, args.max_patient_dirs)
    print(f"  waveform segments: {len(segments):,}", flush=True)
    print(f"Loading ICU stays and vasopressor rows from {args.mimic_clinical_dir}", flush=True)
    icustays = load_icustays(args.mimic_clinical_dir)
    vaso, continuous_vaso, uncertain_vaso, vaso_qc = load_vasopressor_intervals(args.mimic_clinical_dir, icustays)
    print(f"  reconstructed vasopressor intervals: {len(vaso):,}", flush=True)
    print("Matching segments to ICU stays by subject and interval overlap", flush=True)
    matched, match_qc = match_icu_stays(segments, icustays)
    print(f"  ICU match status: {match_qc}", flush=True)
    print("Classifying segment-level vasopressor overlap by matched ICUSTAY_ID", flush=True)
    manifest, classify_qc, overlap_details = classify_segments(matched, vaso, uncertain_vaso)
    qc = build_qc(manifest, vaso, continuous_vaso, overlap_details, {**segment_qc, **vaso_qc, **classify_qc})
    qc["inputs"] = {"waveform_root": str(args.waveform_root), "mimic_clinical_dir": str(args.mimic_clinical_dir), "segment_metadata_json": str(args.segment_metadata_json) if args.segment_metadata_json else None}
    qc["vasopressor_definition"] = {
        "metavision_itemids": MV_VASOPRESSOR_ITEMIDS,
        "carevue_itemids": CV_VASOPRESSOR_ITEMIDS,
        "metavision_itemid_to_drug": MV_ITEMID_TO_DRUG,
        "carevue_itemid_to_drug": CV_ITEMID_TO_DRUG,
        "metavision_status_rule": "exclude STATUSDESCRIPTION == 'Rewritten'; retain Changed, FinishedRunning, Stopped, Paused, Flushed after required-field/timestamp checks",
        "carevue_interval_rule": "reconstructed at clinical drug level from RATE/STOPPED charting sequence after ITEMID -> vasopressor_name mapping; no fixed CHARTTIME+duration exposure rule",
        "segment_level_vasopressor_free_definition": "Segment timing is valid, metadata/WFDB timestamps agree when both are available, the segment is fully contained within one unambiguous ICU stay, no confirmed vasopressor interval overlaps it, and no unresolved CareVue or MetaVision evidence prevents confident classification. This is a coarse segment-level prefilter, not final feature-window labeling.",
        "future_window_filtering_note": "Future feature-window filtering must check both confirmed continuous vasopressor intervals and uncertain vasopressor evidence before assigning a window vasopressor_free=True.",
        "overlap_rule": HALF_OPEN_OVERLAP_RULE,
        "metadata_wfdb_timestamp_tolerance_seconds": METADATA_WFDB_TIMESTAMP_TOLERANCE_SECONDS,
        "icu_containment_tolerance_seconds": ICU_CONTAINMENT_TOLERANCE_SECONDS,
    }
    write_outputs(manifest, qc, continuous_vaso, uncertain_vaso, args.output, args.qc_output, free_segments_output, args.vasopressor_intervals_output, args.uncertain_vasopressor_output)
    print_manual_examples(qc)
    print(json.dumps({
        "manifest": str(args.output),
        "qc": str(args.qc_output),
        "free_segments": str(free_segments_output),
        "vasopressor_intervals": str(args.vasopressor_intervals_output),
        "uncertain_vasopressor_evidence": str(args.uncertain_vasopressor_output),
        "total_segments": qc["total_waveform_segments"],
        "matched_segments": qc["segments_matched_to_icu_stays"],
        "overlap_segments": qc["segments_with_vasopressor_overlap"],
        "vasopressor_free_segments": qc["segments_classified_vasopressor_free"],
        "unknown_segments": qc["segments_with_unknown_classification"],
        "percent_free_among_classified": qc["percentage_vasopressor_free_among_classified"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
