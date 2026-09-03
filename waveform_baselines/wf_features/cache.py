from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from waveform_baselines.data_index import AlignedWaveformDataset, build_aligned_20m_anchor_table
from .config import CACHE_ROOT, DEFAULT_EXTRACTION_CONFIG, ExtractionConfig
from .definitions import FEATURE_DEFINITIONS, feature_definition_map, feature_names
from .pipeline import extract_feature_sequence
from .utils import linear_trend

try:
    from waveform_baselines.wf_features_v8.definitions import feature_definition_map as v8_feature_definition_map
except ImportError:  # pragma: no cover - v8 is optional for older cache-only uses
    v8_feature_definition_map = None


@dataclass
class FeatureCache:
    values: np.ndarray
    mask: np.ndarray
    patient_ids: np.ndarray
    anchor_times: np.ndarray
    anchor_ids: np.ndarray
    split_labels: np.ndarray
    feature_names: list[str]
    metadata: dict[str, object]
    cache_dir: Path
    segment_ids: np.ndarray | None = None
    segment_names: np.ndarray | None = None




def _ensure_unique_keys(keys: list[tuple[object, ...]], label: str) -> None:
    seen: set[tuple[object, ...]] = set()
    duplicates: list[tuple[object, ...]] = []
    for key in keys:
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        examples = ", ".join(str(item) for item in duplicates[:5])
        raise ValueError(f"Duplicate {label} keys found: {examples}")


def _count_labels(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels, dtype=object), return_counts=True)
    return {str(label): int(count) for label, count in zip(values.tolist(), counts.tolist())}


def _canonicalize_anchor_table(anchors: pd.DataFrame) -> pd.DataFrame:
    return anchors.sort_values(["anchor_id"], kind="stable").reset_index(drop=True)


def _validate_anchor_table_keys(anchors: pd.DataFrame, label_prefix: str) -> None:
    patient_ids = anchors["patient_id"].astype(str).to_numpy()
    anchor_times = anchors["anchor_time"].to_numpy(dtype=np.float64)
    anchor_ids = anchors["anchor_id"].to_numpy(dtype=np.int64)
    _ensure_unique_keys(
        list(zip(patient_ids.tolist(), anchor_times.tolist())),
        f"{label_prefix} feature (patient_id, anchor_time)",
    )
    _ensure_unique_keys(
        [(int(anchor_id),) for anchor_id in anchor_ids.tolist()],
        f"{label_prefix} feature anchor_id",
    )


def _shard_anchor_table(anchors: pd.DataFrame, shard_index: int, shard_count: int) -> pd.DataFrame:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count})")
    return anchors.iloc[shard_index::shard_count].reset_index(drop=True)


def _success_marker(cache_dir: Path) -> Path:
    return Path(cache_dir) / "_SUCCESS"


def _split_patient_lookup(splits_path: Path) -> dict[str, str]:
    splits = json.loads(Path(splits_path).read_text())
    label_lookup: dict[str, str] = {}
    duplicate_assignments: list[tuple[str, str, str]] = []
    for split_name, split_patients in splits.items():
        if not isinstance(split_patients, list):
            continue
        for pid in split_patients:
            pid_str = str(pid)
            previous = label_lookup.get(pid_str)
            if previous is not None and previous != split_name:
                duplicate_assignments.append((pid_str, previous, split_name))
            label_lookup[pid_str] = split_name
    if duplicate_assignments:
        examples = ", ".join(f"{pid}:{a}/{b}" for pid, a, b in duplicate_assignments[:5])
        raise ValueError(f"Patients assigned to multiple splits in {splits_path}: {examples}")
    return label_lookup

def _split_labels_from_patients(patient_ids: np.ndarray, splits_path: Path) -> np.ndarray:
    label_lookup = _split_patient_lookup(splits_path)
    labels = np.asarray([label_lookup.get(str(pid), "unknown") for pid in patient_ids], dtype=object)
    if np.any(labels == "unknown"):
        unknown = sorted(set(np.asarray(patient_ids, dtype=str)[labels == "unknown"].tolist()))
        examples = ", ".join(unknown[:10])
        raise ValueError(f"Patients missing from split file {splits_path}: {examples}")
    return labels


def _feature_stats(values: np.ndarray, mask: np.ndarray, feature_order: list[str]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(feature_order):
        valid = mask[:, :, idx] & np.isfinite(values[:, :, idx])
        arr = values[:, :, idx][valid].astype(np.float64, copy=False)
        if arr.size == 0:
            report[name] = {
                "count": 0,
                "missing_fraction": 1.0,
                "non_finite_fraction": 1.0,
            }
            continue
        report[name] = {
            "count": int(arr.size),
            "missing_fraction": float(1.0 - np.mean(valid)),
            "non_finite_fraction": float(np.mean(~np.isfinite(values[:, :, idx]))),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "iqr": float(np.percentile(arr, 75.0) - np.percentile(arr, 25.0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p01": float(np.percentile(arr, 1.0)),
            "p05": float(np.percentile(arr, 5.0)),
            "p95": float(np.percentile(arr, 95.0)),
            "p99": float(np.percentile(arr, 99.0)),
            "near_constant_fraction": float(np.mean(np.isclose(arr, np.median(arr), atol=1e-8))),
        }
    return report


class FeatureCacheBuilder:
    def __init__(
        self,
        config: ExtractionConfig = DEFAULT_EXTRACTION_CONFIG,
        cache_root: Path = CACHE_ROOT,
    ) -> None:
        self.config = config
        self.cache_root = Path(cache_root)
        self._waveform_metadata_cache: dict[str, dict[str, object]] = {}
        self._waveform_array_cache: dict[str, np.memmap] = {}

    def build(
        self,
        output_name: str,
        splits_path: Path,
        target_bundle_path: Path | None = None,
        raw_root: Path | None = None,
        icu_output_dir: Path | None = None,
        waveform_dir: Path | None = None,
        max_samples: int | None = None,
        overwrite: bool = False,
        shard_index: int | None = None,
        shard_count: int | None = None,
    ) -> FeatureCache:
        anchor_kwargs = {}
        if raw_root is not None:
            anchor_kwargs["raw_root"] = raw_root
        if icu_output_dir is not None:
            anchor_kwargs["icu_output_dir"] = icu_output_dir
        anchors = build_aligned_20m_anchor_table(**anchor_kwargs)
        split_lookup = _split_patient_lookup(splits_path)
        anchors = anchors.loc[anchors["patient_id"].astype(str).isin(split_lookup)].reset_index(drop=True)
        if anchors.empty:
            raise ValueError(f"No anchors matched patients in split file {splits_path}")
        anchors = _canonicalize_anchor_table(anchors)
        _validate_anchor_table_keys(anchors, "global")
        expected_full_n_samples = int(len(anchors))
        if max_samples is not None:
            anchors = anchors.iloc[:max_samples].copy()
        if (shard_index is None) != (shard_count is None):
            raise ValueError("shard_index and shard_count must be provided together")
        if shard_count is not None:
            anchors = _shard_anchor_table(anchors, int(shard_index), int(shard_count))
            if anchors.empty:
                raise ValueError(f"Shard {shard_index}/{shard_count} has no anchors")
        feature_order = feature_names()
        n_samples = len(anchors)
        patient_ids = anchors["patient_id"].astype(str).to_numpy()
        anchor_times = anchors["anchor_time"].to_numpy(dtype=np.float64)
        anchor_ids = anchors["anchor_id"].to_numpy(dtype=np.int64)
        _ensure_unique_keys(list(zip(patient_ids.tolist(), anchor_times.tolist())), "feature (patient_id, anchor_time)")
        _ensure_unique_keys([(int(anchor_id),) for anchor_id in anchor_ids.tolist()], "feature anchor_id")
        split_labels = _split_labels_from_patients(patient_ids, splits_path)
        split_patient_counts = _count_labels(_split_labels_from_patients(np.unique(patient_ids), splits_path))
        split_sample_counts = _count_labels(split_labels)
        cache_dir = self.cache_root / self.config.feature_version / output_name
        if cache_dir.exists() and any(cache_dir.iterdir()):
            metadata_path = cache_dir / "metadata.json"
            if metadata_path.exists() and shard_count is not None:
                existing = json.loads(metadata_path.read_text())
                if existing.get("shard_index") != shard_index or existing.get("shard_count") != shard_count:
                    raise FileExistsError(
                        f"Cache directory {cache_dir} already exists with incompatible shard metadata"
                    )
            if not overwrite:
                raise FileExistsError(f"Cache directory {cache_dir} already exists and is nonempty; pass overwrite=True/--overwrite to replace it")
        print(
            json.dumps({
                "event": "waveform_feature_cache_build_start",
                "shard_index": shard_index,
                "shard_count": shard_count,
                "output_name": output_name,
                "cache_dir": str(cache_dir),
                "n_samples": int(n_samples),
                "expected_full_n_samples": expected_full_n_samples,
            }, sort_keys=True),
            flush=True,
        )
        values = np.full((n_samples, self.config.n_feature_windows, len(feature_order)), np.nan, dtype=np.float32)
        mask = np.zeros((n_samples, self.config.n_feature_windows, len(feature_order)), dtype=bool)
        extraction_diagnostics: dict[str, int] = {}
        for idx in range(n_samples):
            waveform = self._load_waveform_sample(
                anchors=anchors,
                index=idx,
                waveform_dir=waveform_dir,
            )
            seq_values, seq_mask, _ = extract_feature_sequence(
                waveform, config=self.config, diagnostics=extraction_diagnostics
            )
            values[idx] = seq_values
            mask[idx] = seq_mask
        if not (len(patient_ids) == len(anchor_times) == len(anchor_ids) == len(split_labels) == n_samples):
            raise ValueError("Feature cache metadata arrays do not all have length N")
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "values.npy", values)
        np.save(cache_dir / "mask.npy", mask)
        np.save(cache_dir / "patient_ids.npy", patient_ids)
        np.save(cache_dir / "anchor_times.npy", anchor_times)
        np.save(cache_dir / "anchor_ids.npy", anchor_ids)
        np.save(cache_dir / "split_labels.npy", split_labels)
        metadata = {
            "feature_version": self.config.feature_version,
            "extraction_config": self.config.to_dict(),
            "sampling_rate_hz": self.config.sampling_rate_hz,
            "input_window_seconds": self.config.input_window_seconds,
            "feature_window_seconds": self.config.feature_window_seconds,
            "micro_window_seconds": self.config.micro_window_seconds,
            "n_feature_windows": self.config.n_feature_windows,
            "micro_windows_per_feature_window": self.config.micro_windows_per_feature_window,
            "max_interpolated_gap_seconds": self.config.max_interpolated_gap_seconds,
            "extreme_value_atol_fraction": self.config.extreme_value_atol_fraction,
            "ecg_hrv_min_beats": self.config.ecg_hrv_min_beats,
            "ecg_hrv_min_successive_pairs": self.config.ecg_hrv_min_successive_pairs,
            "ecg_peak_search_radius_s": self.config.ecg_peak_search_radius_s,
            "abp_peak_search_radius_s": self.config.abp_peak_search_radius_s,
            "abp_trough_search_radius_s": self.config.abp_trough_search_radius_s,
            "abp_min_pulse_bpm": self.config.abp_min_pulse_bpm,
            "abp_max_pulse_bpm": self.config.abp_max_pulse_bpm,
            "pleth_peak_search_radius_s": self.config.pleth_peak_search_radius_s,
            "pleth_trough_search_radius_s": self.config.pleth_trough_search_radius_s,
            "channel_order": list(self.config.channel_order),
            "feature_names": feature_order,
            "feature_units": {feature.name: feature.unit for feature in FEATURE_DEFINITIONS},
            "feature_descriptions": {feature.name: feature.description for feature in FEATURE_DEFINITIONS},
            "target_bundle_path": str(target_bundle_path) if target_bundle_path else None,
            "splits_path": str(splits_path),
            "n_samples": n_samples,
            "expected_full_n_samples": expected_full_n_samples,
            "split_patient_counts": split_patient_counts,
            "split_sample_counts": split_sample_counts,
            "extraction_diagnostics": extraction_diagnostics,
            "shard_index": shard_index,
            "shard_count": shard_count,
        }
        (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        quality_report = _feature_stats(values, mask, feature_order)
        (cache_dir / "feature_quality_report.json").write_text(json.dumps(quality_report, indent=2))
        _success_marker(cache_dir).write_text(f"completed_at_unix={time.time():.6f}\n")
        return FeatureCache(
            values=values,
            mask=mask,
            patient_ids=patient_ids,
            anchor_times=anchor_times,
            anchor_ids=anchor_ids,
            split_labels=split_labels,
            feature_names=feature_order,
            metadata=metadata,
            cache_dir=cache_dir,
        )

    def _load_waveform_sample(
        self,
        anchors: pd.DataFrame,
        index: int,
        waveform_dir: Path | None,
    ) -> np.ndarray:
        row = anchors.iloc[index]
        if {"raw_file", "raw_window_index"}.issubset(anchors.columns):
            raw_file = str(row["raw_file"])
            raw = self._raw_array(raw_file)
            channel_indices = self._channel_indices(tuple(row["signal_order"].split(",")))
            return np.asarray(raw[int(row["raw_window_index"]), channel_indices, :], dtype=np.float32)
        resolved_waveform_dir = Path(
            waveform_dir
            if waveform_dir is not None
            else "/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms_ii_pleth_abp_resp"
        )
        metadata = self._waveform_metadata(resolved_waveform_dir)
        patient_id = str(row["patient_id"])
        anchor_time = float(row["anchor_time"])
        patient_meta = metadata["patients"].get(patient_id)
        if patient_meta is None:
            raise KeyError(f"Patient {patient_id} missing from waveform metadata {resolved_waveform_dir}")
        channel_indices = self._channel_indices(tuple(metadata["channels"]))
        fs = int(metadata["fs"])
        if fs != self.config.sampling_rate_hz:
            raise ValueError(f"Waveform metadata fs={fs} does not match extraction fs={self.config.sampling_rate_hz}")
        seg_start = float(patient_meta["seg_start_secs"])
        anchor_center = int(round((anchor_time - seg_start) * fs))
        start = anchor_center - (self.config.input_samples // 2)
        end = start + self.config.input_samples
        arr = self._raw_array(str(resolved_waveform_dir / f"{patient_id}.npy"))
        if start < 0 or end > arr.shape[1]:
            raise IndexError(
                f"Anchor {anchor_time} for patient {patient_id} is outside waveform bounds in {resolved_waveform_dir}"
            )
        return np.asarray(arr[channel_indices, start:end], dtype=np.float32)

    def _waveform_metadata(self, waveform_dir: Path) -> dict[str, object]:
        key = str(waveform_dir.resolve())
        cached = self._waveform_metadata_cache.get(key)
        if cached is None:
            cached = json.loads((waveform_dir / "metadata.json").read_text())
            self._waveform_metadata_cache[key] = cached
        return cached

    def _raw_array(self, path: str) -> np.memmap:
        cached = self._waveform_array_cache.get(path)
        if cached is None:
            cached = np.load(path, mmap_mode="r")
            self._waveform_array_cache[path] = cached
        return cached

    def _channel_indices(self, available_channels: tuple[str, ...]) -> list[int]:
        return [available_channels.index(ch) for ch in self.config.channel_order]


def load_feature_cache(cache_dir: Path, require_success: bool | None = None) -> FeatureCache:
    cache_dir = Path(cache_dir)
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    if require_success is None:
        require_success = bool(metadata.get("is_merged_cache", False))
    if require_success and not _success_marker(cache_dir).exists():
        raise FileNotFoundError(f"Feature cache {cache_dir} is missing completion marker _SUCCESS")
    values = np.load(cache_dir / "values.npy", mmap_mode="r")
    mask = np.load(cache_dir / "mask.npy", mmap_mode="r")
    patient_ids = np.load(cache_dir / "patient_ids.npy", allow_pickle=True)
    anchor_times = np.load(cache_dir / "anchor_times.npy")
    anchor_ids = np.load(cache_dir / "anchor_ids.npy")
    split_labels = np.load(cache_dir / "split_labels.npy", allow_pickle=True)
    segment_ids = np.load(cache_dir / "segment_ids.npy", allow_pickle=True).astype(str) if (cache_dir / "segment_ids.npy").exists() else None
    segment_names = np.load(cache_dir / "segment_names.npy", allow_pickle=True).astype(str) if (cache_dir / "segment_names.npy").exists() else None
    anchors_path = cache_dir / "anchors.csv"
    if (segment_ids is None or segment_names is None) and anchors_path.exists():
        anchors = pd.read_csv(anchors_path)
        if "anchor_id" in anchors.columns:
            anchors_by_id = anchors.set_index("anchor_id", drop=False)
            requested_ids = np.asarray(anchor_ids, dtype=np.int64)
            if set(requested_ids.tolist()).issubset(set(anchors_by_id.index.astype(np.int64).tolist())):
                ordered = anchors_by_id.loc[requested_ids.tolist()]
                if segment_ids is None and "segment_id" in ordered.columns:
                    segment_ids = ordered["segment_id"].astype(str).to_numpy()
                if segment_names is None and "seg_name" in ordered.columns:
                    segment_names = ordered["seg_name"].astype(str).to_numpy()
    feature_order = list(metadata["feature_names"])
    if values.shape != mask.shape:
        raise ValueError(f"Feature cache {cache_dir} has values/mask shape mismatch: {values.shape} vs {mask.shape}")
    if values.ndim != 3:
        raise ValueError(f"Feature cache {cache_dir} values must be 3D, got shape {values.shape}")
    n_samples = values.shape[0]
    if not (len(patient_ids) == len(anchor_times) == len(anchor_ids) == len(split_labels) == n_samples):
        raise ValueError(f"Feature cache {cache_dir} metadata arrays do not all have length N={n_samples}")
    if segment_ids is not None and len(segment_ids) != n_samples:
        raise ValueError(f"Feature cache {cache_dir} segment_ids length does not match N={n_samples}")
    if segment_names is not None and len(segment_names) != n_samples:
        raise ValueError(f"Feature cache {cache_dir} segment_names length does not match N={n_samples}")
    if values.shape[2] != len(feature_order):
        raise ValueError(
            f"Feature cache {cache_dir} feature dimension {values.shape[2]} does not match metadata feature_names {len(feature_order)}"
        )
    if "n_samples" in metadata and int(metadata["n_samples"]) != n_samples:
        raise ValueError(f"Feature cache {cache_dir} metadata n_samples={metadata['n_samples']} does not match values N={n_samples}")
    if "n_feature_windows" in metadata and int(metadata["n_feature_windows"]) != values.shape[1]:
        raise ValueError(
            f"Feature cache {cache_dir} metadata n_feature_windows={metadata['n_feature_windows']} does not match values shape {values.shape}"
        )
    return FeatureCache(
        values=values,
        mask=mask,
        patient_ids=patient_ids,
        anchor_times=anchor_times,
        anchor_ids=anchor_ids,
        split_labels=split_labels,
        feature_names=feature_order,
        metadata=metadata,
        cache_dir=cache_dir,
        segment_ids=segment_ids,
        segment_names=segment_names,
    )


class FeaturePreprocessor:
    def __init__(self, feature_order: list[str]):
        self.feature_order = feature_order
        self.definition_map = feature_definition_map()
        if v8_feature_definition_map is not None:
            self.definition_map.update(v8_feature_definition_map())
        self.impute_values: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.train_valid_counts: dict[str, int] = {}
        self.train_valid_fractions: dict[str, float] = {}

    def fit(self, values: np.ndarray, mask: np.ndarray, split_labels: np.ndarray) -> None:
        train_rows = np.asarray(split_labels) == "train"
        if not train_rows.any():
            raise ValueError("No training rows available for preprocessing fit.")
        train_values = values[train_rows]
        train_mask = mask[train_rows]
        for idx, name in enumerate(self.feature_order):
            valid = train_mask[:, :, idx] & np.isfinite(train_values[:, :, idx])
            arr = train_values[:, :, idx][valid].astype(np.float64, copy=False)
            self.train_valid_counts[name] = int(arr.size)
            self.train_valid_fractions[name] = float(np.mean(valid))
            if arr.size == 0:
                self.impute_values[name] = 0.0
                self.means[name] = 0.0
                self.stds[name] = 1.0
                continue
            median = float(np.median(arr))
            self.impute_values[name] = median
            definition = self.definition_map.get(name)
            normalize = bool(getattr(definition, "normalize", True))
            if normalize:
                mean = float(arr.mean())
                std = float(arr.std())
                self.means[name] = mean
                self.stds[name] = std if std > 1e-8 else 1.0
            else:
                self.means[name] = 0.0
                self.stds[name] = 1.0

    def transform(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.asarray(values, dtype=np.float32).copy()
        valid_mask = mask.astype(bool)
        for idx, name in enumerate(self.feature_order):
            feature = out[:, :, idx]
            feature[~valid_mask[:, :, idx]] = self.impute_values[name]
            feature = (feature - self.means[name]) / self.stds[name]
            out[:, :, idx] = feature
        mask_float = valid_mask.astype(np.float32)
        return np.concatenate([out, mask_float], axis=2)

    def save(self, output_path: Path) -> None:
        payload = {
            "feature_order": self.feature_order,
            "impute_values": self.impute_values,
            "means": self.means,
            "stds": self.stds,
            "train_valid_counts": self.train_valid_counts,
            "train_valid_fractions": self.train_valid_fractions,
        }
        Path(output_path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, output_path: Path) -> "FeaturePreprocessor":
        payload = json.loads(Path(output_path).read_text())
        obj = cls(feature_order=list(payload["feature_order"]))
        obj.impute_values = {str(k): float(v) for k, v in payload["impute_values"].items()}
        obj.means = {str(k): float(v) for k, v in payload["means"].items()}
        obj.stds = {str(k): float(v) for k, v in payload["stds"].items()}
        obj.train_valid_counts = {str(k): int(v) for k, v in payload.get("train_valid_counts", {}).items()}
        obj.train_valid_fractions = {str(k): float(v) for k, v in payload.get("train_valid_fractions", {}).items()}
        return obj


class HistorySummaryBuilder:
    SUMMARY_NAMES = ("mean", "median", "std", "min", "max", "first", "last", "delta", "slope", "valid_fraction")

    def __init__(self, feature_order: list[str]) -> None:
        self.feature_order = feature_order

    def summary_names(self) -> list[str]:
        names = []
        for name in self.feature_order:
            for summary_name in self.SUMMARY_NAMES:
                names.append(f"{name}__{summary_name}")
        return names

    def transform(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n_samples, _, n_features = values.shape
        out = np.full((n_samples, n_features * len(self.SUMMARY_NAMES)), np.nan, dtype=np.float32)
        for sample_idx in range(n_samples):
            row_values = values[sample_idx]
            row_mask = mask[sample_idx]
            col_idx = 0
            for feat_idx in range(n_features):
                valid = row_mask[:, feat_idx] & np.isfinite(row_values[:, feat_idx])
                arr = row_values[:, feat_idx][valid].astype(np.float64, copy=False)
                time_idx = np.flatnonzero(valid).astype(np.float64)
                if arr.size:
                    first = float(arr[0])
                    last = float(arr[-1])
                    summary = [
                        float(arr.mean()),
                        float(np.median(arr)),
                        float(arr.std()),
                        float(arr.min()),
                        float(arr.max()),
                        first,
                        last,
                        last - first,
                        linear_trend(arr, time_idx),
                        float(np.mean(valid)),
                    ]
                else:
                    summary = [np.nan] * 9 + [0.0]
                out[sample_idx, col_idx : col_idx + len(self.SUMMARY_NAMES)] = np.asarray(summary, dtype=np.float32)
                col_idx += len(self.SUMMARY_NAMES)
        return out
