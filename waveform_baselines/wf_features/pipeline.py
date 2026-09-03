from __future__ import annotations

import numpy as np

from .config import DEFAULT_EXTRACTION_CONFIG, ExtractionConfig
from .definitions import feature_names
from .ecg import extract_ecg_features
from .pulsatile import extract_abp_features, extract_pleth_features
from .resp import extract_resp_features
from .utils import agreement_score, minute_window_slices, pad_nan_matrix


def _populate_row(feature_order: list[str], feature_values: dict[str, float], row: np.ndarray, mask: np.ndarray) -> None:
    for idx, name in enumerate(feature_order):
        value = feature_values.get(name, float("nan"))
        row[idx] = value
        mask[idx] = np.isfinite(value)


def _minute_features(
    window: np.ndarray,
    config: ExtractionConfig,
    diagnostics: dict[str, int] | None = None,
) -> dict[str, float]:
    ecg = extract_ecg_features(
        window[0],
        fs=config.sampling_rate_hz,
        micro_window_samples=config.micro_window_samples,
        template_points=config.morphology_template_points,
        hrv_min_beats=config.ecg_hrv_min_beats,
        hrv_min_successive_pairs=config.ecg_hrv_min_successive_pairs,
        detector=config.ecg_detector,
        allow_energy_fallback=config.ecg_allow_energy_fallback,
        detector_low_hz=config.ecg_detector_low_hz,
        detector_high_hz=config.ecg_detector_high_hz,
        morphology_low_hz=config.ecg_morphology_low_hz,
        morphology_high_hz=config.ecg_morphology_high_hz,
        peak_search_radius_s=config.ecg_peak_search_radius_s,
        quality_min_finite_fraction=config.quality_min_finite_fraction,
        extreme_value_atol_fraction=config.extreme_value_atol_fraction,
        max_interp_gap_s=config.max_interpolated_gap_seconds,
        diagnostics=diagnostics,
    )
    abp = extract_abp_features(
        window[1],
        fs=config.sampling_rate_hz,
        micro_window_samples=config.micro_window_samples,
        template_points=config.morphology_template_points,
        detector_low_hz=config.abp_detector_low_hz,
        detector_high_hz=config.abp_detector_high_hz,
        morphology_high_hz=config.abp_morphology_high_hz,
        peak_search_radius_s=config.abp_peak_search_radius_s,
        trough_search_radius_s=config.abp_trough_search_radius_s,
        min_pulse_bpm=config.abp_min_pulse_bpm,
        max_pulse_bpm=config.abp_max_pulse_bpm,
        quality_min_finite_fraction=config.quality_min_finite_fraction,
        extreme_value_atol_fraction=config.extreme_value_atol_fraction,
        max_interp_gap_s=config.max_interpolated_gap_seconds,
    )
    pleth = extract_pleth_features(
        window[2],
        fs=config.sampling_rate_hz,
        micro_window_samples=config.micro_window_samples,
        template_points=config.morphology_template_points,
        detector_low_hz=config.pleth_detector_low_hz,
        detector_high_hz=config.pleth_detector_high_hz,
        morphology_high_hz=config.pleth_morphology_high_hz,
        peak_search_radius_s=config.pleth_peak_search_radius_s,
        trough_search_radius_s=config.pleth_trough_search_radius_s,
        quality_min_finite_fraction=config.quality_min_finite_fraction,
        extreme_value_atol_fraction=config.extreme_value_atol_fraction,
        max_interp_gap_s=config.max_interpolated_gap_seconds,
    )
    resp = extract_resp_features(
        window[3],
        fs=config.sampling_rate_hz,
        micro_window_samples=config.micro_window_samples,
        template_points=config.morphology_template_points,
        detector_low_hz=config.resp_detector_low_hz,
        detector_high_hz=config.resp_detector_high_hz,
        min_cycle_s=config.resp_min_cycle_s,
        max_cycle_s=config.resp_max_cycle_s,
        quality_min_finite_fraction=config.quality_min_finite_fraction,
        extreme_value_atol_fraction=config.extreme_value_atol_fraction,
        max_interp_gap_s=config.max_interpolated_gap_seconds,
    )
    features = {}
    features.update(ecg)
    features.update(abp)
    features.update(pleth)
    features.update(resp)
    hr_ecg = features.get("ecg_hr_bpm", float("nan"))
    hr_abp = features.get("abp_pulse_rate_bpm", float("nan"))
    hr_pleth = features.get("pleth_pulse_rate_bpm", float("nan"))
    features["cross_ecg_abp_rate_diff_bpm"] = hr_ecg - hr_abp if np.isfinite(hr_ecg) and np.isfinite(hr_abp) else float("nan")
    features["cross_ecg_pleth_rate_diff_bpm"] = hr_ecg - hr_pleth if np.isfinite(hr_ecg) and np.isfinite(hr_pleth) else float("nan")
    features["cross_ecg_abp_rate_agreement"] = agreement_score(hr_ecg, hr_abp)
    features["cross_ecg_pleth_rate_agreement"] = agreement_score(hr_ecg, hr_pleth)
    return features


def _apply_deltas(values: np.ndarray, mask: np.ndarray, feature_order: list[str]) -> None:
    delta_map = {
        "delta_ecg_hr_bpm": "ecg_hr_bpm",
        "delta_abp_map_median_mmhg": "abp_map_median_mmhg",
        "delta_abp_sbp_median_mmhg": "abp_sbp_median_mmhg",
        "delta_abp_dbp_median_mmhg": "abp_dbp_median_mmhg",
        "delta_abp_pulse_pressure_median_mmhg": "abp_pulse_pressure_median_mmhg",
        "delta_pleth_amplitude_median": "pleth_amplitude_median",
        "delta_resp_rate_bpm": "resp_rate_bpm",
    }
    for delta_name, base_name in delta_map.items():
        delta_idx = feature_order.index(delta_name)
        base_idx = feature_order.index(base_name)
        values[:, delta_idx] = np.nan
        mask[:, delta_idx] = False
        for row_idx in range(1, values.shape[0]):
            if mask[row_idx, base_idx] and mask[row_idx - 1, base_idx]:
                values[row_idx, delta_idx] = values[row_idx, base_idx] - values[row_idx - 1, base_idx]
                mask[row_idx, delta_idx] = True


def extract_feature_sequence(
    waveform: np.ndarray,
    config: ExtractionConfig = DEFAULT_EXTRACTION_CONFIG,
    diagnostics: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_order = feature_names()
    values, mask = pad_nan_matrix(config.n_feature_windows, len(feature_order))
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.shape != (len(config.channel_order), config.input_samples):
        raise ValueError(
            f"Expected waveform shape {(len(config.channel_order), config.input_samples)}, got {waveform.shape}"
        )
    for row_idx, slc in enumerate(minute_window_slices(config.input_samples, config.feature_window_samples)):
        minute = waveform[:, slc]
        features = _minute_features(minute, config, diagnostics=diagnostics)
        _populate_row(feature_order, features, values[row_idx], mask[row_idx])
    _apply_deltas(values, mask, feature_order)
    return values, mask, feature_order


def extract_feature_sequence_from_signal(
    waveform: np.ndarray,
    input_start: int,
    input_end: int,
    config: ExtractionConfig = DEFAULT_EXTRACTION_CONFIG,
    diagnostics: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if input_start < 0 or input_end <= input_start:
        raise ValueError(f"Invalid input interval [{input_start}, {input_end})")
    window = np.asarray(waveform[:, input_start:input_end], dtype=np.float64).copy()
    if window.shape[1] != config.input_samples:
        raise ValueError(
            f"Input slice length must equal {config.input_samples} samples, got {window.shape[1]}"
        )
    return extract_feature_sequence(window, config=config, diagnostics=diagnostics)
