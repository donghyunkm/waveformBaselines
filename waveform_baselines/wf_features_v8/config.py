from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from waveform_baselines.wf_features.config import CACHE_ROOT, CHANNEL_ORDER, FEATURE_WINDOW_SECONDS, FS, INPUT_WINDOW_SECONDS, MICRO_WINDOW_SECONDS


@dataclass(frozen=True)
class V8ExtractionConfig:
    sampling_rate_hz: int = FS
    input_window_seconds: int = INPUT_WINDOW_SECONDS
    feature_window_seconds: int = FEATURE_WINDOW_SECONDS
    micro_window_seconds: int = MICRO_WINDOW_SECONDS
    channel_order: tuple[str, ...] = CHANNEL_ORDER
    feature_version: str = "v8"
    quality_min_finite_fraction: float = 1.0
    max_interpolated_gap_seconds: float = 0.2
    morphology_template_points: int = 64
    ecg_detector: str = "xqrs"
    ecg_allow_energy_fallback: bool = True
    enable_cross_signal_timing: bool = False
    enable_abp_advanced_morphology: bool = False
    enable_pulse_deficit_features: bool = False
    enable_pleth_fiducials: bool = False
    enable_pleth_derivative_fiducials: bool = False
    enable_systolic_time_features: bool = False
    rolling_history_seconds: int = 5 * 60
    rolling_rhythm_history_seconds: int = 5 * 60
    minimum_rhythm_rr_count: int = 30
    min_resp_cycles_for_variation: int = 3
    min_pulses_per_resp_cycle: int = 2
    min_baroreflex_pairs: int = 12
    min_baroreflex_coverage_seconds: float = 180.0
    min_hrv_rr_5m: int = 30
    min_dfa_rr_5m: int = 40
    min_hrv_coverage_seconds: float = 240.0
    min_timing_pairs: int = 5
    short_rr_ratio_max: float = 0.80
    long_rr_ratio_min: float = 1.20
    compensatory_sum_ratio_tolerance: float = 0.25
    minimum_local_baseline_intervals: int = 5
    qrs_outlier_correlation_threshold: float = 0.85
    qrs_outlier_distance_threshold: float = 0.60
    pleth_width_levels: tuple[float, ...] = (0.25, 0.50, 0.75)
    pleth_morphology_outlier_threshold: float = 0.60
    abp_decline_min_mmhg: float = 0.5
    minimum_abp_successive_pairs: int = 3
    abp_morphology_outlier_threshold: float = 0.60
    derived_resp_minimum_duration_seconds: float = 240.0
    derived_resp_minimum_events: int = 30
    derived_resp_frequency_low_hz: float = 0.08
    derived_resp_frequency_high_hz: float = 0.50
    derived_resp_minimum_peak_strength: float = 0.20
    derived_resp_peak_half_width_hz: float = 0.015
    derived_resp_frequency_bins: int = 256
    resp_periodic_modulation_low_hz: float = 0.005
    resp_periodic_modulation_high_hz: float = 0.05
    resp_coupling_max_lag_seconds: float = 6.0
    resp_coupling_lag_step_seconds: float = 0.5
    resp_sigh_mad_threshold: float = 4.0
    resp_suppressed_amplitude_fraction: float = 0.25
    quality_step_change_mad_threshold: float = 8.0
    quality_scale_change_ratio_threshold: float = 2.5
    quality_high_frequency_low_hz: float = 8.0
    quality_total_high_hz: float = 20.0
    flat_top_atol_fraction: float = 0.03
    flat_top_min_duration_fraction: float = 0.12
    ecg_abp_pat_bounds_ms: tuple[float, float] = (40.0, 500.0)
    ecg_pleth_pat_bounds_ms: tuple[float, float] = (80.0, 900.0)
    abp_pleth_delay_bounds_ms: tuple[float, float] = (20.0, 600.0)
    min_abp_nonlinear_beats: int = 30
    min_abp_morphology_beats: int = 5
    min_pleth_derivative_beats: int = 5
    min_pleth_morphology_dynamics_beats: int = 30
    pleth_derivative_smoothing_seconds: float = 0.064
    pleth_derivative_polynomial_order: int = 3
    pleth_derivative_minimum_prominence: float = 0.02
    pleth_vpg_min_prominence_fraction: float = 0.05
    pleth_apg_min_prominence_fraction: float = 0.05
    pleth_notch_min_drop_fraction: float = 0.05
    pleth_notch_min_recovery_fraction: float = 0.03
    pleth_notch_min_candidate_score: float = 2.2
    pleth_notch_candidate_score_separation: float = 0.15
    abp_notch_candidate_score_separation: float = 0.15
    abp_notch_min_candidate_score: float = 2.4
    abp_diastolic_peak_candidate_score_separation: float = 0.10
    abp_diastolic_peak_min_candidate_score: float = 2.0
    min_abp_nonlinear_coverage_seconds: float = 180.0
    min_abp_morphology_dynamics_coverage_seconds: float = 180.0
    min_pleth_morphology_dynamics_coverage_seconds: float = 180.0
    min_resp_rrv_cycles: int = 12
    min_resp_rrv_coverage_seconds: float = 240.0
    resp_rrv_frequency_low_hz: float = 0.0033
    resp_rrv_frequency_high_hz: float = 0.20
    resp_rrv_peak_half_width_hz: float = 0.01
    resp_rrv_frequency_bins: int = 256
    resp_min_cycle_seconds: float = 0.75
    resp_max_cycle_seconds: float = 15.0
    resp_pause_min_seconds: float = 10.0
    resp_pause_context_seconds: float = 6.0
    resp_pause_suppression_ratio: float = 0.35
    resp_pause_min_finite_fraction: float = 0.90
    tau_min_dynamic_range_fraction: float = 0.08
    tau_bound_margin_fraction: float = 0.03
    tau_post_rebound_exclusion_seconds: float = 0.04

    def __post_init__(self) -> None:
        if self.ecg_detector not in {"xqrs", "energy"}:
            raise ValueError(f"Unsupported ECG detector {self.ecg_detector!r}")
        if tuple(self.channel_order) != ("II", "ABP", "PLETH", "RESP"):
            raise ValueError(f"v8 requires channel_order ('II', 'ABP', 'PLETH', 'RESP'), got {self.channel_order}")
        if self.feature_window_seconds != 60:
            raise ValueError("v8 fixed schema requires feature_window_seconds == 60")
        if self.rolling_history_seconds != 300:
            raise ValueError("v8 fixed _5m schema requires rolling_history_seconds == 300")
        if self.rolling_rhythm_history_seconds > self.rolling_history_seconds:
            raise ValueError("rolling_rhythm_history_seconds must not exceed rolling_history_seconds")
        if tuple(round(float(level), 2) for level in self.pleth_width_levels) != (0.25, 0.50, 0.75):
            raise ValueError("v8 fixed schema requires pleth_width_levels == (0.25, 0.50, 0.75)")
        positive_names = [
            "sampling_rate_hz",
            "input_window_seconds",
            "feature_window_seconds",
            "micro_window_seconds",
            "rolling_history_seconds",
            "rolling_rhythm_history_seconds",
            "minimum_rhythm_rr_count",
            "min_resp_cycles_for_variation",
            "min_pulses_per_resp_cycle",
            "min_baroreflex_pairs",
            "min_baroreflex_coverage_seconds",
            "min_hrv_rr_5m",
            "min_dfa_rr_5m",
            "min_hrv_coverage_seconds",
            "min_timing_pairs",
            "minimum_local_baseline_intervals",
            "derived_resp_minimum_duration_seconds",
            "derived_resp_minimum_events",
            "resp_coupling_lag_step_seconds",
            "min_abp_nonlinear_beats",
            "min_abp_morphology_beats",
            "min_pleth_derivative_beats",
            "min_pleth_morphology_dynamics_beats",
            "pleth_derivative_smoothing_seconds",
            "pleth_derivative_polynomial_order",
            "pleth_derivative_minimum_prominence",
            "pleth_vpg_min_prominence_fraction",
            "pleth_apg_min_prominence_fraction",
            "pleth_notch_min_drop_fraction",
            "pleth_notch_min_recovery_fraction",
            "pleth_notch_min_candidate_score",
            "pleth_notch_candidate_score_separation",
            "abp_notch_candidate_score_separation",
            "abp_notch_min_candidate_score",
            "abp_diastolic_peak_candidate_score_separation",
            "abp_diastolic_peak_min_candidate_score",
            "min_abp_nonlinear_coverage_seconds",
            "min_abp_morphology_dynamics_coverage_seconds",
            "min_pleth_morphology_dynamics_coverage_seconds",
            "min_resp_rrv_cycles",
            "min_resp_rrv_coverage_seconds",
            "resp_rrv_frequency_low_hz",
            "resp_rrv_frequency_high_hz",
            "resp_rrv_peak_half_width_hz",
            "resp_rrv_frequency_bins",
            "resp_min_cycle_seconds",
            "resp_max_cycle_seconds",
            "resp_pause_min_seconds",
            "resp_pause_context_seconds",
            "resp_pause_suppression_ratio",
            "resp_pause_min_finite_fraction",
            "tau_min_dynamic_range_fraction",
            "tau_bound_margin_fraction",
            "tau_post_rebound_exclusion_seconds",
        ]
        for name in positive_names:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("short_rr_ratio_max", "long_rr_ratio_min", "quality_scale_change_ratio_threshold"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not (0.0 < self.short_rr_ratio_max < 1.0):
            raise ValueError("short_rr_ratio_max must be in (0, 1)")
        if self.long_rr_ratio_min <= 1.0:
            raise ValueError("long_rr_ratio_min must be > 1")
        if not (0.0 < self.derived_resp_frequency_low_hz < self.derived_resp_frequency_high_hz):
            raise ValueError("derived respiratory frequency bounds are invalid")
        if self.derived_resp_peak_half_width_hz <= 0:
            raise ValueError("derived_resp_peak_half_width_hz must be positive")
        if self.derived_resp_frequency_bins < 32:
            raise ValueError("derived_resp_frequency_bins must be at least 32")
        if not (0.0 < self.resp_periodic_modulation_low_hz < self.resp_periodic_modulation_high_hz):
            raise ValueError("RESP periodic modulation frequency bounds are invalid")
        if self.resp_coupling_max_lag_seconds <= 0:
            raise ValueError("resp_coupling_max_lag_seconds must be positive")
        if self.resp_coupling_lag_step_seconds > self.resp_coupling_max_lag_seconds:
            raise ValueError("resp_coupling_lag_step_seconds must not exceed resp_coupling_max_lag_seconds")
        if any(level <= 0.0 or level >= 1.0 for level in self.pleth_width_levels):
            raise ValueError("pleth_width_levels must be fractions in (0, 1)")
        for bounds_name in ("ecg_abp_pat_bounds_ms", "ecg_pleth_pat_bounds_ms", "abp_pleth_delay_bounds_ms"):
            lo, hi = getattr(self, bounds_name)
            if lo <= 0 or hi <= lo:
                raise ValueError(f"{bounds_name} must be positive increasing bounds")
        if self.resp_min_cycle_seconds >= self.resp_max_cycle_seconds:
            raise ValueError("resp_min_cycle_seconds must be less than resp_max_cycle_seconds")
        if self.resp_rrv_frequency_low_hz >= self.resp_rrv_frequency_high_hz:
            raise ValueError("resp_rrv_frequency_low_hz must be less than resp_rrv_frequency_high_hz")
        if self.quality_high_frequency_low_hz >= self.quality_total_high_hz:
            raise ValueError("quality_high_frequency_low_hz must be less than quality_total_high_hz")
        if self.resp_rrv_frequency_bins < 32:
            raise ValueError("resp_rrv_frequency_bins must be at least 32")
        if not (0.0 < self.resp_pause_suppression_ratio < 1.0):
            raise ValueError("resp_pause_suppression_ratio must be in (0, 1)")
        if not (0.0 < self.resp_pause_min_finite_fraction <= 1.0):
            raise ValueError("resp_pause_min_finite_fraction must be in (0, 1]")
        if self.pleth_derivative_polynomial_order < 2:
            raise ValueError("pleth_derivative_polynomial_order must be at least 2")
        if self.enable_pleth_derivative_fiducials and self.pleth_derivative_polynomial_order < 3:
            raise ValueError("JPG-backed PLETH derivative fiducials require pleth_derivative_polynomial_order >= 3")
        derived_window = max(5, int(round(self.pleth_derivative_smoothing_seconds * self.sampling_rate_hz)) | 1)
        if self.pleth_derivative_polynomial_order >= derived_window:
            raise ValueError("pleth_derivative_polynomial_order must be less than the smoothing window")
        if not (0.0 < self.pleth_notch_min_drop_fraction < 1.0):
            raise ValueError("pleth_notch_min_drop_fraction must be in (0, 1)")
        if not (0.0 < self.pleth_notch_min_recovery_fraction < 1.0):
            raise ValueError("pleth_notch_min_recovery_fraction must be in (0, 1)")
        for name in (
            "compensatory_sum_ratio_tolerance",
            "flat_top_atol_fraction",
            "flat_top_min_duration_fraction",
            "derived_resp_minimum_peak_strength",
            "resp_suppressed_amplitude_fraction",
        ):
            value = getattr(self, name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "pleth_notch_candidate_score_separation",
            "abp_notch_candidate_score_separation",
            "abp_diastolic_peak_candidate_score_separation",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.min_hrv_coverage_seconds > self.rolling_history_seconds:
            raise ValueError("min_hrv_coverage_seconds must not exceed rolling_history_seconds")
        if self.min_resp_rrv_coverage_seconds > self.rolling_history_seconds:
            raise ValueError("min_resp_rrv_coverage_seconds must not exceed rolling_history_seconds")
        if self.min_baroreflex_coverage_seconds > self.rolling_history_seconds:
            raise ValueError("min_baroreflex_coverage_seconds must not exceed rolling_history_seconds")

    @property
    def input_samples(self) -> int:
        return self.sampling_rate_hz * self.input_window_seconds

    @property
    def feature_window_samples(self) -> int:
        return self.sampling_rate_hz * self.feature_window_seconds

    @property
    def micro_window_samples(self) -> int:
        return self.sampling_rate_hz * self.micro_window_seconds

    @property
    def n_feature_windows(self) -> int:
        return self.input_window_seconds // self.feature_window_seconds

    @property
    def rolling_history_samples(self) -> int:
        return self.sampling_rate_hz * self.rolling_history_seconds

    @property
    def rolling_rhythm_history_samples(self) -> int:
        return self.sampling_rate_hz * self.rolling_rhythm_history_seconds

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_V8_EXTRACTION_CONFIG = V8ExtractionConfig()
V8_CACHE_ROOT = Path(CACHE_ROOT)
