from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    unit: str
    channel: str
    description: str
    normalize: bool = True


FEATURE_DEFINITIONS = [
    FeatureDefinition("ecg_hr_bpm", "bpm", "ECG", "Median heart rate from valid same-run ECG RR intervals."),
    FeatureDefinition("ecg_rr_median_s", "s", "ECG", "Median valid same-run RR interval."),
    FeatureDefinition("ecg_rr_iqr_s", "s", "ECG", "Valid same-run RR interquartile range."),
    FeatureDefinition("ecg_rr_min_s", "s", "ECG", "Minimum valid same-run RR interval."),
    FeatureDefinition("ecg_rr_max_s", "s", "ECG", "Maximum valid same-run RR interval."),
    FeatureDefinition("ecg_hrv_sdnn_s", "s", "ECG", "SDNN over valid same-run RR intervals."),
    FeatureDefinition("ecg_hrv_rmssd_s", "s", "ECG", "RMSSD over valid adjacent valid RR pairs."),
    FeatureDefinition("ecg_hrv_sdsd_s", "s", "ECG", "SDSD over valid adjacent valid RR pairs."),
    FeatureDefinition("ecg_hrv_pnn20", "fraction", "ECG", "Fraction of valid adjacent RR differences > 20 ms.", normalize=False),
    FeatureDefinition("ecg_hrv_pnn50", "fraction", "ECG", "Fraction of valid adjacent RR differences > 50 ms.", normalize=False),
    FeatureDefinition("ecg_r_amp_median", "mV_or_native", "ECG", "Median calibrated R-peak amplitude."),
    FeatureDefinition("ecg_r_amp_iqr", "mV_or_native", "ECG", "IQR of R-peak amplitude."),
    FeatureDefinition("ecg_qrs_width_median_s", "s", "ECG", "Approximate QRS width from peak width."),
    FeatureDefinition("ecg_qrs_width_iqr_s", "s", "ECG", "QRS width IQR."),
    FeatureDefinition("ecg_max_abs_slope", "unit_per_s", "ECG", "Maximum absolute ECG slope within finite morphology runs."),
    FeatureDefinition("ecg_morphology_consistency", "corr", "ECG", "Median beat-template correlation.", normalize=False),
    FeatureDefinition("ecg_valid_micro_fraction", "fraction", "ECG", "Fraction of valid ECG micro-windows.", normalize=False),
    FeatureDefinition("ecg_missing_micro_fraction", "fraction", "ECG", "Fraction of ECG micro-windows with insufficient finite coverage.", normalize=False),
    FeatureDefinition("ecg_plausible_beat_fraction", "fraction", "ECG", "Fraction of same-run RR intervals in physiologic range.", normalize=False),
    FeatureDefinition("ecg_flatline_fraction", "fraction", "ECG", "Fraction of flat ECG micro-windows.", normalize=False),
    FeatureDefinition("ecg_extreme_value_fraction", "fraction", "ECG", "Fraction of ECG samples within 1% of the minute's observed minimum or maximum.", normalize=False),
    FeatureDefinition("abp_sbp_median_mmhg", "mmHg", "ABP", "Median refined raw systolic blood pressure."),
    FeatureDefinition("abp_dbp_median_mmhg", "mmHg", "ABP", "Median refined raw diastolic blood pressure."),
    FeatureDefinition("abp_map_median_mmhg", "mmHg", "ABP", "Median beat-mean arterial pressure."),
    FeatureDefinition("abp_pulse_pressure_median_mmhg", "mmHg", "ABP", "Median pulse pressure."),
    FeatureDefinition("abp_pulse_rate_bpm", "bpm", "ABP", "Pulse rate from ABP beats."),
    FeatureDefinition("abp_sbp_sd_mmhg", "mmHg", "ABP", "SBP standard deviation."),
    FeatureDefinition("abp_sbp_iqr_mmhg", "mmHg", "ABP", "SBP IQR."),
    FeatureDefinition("abp_dbp_sd_mmhg", "mmHg", "ABP", "DBP standard deviation."),
    FeatureDefinition("abp_dbp_iqr_mmhg", "mmHg", "ABP", "DBP IQR."),
    FeatureDefinition("abp_map_sd_mmhg", "mmHg", "ABP", "MAP standard deviation."),
    FeatureDefinition("abp_map_iqr_mmhg", "mmHg", "ABP", "MAP IQR."),
    FeatureDefinition("abp_pp_sd_mmhg", "mmHg", "ABP", "Pulse pressure standard deviation."),
    FeatureDefinition("abp_pp_iqr_mmhg", "mmHg", "ABP", "Pulse pressure IQR."),
    FeatureDefinition("abp_upstroke_slope_median", "mmHg_per_s", "ABP", "Median refined raw ABP upstroke slope."),
    FeatureDefinition("abp_dpdt_max_median", "mmHg_per_s", "ABP", "Median maximum positive dP/dt from morphology-preserving ABP."),
    FeatureDefinition("abp_dpdt_min_median", "mmHg_per_s", "ABP", "Median maximum negative dP/dt from morphology-preserving ABP."),
    FeatureDefinition("abp_pulse_area_median", "mmHg_s", "ABP", "Median raw ABP beat area above refined foot."),
    FeatureDefinition("abp_pulse_width_median_s", "s", "ABP", "Median refined trough-to-trough pulse width."),
    FeatureDefinition("abp_upstroke_duration_median_s", "s", "ABP", "Median refined foot-to-peak duration."),
    FeatureDefinition("abp_decay_duration_median_s", "s", "ABP", "Median refined peak-to-next-foot duration."),
    FeatureDefinition("abp_upstroke_decay_ratio_median", "ratio", "ABP", "Median refined upstroke/decay timing ratio."),
    FeatureDefinition("abp_morphology_consistency", "corr", "ABP", "Median valid-beat template correlation.", normalize=False),
    FeatureDefinition("abp_valid_pulse_fraction", "fraction", "ABP", "Fraction of valid ABP pulses.", normalize=False),
    FeatureDefinition("abp_plausible_sbp_fraction", "fraction", "ABP", "Fraction of pulses with plausible SBP.", normalize=False),
    FeatureDefinition("abp_plausible_dbp_fraction", "fraction", "ABP", "Fraction of pulses with plausible DBP.", normalize=False),
    FeatureDefinition("abp_sbp_gt_dbp_fraction", "fraction", "ABP", "Fraction of pulses with SBP > DBP.", normalize=False),
    FeatureDefinition("abp_valid_micro_fraction", "fraction", "ABP", "Fraction of valid ABP micro-windows.", normalize=False),
    FeatureDefinition("abp_missing_micro_fraction", "fraction", "ABP", "Fraction of ABP micro-windows with insufficient finite coverage.", normalize=False),
    FeatureDefinition("abp_flatline_fraction", "fraction", "ABP", "Fraction of flat ABP micro-windows.", normalize=False),
    FeatureDefinition("abp_extreme_value_fraction", "fraction", "ABP", "Fraction of ABP samples within 1% of the minute's observed minimum or maximum.", normalize=False),
    FeatureDefinition("pleth_pulse_rate_bpm", "bpm", "PLETH", "Pulse rate from pleth pulses."),
    FeatureDefinition("pleth_amplitude_median", "native", "PLETH", "Median pulse amplitude."),
    FeatureDefinition("pleth_amplitude_iqr", "native", "PLETH", "Pulse amplitude IQR."),
    FeatureDefinition("pleth_rise_time_median_s", "s", "PLETH", "Median refined foot-to-peak rise time."),
    FeatureDefinition("pleth_decay_time_median_s", "s", "PLETH", "Median refined peak-to-next-foot decay time."),
    FeatureDefinition("pleth_rise_slope_median", "unit_per_s", "PLETH", "Median rise slope."),
    FeatureDefinition("pleth_decay_slope_median", "unit_per_s", "PLETH", "Median decay slope."),
    FeatureDefinition("pleth_width_median_s", "s", "PLETH", "Median refined trough-to-trough pulse width."),
    FeatureDefinition("pleth_area_median", "unit_s", "PLETH", "Median pulse area above refined foot."),
    FeatureDefinition("pleth_morphology_consistency", "corr", "PLETH", "Median valid-pulse template correlation.", normalize=False),
    FeatureDefinition("pleth_valid_pulse_fraction", "fraction", "PLETH", "Fraction of valid pleth pulses.", normalize=False),
    FeatureDefinition("pleth_valid_micro_fraction", "fraction", "PLETH", "Fraction of valid pleth micro-windows.", normalize=False),
    FeatureDefinition("pleth_missing_micro_fraction", "fraction", "PLETH", "Fraction of PLETH micro-windows with insufficient finite coverage.", normalize=False),
    FeatureDefinition("pleth_flatline_fraction", "fraction", "PLETH", "Fraction of flat pleth micro-windows.", normalize=False),
    FeatureDefinition("pleth_extreme_value_fraction", "fraction", "PLETH", "Fraction of PLETH samples within 1% of the minute's observed minimum or maximum.", normalize=False),
    FeatureDefinition("resp_rate_bpm", "breaths_per_min", "RESP", "Respiratory rate from accepted trough-to-peak-to-trough cycles."),
    FeatureDefinition("resp_cycle_length_median_s", "s", "RESP", "Median accepted respiratory cycle length."),
    FeatureDefinition("resp_cycle_length_iqr_s", "s", "RESP", "Accepted respiratory cycle-length IQR."),
    FeatureDefinition("resp_amplitude_median", "native", "RESP", "Median respiratory amplitude."),
    FeatureDefinition("resp_amplitude_iqr", "native", "RESP", "Respiratory amplitude IQR."),
    FeatureDefinition("resp_rise_time_median_s", "s", "RESP", "Median trough-to-peak rise time for accepted RESP cycles."),
    FeatureDefinition("resp_fall_time_median_s", "s", "RESP", "Median peak-to-trough fall time for accepted RESP cycles."),
    FeatureDefinition("resp_rise_fall_ratio_median", "ratio", "RESP", "Median RESP rise/fall timing ratio."),
    FeatureDefinition("resp_rise_slope_median", "unit_per_s", "RESP", "Median RESP rise slope."),
    FeatureDefinition("resp_fall_slope_median", "unit_per_s", "RESP", "Median RESP fall slope."),
    FeatureDefinition("resp_cycle_area_median", "unit_s", "RESP", "Median cycle area."),
    FeatureDefinition("resp_morphology_consistency", "corr", "RESP", "Median accepted-cycle template correlation.", normalize=False),
    FeatureDefinition("resp_valid_cycle_fraction", "fraction", "RESP", "Accepted trough-start cycles divided by candidate trough-start cycles.", normalize=False),
    FeatureDefinition("resp_valid_micro_fraction", "fraction", "RESP", "Fraction of valid resp micro-windows.", normalize=False),
    FeatureDefinition("resp_missing_micro_fraction", "fraction", "RESP", "Fraction of RESP micro-windows with insufficient finite coverage.", normalize=False),
    FeatureDefinition("resp_flatline_fraction", "fraction", "RESP", "Fraction of flat resp micro-windows.", normalize=False),
    FeatureDefinition("cross_ecg_abp_rate_diff_bpm", "bpm", "CROSS", "ECG heart rate minus ABP pulse rate."),
    FeatureDefinition("cross_ecg_pleth_rate_diff_bpm", "bpm", "CROSS", "ECG heart rate minus pleth pulse rate."),
    FeatureDefinition("cross_ecg_abp_rate_agreement", "fraction", "CROSS", "1 / (1 + |HR_ecg - HR_abp| / 10).", normalize=False),
    FeatureDefinition("cross_ecg_pleth_rate_agreement", "fraction", "CROSS", "1 / (1 + |HR_ecg - HR_pleth| / 10).", normalize=False),
    FeatureDefinition("delta_ecg_hr_bpm", "bpm", "DELTA", "Current ECG heart rate minus previous minute."),
    FeatureDefinition("delta_abp_map_median_mmhg", "mmHg", "DELTA", "Current MAP minus previous minute."),
    FeatureDefinition("delta_abp_sbp_median_mmhg", "mmHg", "DELTA", "Current SBP minus previous minute."),
    FeatureDefinition("delta_abp_dbp_median_mmhg", "mmHg", "DELTA", "Current DBP minus previous minute."),
    FeatureDefinition("delta_abp_pulse_pressure_median_mmhg", "mmHg", "DELTA", "Current pulse pressure minus previous minute."),
    FeatureDefinition("delta_pleth_amplitude_median", "native", "DELTA", "Current pleth amplitude minus previous minute."),
    FeatureDefinition("delta_resp_rate_bpm", "breaths_per_min", "DELTA", "Current respiratory rate minus previous minute."),
]


def feature_names() -> list[str]:
    return [feature.name for feature in FEATURE_DEFINITIONS]


def feature_definition_map() -> dict[str, FeatureDefinition]:
    return {feature.name: feature for feature in FEATURE_DEFINITIONS}
