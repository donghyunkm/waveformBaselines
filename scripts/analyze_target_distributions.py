#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


TARGET_UNITS = {
    "HR": "bpm",
    "RR": "breaths/min",
    "SBP": "mmHg",
    "DBP": "mmHg",
    "PP": "mmHg",
    "MAP": "mmHg",
    "ABP_area": "mmHg*s (inferred)",
    "PLETH_ACDC": "ratio",
    "PLETH_amp": "pleth AU",
    "ECG_Ramp": "ECG AU",
    "HRV_RMSSD": "ms (inferred)",
    "HR_range": "bpm",
    "ShockIdx": "ratio",
    "PPV": "%",
    "PVI": "%",
    "PTT": "ms (inferred)",
    "dPdt_max": "mmHg/s (inferred)",
    "ABP_tau": "s (inferred)",
    "RESP_amp": "resp AU",
    "PLETH_ACDC_PLETH_amp": "correlation coefficient",
    "ABP_area_ABP_tau": "correlation coefficient",
    "ABP_area_ShockIdx": "correlation coefficient",
    "PLETH_amp_ShockIdx": "correlation coefficient",
    "PLETH_ACDC_ShockIdx": "correlation coefficient",
    "ShockIdx_ABP_tau": "correlation coefficient",
    "PLETH_ACDC_ABP_tau": "correlation coefficient",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze distribution shape for the vasopressor-free t+0 gap regression targets."
    )
    parser.add_argument("--target-path", type=Path, required=True)
    parser.add_argument("--splits-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-doc", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--date", type=str, required=True)
    return parser.parse_args()


def slugify(value: str) -> str:
    return value.replace("%", "pct").replace("+", "plus").replace("/", "_").replace(" ", "_")


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if not np.isfinite(value):
        return "NA"
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:,.1f}"
    if abs_value >= 100:
        return f"{value:,.2f}"
    if abs_value >= 1:
        return f"{value:,.3f}"
    return f"{value:,.4f}"


def format_percent(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{100.0 * value:.{digits}f}%"


def load_target_names(metadata_path: Path) -> list[str]:
    metadata = json.loads(metadata_path.read_text())
    return [str(name) for name in metadata["feature_target_names"]]


def patient_split_masks(patient_ids: np.ndarray, splits: dict[str, list[str]]) -> dict[str, np.ndarray]:
    patient_ids = patient_ids.astype(str)
    masks = {}
    for split_name in ("train", "val", "test"):
        split_patients = set(splits.get(split_name, []))
        masks[split_name] = np.array([pid in split_patients for pid in patient_ids], dtype=bool)
    return masks


def safe_std(values: np.ndarray) -> float:
    return float(values.std()) if values.size else float("nan")


def safe_skewness(values: np.ndarray) -> float:
    if values.size < 3:
        return float("nan")
    centered = values - values.mean()
    std = values.std()
    if std == 0:
        return 0.0
    return float(np.mean((centered / std) ** 3))


def safe_kurtosis(values: np.ndarray) -> float:
    if values.size < 4:
        return float("nan")
    centered = values - values.mean()
    std = values.std()
    if std == 0:
        return 0.0
    return float(np.mean((centered / std) ** 4) - 3.0)


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size == 0 or y.size == 0:
        return None
    x = np.sort(x)
    y = np.sort(y)
    merged = np.sort(np.concatenate([x, y]))
    x_cdf = np.searchsorted(x, merged, side="right") / x.size
    y_cdf = np.searchsorted(y, merged, side="right") / y.size
    return float(np.max(np.abs(x_cdf - y_cdf)))


def multimodality_heuristic(values: np.ndarray) -> bool:
    if values.size < 200:
        return False
    q01, q99 = np.percentile(values, [1, 99])
    focused = values[(values >= q01) & (values <= q99)]
    if focused.size < 200:
        return False
    counts, _ = np.histogram(focused, bins="fd")
    if counts.size < 15:
        return False
    smooth = np.convolve(counts, np.ones(3) / 3.0, mode="same")
    max_count = float(smooth.max())
    if max_count <= 0:
        return False
    peak_indices: list[int] = []
    for idx in range(2, len(smooth) - 2):
        if smooth[idx] > smooth[idx - 1] and smooth[idx] >= smooth[idx + 1] and smooth[idx] >= 0.25 * max_count:
            peak_indices.append(idx)
    if len(peak_indices) < 2:
        return False
    for left, right in zip(peak_indices[:-1], peak_indices[1:]):
        if right - left < 3:
            continue
        valley = float(smooth[left:right + 1].min())
        smaller_peak = min(float(smooth[left]), float(smooth[right]))
        if smaller_peak > 0 and valley <= 0.75 * smaller_peak:
            return True
    return False


def summarize_distribution(values: np.ndarray) -> dict[str, float | int | None]:
    n_valid = int(values.size)
    if n_valid == 0:
        return {
            "n_valid": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "iqr": None,
            "skewness": None,
            "kurtosis": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "n_zero": 0,
            "frac_zero": 0.0,
            "n_negative": 0,
            "frac_negative": 0.0,
            "n_boundary_abs_ge_0p999": 0,
            "frac_boundary_abs_ge_0p999": 0.0,
        }

    values = values.astype(np.float64, copy=False)
    q01, q05, q25, q50, q75, q95, q99 = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    n_zero = int(np.count_nonzero(values == 0))
    n_negative = int(np.count_nonzero(values < 0))
    return {
        "n_valid": n_valid,
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": safe_std(values),
        "iqr": float(q75 - q25),
        "skewness": safe_skewness(values),
        "kurtosis": safe_kurtosis(values),
        "p01": float(q01),
        "p05": float(q05),
        "p25": float(q25),
        "p50": float(q50),
        "p75": float(q75),
        "p95": float(q95),
        "p99": float(q99),
        "n_zero": n_zero,
        "frac_zero": float(n_zero / n_valid),
        "n_negative": n_negative,
        "frac_negative": float(n_negative / n_valid),
        "n_boundary_abs_ge_0p999": int(np.count_nonzero(np.abs(values) >= 0.999)),
        "frac_boundary_abs_ge_0p999": float(np.count_nonzero(np.abs(values) >= 0.999) / n_valid),
    }


def outlier_summary(values: np.ndarray, lower_fence: float, upper_fence: float) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "n_outliers": 0,
            "frac_outliers": 0.0,
            "n_low_outliers": 0,
            "n_high_outliers": 0,
        }
    low_mask = values < lower_fence
    high_mask = values > upper_fence
    outlier_mask = low_mask | high_mask
    n_outliers = int(outlier_mask.sum())
    return {
        "n_outliers": n_outliers,
        "frac_outliers": float(n_outliers / values.size),
        "n_low_outliers": int(low_mask.sum()),
        "n_high_outliers": int(high_mask.sum()),
    }


def patient_concentration(
    patient_ids: np.ndarray,
    outlier_mask: np.ndarray,
) -> dict[str, float | int]:
    outlier_patients = patient_ids[outlier_mask]
    n_outliers = int(outlier_patients.size)
    if n_outliers == 0:
        return {
            "n_outliers": 0,
            "n_patients": 0,
            "top1_share": 0.0,
            "top5_share": 0.0,
        }
    counts = Counter(outlier_patients.tolist())
    ranked = [count for _, count in counts.most_common()]
    top1 = ranked[0]
    top5 = sum(ranked[:5])
    return {
        "n_outliers": n_outliers,
        "n_patients": len(counts),
        "top1_share": float(top1 / n_outliers),
        "top5_share": float(top5 / n_outliers),
    }


def build_shape_flags(
    target_name: str,
    train_stats: dict[str, float | int | None],
    multimodal: bool,
    concentration: dict[str, float | int],
    outliers: dict[str, float | int],
) -> list[str]:
    flags: list[str] = []
    skewness = float(train_stats["skewness"]) if train_stats["skewness"] is not None else float("nan")
    kurtosis = float(train_stats["kurtosis"]) if train_stats["kurtosis"] is not None else float("nan")
    frac_zero = float(train_stats["frac_zero"]) if train_stats["frac_zero"] is not None else 0.0
    frac_boundary = (
        float(train_stats["frac_boundary_abs_ge_0p999"])
        if train_stats["frac_boundary_abs_ge_0p999"] is not None
        else 0.0
    )
    min_value = float(train_stats["min"]) if train_stats["min"] is not None else float("nan")
    max_value = float(train_stats["max"]) if train_stats["max"] is not None else float("nan")
    frac_outliers = float(outliers["frac_outliers"])

    if np.isfinite(skewness) and abs(skewness) < 0.5:
        flags.append("approximately symmetric")
    elif np.isfinite(skewness) and skewness >= 0.5:
        flags.append("right-skewed")
    elif np.isfinite(skewness) and skewness <= -0.5:
        flags.append("left-skewed")

    if np.isfinite(kurtosis) and kurtosis >= 3.0:
        flags.append("heavy-tailed")
    elif frac_outliers >= 0.05:
        flags.append("heavy-tailed")

    if multimodal:
        flags.append("possible multimodality")

    if frac_zero >= 0.05:
        flags.append("zero-inflated")

    if (
        TARGET_UNITS.get(target_name) == "correlation coefficient"
        and np.isfinite(min_value)
        and np.isfinite(max_value)
        and min_value >= -1.01
        and max_value <= 1.01
        and frac_boundary >= 0.01
    ):
        flags.append("bounded with spike at +/-1")

    if frac_outliers >= 0.01:
        flags.append("affected by extreme values")

    if (
        frac_outliers >= 0.005
        and float(concentration["top5_share"]) >= 0.5
        and int(concentration["n_outliers"]) >= 100
    ):
        flags.append("train outliers concentrated in a few patients")

    if not flags:
        flags.append("roughly compact/unimodal")
    return flags


def assess_shift(
    train_values: np.ndarray,
    other_values: np.ndarray,
    train_stats: dict[str, float | int | None],
    other_stats: dict[str, float | int | None],
) -> dict[str, float | str | None]:
    ks = ks_statistic(train_values, other_values)
    if train_stats["median"] is None or other_stats["median"] is None or train_stats["iqr"] in (None, 0):
        median_shift_iqr = None
    else:
        median_shift_iqr = float(
            abs(float(other_stats["median"]) - float(train_stats["median"])) / max(float(train_stats["iqr"]), 1e-8)
        )
    if train_stats["std"] is None or other_stats["std"] is None or float(train_stats["std"]) == 0:
        std_ratio = None
    else:
        std_ratio = float(float(other_stats["std"]) / float(train_stats["std"]))

    if ks is None:
        label = "insufficient data"
    elif ks >= 0.12 or (median_shift_iqr is not None and median_shift_iqr >= 0.5):
        label = "possible meaningful shift"
    elif ks >= 0.08 or (median_shift_iqr is not None and median_shift_iqr >= 0.25):
        label = "mild shift"
    else:
        label = "no clear shift"

    return {
        "ks": ks,
        "median_shift_iqr": median_shift_iqr,
        "std_ratio": std_ratio,
        "label": label,
    }


def transformation_flag(
    target_name: str,
    train_stats: dict[str, float | int | None],
    flags: list[str],
) -> str:
    skewness = float(train_stats["skewness"]) if train_stats["skewness"] is not None else float("nan")
    min_value = float(train_stats["min"]) if train_stats["min"] is not None else float("nan")
    frac_zero = float(train_stats["frac_zero"]) if train_stats["frac_zero"] is not None else 0.0
    frac_negative = float(train_stats["frac_negative"]) if train_stats["frac_negative"] is not None else 0.0
    frac_boundary = (
        float(train_stats["frac_boundary_abs_ge_0p999"])
        if train_stats["frac_boundary_abs_ge_0p999"] is not None
        else 0.0
    )

    if TARGET_UNITS.get(target_name) == "correlation coefficient" and frac_boundary >= 0.01:
        return "bounded correlation feature; simple z-score may be insufficient, but log/Box-Cox are not appropriate"

    if frac_negative > 0 and (abs(skewness) >= 0.75 or "heavy-tailed" in flags):
        return "investigate Yeo-Johnson; z-score alone may be insufficient"
    if min_value > 0 and ("heavy-tailed" in flags or skewness >= 0.75):
        return "investigate Box-Cox or log-type transform"
    if min_value >= 0 and frac_zero > 0 and ("heavy-tailed" in flags or skewness >= 0.75):
        return "investigate log1p"
    if min_value >= 0 and ("heavy-tailed" in flags or skewness >= 0.75):
        return "investigate log1p or Box-Cox"
    return "simple z-score normalization is likely enough if any normalization is used"


def build_target_interpretation(
    target_name: str,
    units: str,
    train_stats: dict[str, float | int | None],
    shape_flags: list[str],
    shift_summary: str,
    transform_note: str,
) -> str:
    median = format_float(train_stats["median"])
    iqr = format_float(train_stats["iqr"])
    zero_pct = format_percent(float(train_stats["frac_zero"]))
    boundary_note = ""
    min_value = float(train_stats["min"]) if train_stats["min"] is not None else float("nan")
    max_value = float(train_stats["max"]) if train_stats["max"] is not None else float("nan")
    if (
        units == "correlation coefficient"
        and np.isfinite(min_value)
        and np.isfinite(max_value)
        and min_value >= -1.01
        and max_value <= 1.01
        and float(train_stats["frac_boundary_abs_ge_0p999"] or 0.0) >= 0.01
    ):
        boundary_note = " includes noticeable mass at +/-1."
    return (
        f"`{target_name}` ({units}) has median `{median}` and IQR `{iqr}`; "
        f"{'; '.join(shape_flags)}; zeros `{zero_pct}`; split comparison `{shift_summary}`; "
        f"{transform_note}.{boundary_note}"
    )


def upper_tail_ratio(summary: dict[str, float | int | None]) -> float | None:
    if summary["p99"] is None or summary["p50"] is None or summary["iqr"] in (None, 0):
        return None
    return float((float(summary["p99"]) - float(summary["p50"])) / max(float(summary["iqr"]), 1e-8))


def lower_tail_ratio(summary: dict[str, float | int | None]) -> float | None:
    if summary["p50"] is None or summary["p01"] is None or summary["iqr"] in (None, 0):
        return None
    return float((float(summary["p50"]) - float(summary["p01"])) / max(float(summary["iqr"]), 1e-8))


def zscore_transform(values: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
    mean = float(values.mean())
    std = float(values.std())
    if std == 0:
        raise ValueError("Cannot z-score constant target")
    transformed = (values - mean) / std
    return transformed, {"mean": mean, "std": std}


def fit_boxcox(values: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
    lam = float(stats.boxcox_normmax(values, method="mle"))
    transformed = stats.boxcox(values, lmbda=lam)
    return np.asarray(transformed, dtype=np.float64), {"lambda": lam}


def fit_yeojohnson(values: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
    transformed, lam = stats.yeojohnson(values)
    return np.asarray(transformed, dtype=np.float64), {"lambda": float(lam)}


def transformation_interpretation(
    target_name: str,
    units: str,
    name: str,
    lambda_value: float | None,
    raw_summary: dict[str, float | int | None],
    transformed_summary: dict[str, float | int | None] | None,
) -> str:
    if transformed_summary is None:
        return "not evaluated"
    raw_skew = float(raw_summary["skewness"]) if raw_summary["skewness"] is not None else float("nan")
    new_skew = float(transformed_summary["skewness"]) if transformed_summary["skewness"] is not None else float("nan")
    raw_kurt = float(raw_summary["kurtosis"]) if raw_summary["kurtosis"] is not None else float("nan")
    new_kurt = float(transformed_summary["kurtosis"]) if transformed_summary["kurtosis"] is not None else float("nan")
    raw_tail = upper_tail_ratio(raw_summary)
    new_tail = upper_tail_ratio(transformed_summary)
    raw_range = (
        float(raw_summary["max"]) - float(raw_summary["min"])
        if raw_summary["max"] is not None and raw_summary["min"] is not None
        else float("nan")
    )
    new_range = (
        float(transformed_summary["max"]) - float(transformed_summary["min"])
        if transformed_summary["max"] is not None and transformed_summary["min"] is not None
        else float("nan")
    )

    if name == "raw":
        return "reference raw distribution"
    if name == "zscore":
        skew_delta = abs(new_skew - raw_skew) if np.isfinite(new_skew) and np.isfinite(raw_skew) else float("nan")
        kurt_delta = abs(new_kurt - raw_kurt) if np.isfinite(new_kurt) and np.isfinite(raw_kurt) else float("nan")
        if skew_delta < 1e-8 and kurt_delta < 1e-8:
            return "scale changed only; shape unchanged as expected"
        return "scale changed only; shape drift should be checked"

    skew_improvement = abs(raw_skew) - abs(new_skew) if np.isfinite(raw_skew) and np.isfinite(new_skew) else float("nan")
    kurt_improvement = raw_kurt - new_kurt if np.isfinite(raw_kurt) and np.isfinite(new_kurt) else float("nan")
    tail_improvement = raw_tail - new_tail if raw_tail is not None and new_tail is not None else None

    notes: list[str] = []
    if np.isfinite(skew_improvement):
        if skew_improvement >= 0.5:
            notes.append("substantially reduces skewness")
        elif skew_improvement >= 0.2:
            notes.append("moderately reduces skewness")
        elif skew_improvement <= -0.2:
            notes.append("worsens skewness")
    if np.isfinite(kurt_improvement):
        if kurt_improvement >= 1.0:
            notes.append("reduces heavy-tail behavior")
        elif kurt_improvement <= -1.0:
            notes.append("increases heavy-tail behavior")
    if tail_improvement is not None:
        if tail_improvement >= 0.5:
            notes.append("compresses the upper tail")
        elif tail_improvement <= -0.5:
            notes.append("stretches the upper tail relative to the center")
    if (
        units == "correlation coefficient"
        and lambda_value is not None
        and abs(lambda_value) >= 5.0
    ):
        notes.append("introduces an extreme learned lambda")
    if (
        units == "correlation coefficient"
        and np.isfinite(raw_range)
        and raw_range > 0
        and np.isfinite(new_range)
        and new_range / raw_range >= 5.0
    ):
        notes.append("expands the bounded scale aggressively")
    if not notes:
        notes.append("little shape change")
    return "; ".join(notes)


def recommendation_for_target(
    target_name: str,
    units: str,
    raw_summary: dict[str, float | int | None],
    comparisons: list[dict[str, object]],
) -> str:
    raw_skew = abs(float(raw_summary["skewness"])) if raw_summary["skewness"] is not None else float("nan")
    raw_kurt = float(raw_summary["kurtosis"]) if raw_summary["kurtosis"] is not None else float("nan")
    raw_range = (
        float(raw_summary["max"]) - float(raw_summary["min"])
        if raw_summary["max"] is not None and raw_summary["min"] is not None
        else float("nan")
    )
    candidates = [row for row in comparisons if row["valid"] and row["transformation"] not in {"raw", "zscore"}]

    def candidate_score(row: dict[str, object]) -> float:
        after = row["after_stats"]
        if after is None:
            return -math.inf
        new_skew = abs(float(after["skewness"]))
        new_kurt = float(after["kurtosis"])
        score = 0.0
        score += max(0.0, raw_skew - new_skew) * 2.0
        if np.isfinite(raw_kurt):
            score += max(0.0, raw_kurt - new_kurt) * 0.5
        score -= max(0.0, new_skew - 0.35)
        if row["transformation"] == "log1p":
            score += 0.4
        if row["transformation"] == "boxcox":
            score += 0.25
        if row["transformation"] == "yeo_johnson":
            score += 0.05
        lambda_value = row.get("lambda")
        if lambda_value is not None and abs(float(lambda_value)) >= 5.0:
            score -= 2.0
        if after["max"] is not None and after["min"] is not None and np.isfinite(raw_range) and raw_range > 0:
            new_range = float(after["max"]) - float(after["min"])
            if new_range / raw_range >= 5.0:
                score -= 1.0
        return score

    best_candidate: dict[str, object] | None = None
    best_score = -math.inf
    for row in candidates:
        score = candidate_score(row)
        if score > best_score:
            best_score = score
            best_candidate = row

    if units == "correlation coefficient":
        yj_row = next((row for row in comparisons if row["transformation"] == "yeo_johnson" and row["valid"]), None)
        if yj_row is not None:
            after = yj_row["after_stats"]
            lambda_value = yj_row.get("lambda")
            if after is not None and lambda_value is not None:
                new_range = float(after["max"]) - float(after["min"])
                if (
                    abs(float(after["skewness"])) <= 0.5
                    and abs(float(lambda_value)) < 5.0
                    and (not np.isfinite(raw_range) or raw_range <= 0 or new_range / raw_range < 5.0)
                    and raw_skew >= 1.0
                ):
                    return "keep raw target as baseline; test Yeo-Johnson + z-score"
        return "keep raw target as baseline; test z-score only"

    if np.isfinite(raw_skew) and raw_skew < 0.75 and (not np.isfinite(raw_kurt) or raw_kurt < 2.0):
        if raw_skew < 0.65:
            return "keep raw target as baseline; transformation probably unnecessary"
        if best_candidate is None:
            return "keep raw target as baseline; transformation probably unnecessary"
        after = best_candidate["after_stats"]
        if after is None:
            return "keep raw target as baseline; transformation probably unnecessary"
        moderate_gain = (
            raw_skew - abs(float(after["skewness"])) >= 0.45
            and max(0.0, raw_kurt) - max(0.0, float(after["kurtosis"])) >= 0.3
        )
        if not moderate_gain:
            return "keep raw target as baseline; transformation probably unnecessary"

    if float(raw_summary["skewness"]) <= -0.75:
        yj_row = next((row for row in comparisons if row["transformation"] == "yeo_johnson" and row["valid"]), None)
        if yj_row is not None:
            after = yj_row["after_stats"]
            if after is not None and (
                abs(float(after["skewness"])) <= 0.35
                or raw_skew - abs(float(after["skewness"])) >= 0.6
            ):
                return "keep raw target as baseline; test Yeo-Johnson + z-score"
        return "keep raw target as baseline; test z-score only"

    if best_candidate is None:
        return "keep raw target as baseline; test z-score only"

    name = str(best_candidate["transformation"])
    after = best_candidate["after_stats"]
    if after is None:
        return "keep raw target as baseline; test z-score only"
    skew_after = abs(float(after["skewness"]))
    kurt_after = float(after["kurtosis"])
    raw_score = raw_skew + max(0.0, raw_kurt)
    new_score = skew_after + max(0.0, kurt_after)

    if new_score > raw_score - 0.4:
        return "keep raw target as baseline; test z-score only"
    if name == "boxcox" and raw_skew >= 0.65 and skew_after <= 0.1:
        log_row = next((row for row in comparisons if row["transformation"] == "log1p" and row["valid"]), None)
        if log_row is None:
            return "keep raw target as baseline; test Box-Cox + z-score"
        log_after = log_row["after_stats"]
        if log_after is not None:
            log_shape = abs(float(log_after["skewness"])) + max(0.0, float(log_after["kurtosis"]))
            box_shape = skew_after + max(0.0, kurt_after)
            if log_shape > box_shape + 0.5:
                return "keep raw target as baseline; test Box-Cox + z-score"
    if (
        name == "boxcox"
        and next((row for row in comparisons if row["transformation"] == "log1p" and row["valid"]), None) is not None
    ):
        log_row = next(row for row in comparisons if row["transformation"] == "log1p" and row["valid"])
        log_after = log_row["after_stats"]
        if log_after is not None:
            log_score = abs(float(log_after["skewness"])) + max(0.0, float(log_after["kurtosis"]))
            box_score = skew_after + max(0.0, kurt_after)
            if log_score <= box_score + 0.2:
                return "keep raw target as baseline; test log1p + z-score"
    if name == "log1p":
        return "keep raw target as baseline; test log1p + z-score"
    if name == "boxcox":
        return "keep raw target as baseline; test Box-Cox + z-score"
    if name == "yeo_johnson":
        return "keep raw target as baseline; test Yeo-Johnson + z-score"
    return "keep raw target as baseline; test z-score only"


def render_transform_comparison(
    target_name: str,
    transforms: list[dict[str, object]],
    path: Path,
) -> None:
    n = len(transforms)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 6.2), squeeze=False)
    for col, row in enumerate(transforms):
        values = np.asarray(row["values"], dtype=np.float64)
        hist_ax = axes[0, col]
        ecdf_ax = axes[1, col]
        bins = "fd" if values.size >= 10 else 10
        hist_ax.hist(values, bins=bins, color="#2B6CB0", alpha=0.85, edgecolor="white")
        title = str(row["display_name"])
        if row.get("lambda") is not None:
            title += f"\nlambda={float(row['lambda']):.3f}"
        hist_ax.set_title(title)
        hist_ax.set_xlabel("Value")
        hist_ax.set_ylabel("Count")
        hist_ax.grid(alpha=0.2)

        sorted_values = np.sort(values)
        y = np.arange(1, sorted_values.size + 1) / sorted_values.size
        ecdf_ax.step(sorted_values, y, where="post", color="#2F855A", linewidth=1.5)
        ecdf_ax.set_xlabel("Value")
        ecdf_ax.set_ylabel("ECDF")
        ecdf_ax.grid(alpha=0.2)

    fig.suptitle(f"{target_name}: raw vs transformed train distributions", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_histogram(values: np.ndarray, title: str, path: Path, xlim: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bins = "fd" if values.size >= 10 else 10
    ax.hist(values, bins=bins, color="#2B6CB0", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Target value")
    ax.set_ylabel("Count")
    if xlim is not None and np.isfinite(xlim[0]) and np.isfinite(xlim[1]) and xlim[0] < xlim[1]:
        ax.set_xlim(*xlim)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_boxplot(split_values: dict[str, np.ndarray], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    labels = []
    values = []
    for split_name in ("train", "val", "test"):
        if split_values[split_name].size:
            labels.append(split_name)
            values.append(split_values[split_name])
    ax.boxplot(values, tick_labels=labels, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel("Target value")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_ecdf(split_values: dict[str, np.ndarray], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = {"train": "#2B6CB0", "val": "#DD6B20", "test": "#2F855A"}
    for split_name in ("train", "val", "test"):
        values = np.sort(split_values[split_name])
        if values.size == 0:
            continue
        y = np.arange(1, values.size + 1) / values.size
        ax.step(values, y, where="post", label=split_name, color=colors[split_name], linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Target value")
    ax.set_ylabel("ECDF")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def relative_figure_path(output_doc: Path, figure_path: Path) -> str:
    return os.path.relpath(figure_path, output_doc.parent).replace(os.sep, "/")


def main() -> None:
    args = parse_args()
    bundle = np.load(args.target_path, allow_pickle=True)
    feature_targets = np.asarray(bundle["feature_targets"], dtype=np.float32)
    feature_mask = bundle["feature_mask"].astype(bool)
    patient_ids = bundle["anchor_patient_ids"].astype(str)

    splits = json.loads(args.splits_path.read_text())
    split_masks = patient_split_masks(patient_ids, splits)
    target_names = load_target_names(args.metadata_path)

    selected = [
        (idx, name)
        for idx, name in enumerate(target_names)
        if name.endswith("_t_plus_0m_gap")
    ]
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {
        "date": args.date,
        "target_path": str(args.target_path),
        "splits_path": str(args.splits_path),
        "metadata_path": str(args.metadata_path),
        "n_total_anchors": int(feature_targets.shape[0]),
        "n_targets": len(selected),
        "targets": [],
    }

    summary_rows: list[dict[str, str]] = []
    transform_summary_rows: list[dict[str, str]] = []
    target_sections: list[str] = []
    transform_fig_dir = args.figure_dir / "transformations"
    transform_fig_dir.mkdir(parents=True, exist_ok=True)

    for col_idx, target_key in selected:
        base_name = target_key.removesuffix("_t_plus_0m_gap")
        units = TARGET_UNITS.get(base_name, "unknown")
        valid_mask = feature_mask[:, col_idx] & np.isfinite(feature_targets[:, col_idx])
        split_values = {
            split_name: feature_targets[split_mask & valid_mask, col_idx].astype(np.float64, copy=False)
            for split_name, split_mask in split_masks.items()
        }
        split_counts = {
            split_name: {
                "n_total": int(split_mask.sum()),
                "n_valid": int((split_mask & valid_mask).sum()),
            }
            for split_name, split_mask in split_masks.items()
        }
        for split_name in split_counts:
            split_counts[split_name]["n_missing"] = split_counts[split_name]["n_total"] - split_counts[split_name]["n_valid"]
            split_counts[split_name]["missing_fraction"] = (
                float(split_counts[split_name]["n_missing"] / split_counts[split_name]["n_total"])
                if split_counts[split_name]["n_total"] > 0
                else 0.0
            )

        train_values = split_values["train"]
        train_stats = summarize_distribution(train_values)
        val_stats = summarize_distribution(split_values["val"])
        test_stats = summarize_distribution(split_values["test"])

        q25 = train_stats["p25"]
        q75 = train_stats["p75"]
        iqr = train_stats["iqr"]
        if q25 is None or q75 is None or iqr is None:
            lower_fence = None
            upper_fence = None
            train_outliers = {"n_outliers": 0, "frac_outliers": 0.0, "n_low_outliers": 0, "n_high_outliers": 0}
            outlier_by_split = {name: train_outliers for name in ("train", "val", "test")}
            concentration = {"n_outliers": 0, "n_patients": 0, "top1_share": 0.0, "top5_share": 0.0}
        else:
            lower_fence = float(q25 - 1.5 * float(iqr))
            upper_fence = float(q75 + 1.5 * float(iqr))
            outlier_by_split = {
                split_name: outlier_summary(values, lower_fence, upper_fence)
                for split_name, values in split_values.items()
            }
            train_outlier_mask = (split_masks["train"] & valid_mask)
            train_outlier_mask = train_outlier_mask & (
                (feature_targets[:, col_idx] < lower_fence) | (feature_targets[:, col_idx] > upper_fence)
            )
            concentration = patient_concentration(patient_ids[split_masks["train"] & valid_mask], (train_values < lower_fence) | (train_values > upper_fence))

        multimodal = multimodality_heuristic(train_values)
        shape_flags = build_shape_flags(base_name, train_stats, multimodal, concentration, outlier_by_split["train"])
        val_shift = assess_shift(train_values, split_values["val"], train_stats, val_stats)
        test_shift = assess_shift(train_values, split_values["test"], train_stats, test_stats)
        if val_shift["label"] == "possible meaningful shift" or test_shift["label"] == "possible meaningful shift":
            shift_summary = "possible meaningful shift"
        elif val_shift["label"] == "mild shift" or test_shift["label"] == "mild shift":
            shift_summary = "mild shift"
        else:
            shift_summary = "no clear shift"
        transform_note = transformation_flag(base_name, train_stats, shape_flags)
        interpretation = build_target_interpretation(
            base_name,
            units,
            train_stats,
            shape_flags,
            shift_summary,
            transform_note,
        )

        transform_candidates: list[dict[str, object]] = []
        z_values, z_params = zscore_transform(train_values)
        transform_candidates.append(
            {
                "transformation": "raw",
                "display_name": "Raw",
                "valid": True,
                "reason": "reference distribution",
                "lambda": None,
                "params": {},
                "values": train_values,
                "after_stats": summarize_distribution(train_values),
            }
        )
        transform_candidates.append(
            {
                "transformation": "zscore",
                "display_name": "Z-score",
                "valid": True,
                "reason": "always valid when train std > 0",
                "lambda": None,
                "params": z_params,
                "values": z_values,
                "after_stats": summarize_distribution(z_values),
            }
        )

        if units != "correlation coefficient" and float(train_stats["min"]) >= 0.0 and float(train_stats["skewness"]) >= 0.5:
            log_values = np.log1p(train_values)
            transform_candidates.append(
                {
                    "transformation": "log1p",
                    "display_name": "log1p",
                    "valid": True,
                    "reason": "non-negative and right-skewed",
                    "lambda": None,
                    "params": {},
                    "values": log_values,
                    "after_stats": summarize_distribution(log_values),
                }
            )
        else:
            transform_candidates.append(
                {
                    "transformation": "log1p",
                    "display_name": "log1p",
                    "valid": False,
                    "reason": "skipped because the target is not both non-negative and clearly right-skewed, or is bounded/correlation-like",
                    "lambda": None,
                    "params": {},
                    "values": None,
                    "after_stats": None,
                }
            )

        if units != "correlation coefficient" and float(train_stats["min"]) > 0.0 and (
            float(train_stats["skewness"]) >= 0.5 or float(train_stats["kurtosis"]) >= 2.0
        ):
            boxcox_values, boxcox_params = fit_boxcox(train_values)
            transform_candidates.append(
                {
                    "transformation": "boxcox",
                    "display_name": "Box-Cox",
                    "valid": True,
                    "reason": "strictly positive with appreciable skew/tail load",
                    "lambda": boxcox_params["lambda"],
                    "params": boxcox_params,
                    "values": boxcox_values,
                    "after_stats": summarize_distribution(boxcox_values),
                }
            )
        else:
            transform_candidates.append(
                {
                    "transformation": "boxcox",
                    "display_name": "Box-Cox",
                    "valid": False,
                    "reason": "skipped because the target is not strictly positive with enough skew/tail load to justify Box-Cox",
                    "lambda": None,
                    "params": {},
                    "values": None,
                    "after_stats": None,
                }
            )

        yj_values, yj_params = fit_yeojohnson(train_values)
        transform_candidates.append(
            {
                "transformation": "yeo_johnson",
                "display_name": "Yeo-Johnson",
                "valid": True,
                "reason": "flexible train-only power transform",
                "lambda": yj_params["lambda"],
                "params": yj_params,
                "values": yj_values,
                "after_stats": summarize_distribution(yj_values),
            }
        )

        for row in transform_candidates:
            row["interpretation"] = transformation_interpretation(
                base_name,
                units,
                str(row["transformation"]),
                None if row["lambda"] is None else float(row["lambda"]),
                train_stats,
                row["after_stats"],
            )

        recommendation = recommendation_for_target(base_name, units, train_stats, transform_candidates)

        slug = slugify(base_name)
        full_hist_path = args.figure_dir / f"{slug}_hist_full.png"
        central_hist_path = args.figure_dir / f"{slug}_hist_central.png"
        boxplot_path = args.figure_dir / f"{slug}_boxplot.png"
        ecdf_path = args.figure_dir / f"{slug}_ecdf.png"
        transform_compare_path = transform_fig_dir / f"{slug}_transform_compare.png"

        render_histogram(train_values, f"{base_name} train histogram", full_hist_path)
        xlim = None
        if train_stats["p01"] is not None and train_stats["p99"] is not None:
            xlim = (float(train_stats["p01"]), float(train_stats["p99"]))
        render_histogram(train_values, f"{base_name} train histogram (1st-99th pct)", central_hist_path, xlim=xlim)
        render_boxplot(split_values, f"{base_name} split comparison boxplot", boxplot_path)
        render_ecdf(split_values, f"{base_name} split comparison ECDF", ecdf_path)
        render_transform_comparison(
            base_name,
            [row for row in transform_candidates if row["valid"]],
            transform_compare_path,
        )

        result_entry = {
            "target": base_name,
            "target_key": target_key,
            "units": units,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "test_stats": test_stats,
            "split_counts": split_counts,
            "train_outlier_fences": {"lower": lower_fence, "upper": upper_fence},
            "outlier_by_split_using_train_fences": outlier_by_split,
            "train_outlier_patient_concentration": concentration,
            "shape_flags": shape_flags,
            "multimodality_heuristic": multimodal,
            "shift_vs_train": {"val": val_shift, "test": test_shift, "summary": shift_summary},
            "transform_note": transform_note,
            "interpretation": interpretation,
            "transformations": [
                {
                    "transformation": row["transformation"],
                    "display_name": row["display_name"],
                    "valid": row["valid"],
                    "reason": row["reason"],
                    "lambda": row["lambda"],
                    "params": row["params"],
                    "after_stats": row["after_stats"],
                    "interpretation": row["interpretation"],
                }
                for row in transform_candidates
            ],
            "recommended_future_experiment": recommendation,
            "figures": {
                "histogram": relative_figure_path(args.output_doc, full_hist_path),
                "central_histogram": relative_figure_path(args.output_doc, central_hist_path),
                "boxplot": relative_figure_path(args.output_doc, boxplot_path),
                "ecdf": relative_figure_path(args.output_doc, ecdf_path),
                "transform_comparison": relative_figure_path(args.output_doc, transform_compare_path),
            },
        }
        results["targets"].append(result_entry)

        summary_rows.append(
            {
                "Target": base_name,
                "N": str(train_stats["n_valid"]),
                "Mean": format_float(train_stats["mean"]),
                "Median": format_float(train_stats["median"]),
                "Std": format_float(train_stats["std"]),
                "Skewness": format_float(train_stats["skewness"]),
                "MinMax": f"{format_float(train_stats['min'])} - {format_float(train_stats['max'])}",
                "ZeroPct": format_percent(float(train_stats["frac_zero"])),
                "Distribution": "; ".join(shape_flags[:3]),
                "Transform": transform_note,
            }
        )
        for row in transform_candidates:
            after = row["after_stats"]
            transform_summary_rows.append(
                {
                    "Target": base_name,
                    "Transformation": str(row["display_name"]),
                    "Valid": "yes" if row["valid"] else "no",
                    "Lambda": format_float(float(row["lambda"])) if row["lambda"] is not None else "—",
                    "SkewBefore": format_float(train_stats["skewness"]),
                    "SkewAfter": format_float(after["skewness"]) if after is not None else "—",
                    "KurtAfter": format_float(after["kurtosis"]) if after is not None else "—",
                    "Interpretation": str(row["interpretation"] if row["valid"] else row["reason"]),
                }
            )

        transform_table_lines = [
            "### Transformation Analysis",
            "",
            f"- recommendation for future experiments: `{recommendation}`",
            "",
            "| Transformation | Valid? | Lambda | Mean | Median | Std | Skewness Before | Skewness After | Kurtosis After | Upper-tail ratio After | Min | Max | P01 | P50 | P99 | Interpretation |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in transform_candidates:
            after = row["after_stats"]
            if row["valid"] and after is not None:
                transform_table_lines.append(
                    f"| `{row['display_name']}` | yes | {format_float(float(row['lambda'])) if row['lambda'] is not None else '—'} | "
                    f"{format_float(after['mean'])} | {format_float(after['median'])} | {format_float(after['std'])} | "
                    f"{format_float(train_stats['skewness'])} | {format_float(after['skewness'])} | {format_float(after['kurtosis'])} | "
                    f"{format_float(upper_tail_ratio(after))} | "
                    f"{format_float(after['min'])} | {format_float(after['max'])} | {format_float(after['p01'])} | "
                    f"{format_float(after['p50'])} | {format_float(after['p99'])} | {row['interpretation']} |"
                )
            else:
                transform_table_lines.append(
                    f"| `{row['display_name']}` | no | — | — | — | — | {format_float(train_stats['skewness'])} | — | — | — | — | — | — | — | — | {row['reason']} |"
                )
        transform_table_lines.extend(
            [
                "",
                "### Transformation Comparison Plot",
                "",
                f"![{base_name} transformation comparison]({relative_figure_path(args.output_doc, transform_compare_path)})",
                "",
            ]
        )

        target_sections.append(
            "\n".join(
                [
                    f"## `{base_name}`",
                    "",
                    f"- target key: `{target_key}`",
                    f"- units: `{units}`",
                    f"- train distribution summary: {'; '.join(shape_flags)}",
                    f"- split-shift summary: `{shift_summary}`",
                    f"- transformation follow-up: {transform_note}",
                    "",
                    "### Valid / Missing Counts",
                    "",
                    "| Split | Total anchors | Valid | Missing | Missing % |",
                    "|---|---:|---:|---:|---:|",
                    f"| train | {split_counts['train']['n_total']} | {split_counts['train']['n_valid']} | {split_counts['train']['n_missing']} | {format_percent(split_counts['train']['missing_fraction'])} |",
                    f"| val | {split_counts['val']['n_total']} | {split_counts['val']['n_valid']} | {split_counts['val']['n_missing']} | {format_percent(split_counts['val']['missing_fraction'])} |",
                    f"| test | {split_counts['test']['n_total']} | {split_counts['test']['n_valid']} | {split_counts['test']['n_missing']} | {format_percent(split_counts['test']['missing_fraction'])} |",
                    "",
                    "### Train Statistics",
                    "",
                    "| Statistic | Value |",
                    "|---|---:|",
                    f"| min | {format_float(train_stats['min'])} |",
                    f"| max | {format_float(train_stats['max'])} |",
                    f"| mean | {format_float(train_stats['mean'])} |",
                    f"| median | {format_float(train_stats['median'])} |",
                    f"| std | {format_float(train_stats['std'])} |",
                    f"| IQR | {format_float(train_stats['iqr'])} |",
                    f"| skewness | {format_float(train_stats['skewness'])} |",
                    f"| kurtosis (excess) | {format_float(train_stats['kurtosis'])} |",
                    "",
                    "### Train Percentiles",
                    "",
                    "| 1st | 5th | 25th | 50th | 75th | 95th | 99th |",
                    "|---:|---:|---:|---:|---:|---:|---:|",
                    f"| {format_float(train_stats['p01'])} | {format_float(train_stats['p05'])} | {format_float(train_stats['p25'])} | {format_float(train_stats['p50'])} | {format_float(train_stats['p75'])} | {format_float(train_stats['p95'])} | {format_float(train_stats['p99'])} |",
                    "",
                    "### Train Zero / Negative / Outlier Counts",
                    "",
                    "| Metric | Count | Percent of valid train values |",
                    "|---|---:|---:|",
                    f"| exact zeros | {train_stats['n_zero']} | {format_percent(float(train_stats['frac_zero']))} |",
                    f"| negative values | {train_stats['n_negative']} | {format_percent(float(train_stats['frac_negative']))} |",
                    f"| Tukey outliers using train 1.5*IQR fences | {outlier_by_split['train']['n_outliers']} | {format_percent(float(outlier_by_split['train']['frac_outliers']))} |",
                    f"| low outliers | {outlier_by_split['train']['n_low_outliers']} | {format_percent(float(outlier_by_split['train']['n_low_outliers']) / max(int(train_stats['n_valid']), 1))} |",
                    f"| high outliers | {outlier_by_split['train']['n_high_outliers']} | {format_percent(float(outlier_by_split['train']['n_high_outliers']) / max(int(train_stats['n_valid']), 1))} |",
                    "",
                    f"Train outlier fences: lower `{format_float(lower_fence)}`, upper `{format_float(upper_fence)}`.",
                    "",
                    "### Split Comparison",
                    "",
                    "| Split | N valid | Mean | Median | Std | IQR | % zero | % negative | % outliers vs train fences | KS vs train | Median shift / train IQR | Assessment |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                    f"| train | {train_stats['n_valid']} | {format_float(train_stats['mean'])} | {format_float(train_stats['median'])} | {format_float(train_stats['std'])} | {format_float(train_stats['iqr'])} | {format_percent(float(train_stats['frac_zero']))} | {format_percent(float(train_stats['frac_negative']))} | {format_percent(float(outlier_by_split['train']['frac_outliers']))} | {format_float(0.0)} | {format_float(0.0)} | reference |",
                    f"| val | {val_stats['n_valid']} | {format_float(val_stats['mean'])} | {format_float(val_stats['median'])} | {format_float(val_stats['std'])} | {format_float(val_stats['iqr'])} | {format_percent(float(val_stats['frac_zero']))} | {format_percent(float(val_stats['frac_negative']))} | {format_percent(float(outlier_by_split['val']['frac_outliers']))} | {format_float(val_shift['ks'])} | {format_float(val_shift['median_shift_iqr'])} | {val_shift['label']} |",
                    f"| test | {test_stats['n_valid']} | {format_float(test_stats['mean'])} | {format_float(test_stats['median'])} | {format_float(test_stats['std'])} | {format_float(test_stats['iqr'])} | {format_percent(float(test_stats['frac_zero']))} | {format_percent(float(test_stats['frac_negative']))} | {format_percent(float(outlier_by_split['test']['frac_outliers']))} | {format_float(test_shift['ks'])} | {format_float(test_shift['median_shift_iqr'])} | {test_shift['label']} |",
                    "",
                    "### Extreme Values By Patient",
                    "",
                    f"- train Tukey outliers: `{concentration['n_outliers']}` across `{concentration['n_patients']}` patients",
                    f"- largest single-patient share of train outliers: `{format_percent(float(concentration['top1_share']))}`",
                    f"- top-5-patient share of train outliers: `{format_percent(float(concentration['top5_share']))}`",
                    "",
                    "### Plots",
                    "",
                    f"![{base_name} full histogram]({relative_figure_path(args.output_doc, full_hist_path)})",
                    "",
                    f"![{base_name} central histogram]({relative_figure_path(args.output_doc, central_hist_path)})",
                    "",
                    f"![{base_name} split boxplot]({relative_figure_path(args.output_doc, boxplot_path)})",
                    "",
                    f"![{base_name} ECDF]({relative_figure_path(args.output_doc, ecdf_path)})",
                    "",
                    "### Short Interpretation",
                    "",
                    f"- {interpretation}",
                    "",
                    *transform_table_lines,
                ]
            )
        )

    summary_lines = [
        "# Target Distribution Analysis",
        "",
        f"Computed on `{args.date}` for the same vasopressor-free regression setup used in `docs/v1_vasopressor_free/regression_results_v1_vaso_free_sorted.md`.",
        "",
        "## Scope",
        "",
        "- cohort: vasopressor-free overlap cohort",
        f"- split file: `{args.splits_path}`",
        f"- target bundle: `{args.target_path}`",
        "- analyzed targets: the `26` leakage-safe `*_t_plus_0m_gap` regression targets evaluated in the regression results doc",
        "- detailed descriptive statistics below use the `train` split, because future normalization or transformation decisions should be fit on training data only",
        "- `val` and `test` are included only to check for distribution shift",
        "",
        "## Methods / Conventions",
        "",
        "- Missing counts are split totals minus finite valid target values from the saved `feature_mask` and target array.",
        "- Obvious outliers are defined as Tukey 1.5*IQR outliers using fences fit on the train split for each target.",
        "- `kurtosis` is reported as excess kurtosis.",
        "- The central-range histogram uses the train 1st-99th percentile x-range.",
        "- Multimodality is a heuristic flag based on smoothed train histograms; treat it as suggestive, not definitive.",
        "- Units for waveform morphology and interaction targets are inferred from feature names because the saved bundle does not store unit metadata explicitly.",
        "- Transformation parameters (`z` mean/std, Box-Cox lambda, Yeo-Johnson lambda) are fit on the train split only.",
        "- Transformed descriptive statistics are reported on the transformed train targets only; val/test are not used to fit or tune transformations.",
        "",
        "## Summary Table",
        "",
        "| Target | N | Mean | Median | Std | Skewness | Min-Max | % Zero | Distribution summary | Transform follow-up |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for row in summary_rows:
        summary_lines.append(
            f"| `{row['Target']}` | {row['N']} | {row['Mean']} | {row['Median']} | {row['Std']} | {row['Skewness']} | {row['MinMax']} | {row['ZeroPct']} | {row['Distribution']} | {row['Transform']} |"
        )

    summary_lines.extend(
        [
            "",
            "## Transformation Analysis",
            "",
            "| Target | Transformation | Valid? | Lambda | Skewness Before | Skewness After | Kurtosis After | Interpretation |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in transform_summary_rows:
        summary_lines.append(
            f"| `{row['Target']}` | `{row['Transformation']}` | {row['Valid']} | {row['Lambda']} | {row['SkewBefore']} | {row['SkewAfter']} | {row['KurtAfter']} | {row['Interpretation']} |"
        )

    summary_lines.extend(
        [
            "",
            "## Recommended Future Transformation Tests",
            "",
            "| Target | Recommendation |",
            "|---|---|",
            *[
                f"| `{target['target']}` | {target['recommended_future_experiment']} |"
                for target in results["targets"]
            ],
            "",
            "## Per-Target Details",
            "",
            *target_sections,
            "## Overall Interpretation",
            "",
            "### Per-Target Interpretation",
            "",
            *[f"- {target['interpretation']}" for target in results["targets"]],
            "",
            "### Recommended Future Tests",
            "",
            *[f"- `{target['target']}`: {target['recommended_future_experiment']}" for target in results["targets"]],
            "",
            "- Targets with strong positive skew or heavy upper tails are the clearest candidates for future nonlinear transform checks.",
            "- Targets with negative support are poor Box-Cox candidates and, if transformation is revisited later, are better matched to Yeo-Johnson or simple z-scoring.",
            "- The bounded correlation targets deserve separate review because they already live on a constrained `[-1, 1]` scale with some boundary mass.",
            "- Split comparisons should be read as a guardrail against learning a train-specific transform that does not transfer cleanly to validation or test.",
            "",
        ]
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_doc.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    args.output_doc.write_text("\n".join(summary_lines))

    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote doc:  {args.output_doc}")
    print(f"Figures:    {args.figure_dir}")


if __name__ == "__main__":
    main()
