from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegressionBootstrapResult:
    mae: float
    mae_ci_lower: float
    mae_ci_upper: float
    rmse: float
    rmse_ci_lower: float
    rmse_ci_upper: float
    r2: float
    r2_ci_lower: float
    r2_ci_upper: float
    n_bootstrap: int
    seed: int | None
    n_test_predictions: int
    n_test_patients: int
    r2_valid_bootstrap_replicates: int
    r2_invalid_bootstrap_replicates: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def regression_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"regression metrics require 1D arrays, got {y_true.shape}")
    if y_true.size == 0:
        raise ValueError("cannot compute regression metrics on zero rows")
    err = y_true - y_pred
    ss_res = float(np.sum(err * err))
    y_centered = y_true - float(np.mean(y_true))
    ss_tot = float(np.sum(y_centered * y_centered))
    if ss_tot <= 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "r2": float(r2),
    }


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray, patient_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    patient_ids = np.asarray(patient_ids).astype(str)
    if y_true.ndim != 1 or y_pred.ndim != 1 or patient_ids.ndim != 1:
        raise ValueError("y_true, y_pred, and patient_ids must be 1D arrays")
    if not (y_true.shape[0] == y_pred.shape[0] == patient_ids.shape[0]):
        raise ValueError(
            "y_true, y_pred, and patient_ids length mismatch: "
            f"{y_true.shape[0]}, {y_pred.shape[0]}, {patient_ids.shape[0]}"
        )
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite.all():
        y_true = y_true[finite]
        y_pred = y_pred[finite]
        patient_ids = patient_ids[finite]
    if y_true.size == 0:
        raise ValueError("no finite prediction rows available for regression bootstrap")
    if np.any(patient_ids == ""):
        raise ValueError("patient_ids must not contain empty strings")
    return y_true, y_pred, patient_ids


def _patient_sufficient_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    unique_patients, inverse = np.unique(patient_ids, return_inverse=True)
    err = y_true - y_pred
    stats = {
        "n": np.bincount(inverse).astype(np.float64),
        "abs_err": np.bincount(inverse, weights=np.abs(err)).astype(np.float64),
        "sq_err": np.bincount(inverse, weights=err * err).astype(np.float64),
        "y_sum": np.bincount(inverse, weights=y_true).astype(np.float64),
        "y_sq_sum": np.bincount(inverse, weights=y_true * y_true).astype(np.float64),
    }
    return unique_patients, stats


def _bootstrap_patient_counts(
    n_patients: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    sampled_patient_indices: np.ndarray | None = None,
) -> np.ndarray:
    if sampled_patient_indices is None:
        sampled_patient_indices = rng.integers(0, n_patients, size=(n_bootstrap, n_patients))
    else:
        sampled_patient_indices = np.asarray(sampled_patient_indices, dtype=np.int64)
        if sampled_patient_indices.ndim != 2:
            raise ValueError("sampled_patient_indices must be a 2D array")
        if sampled_patient_indices.shape[1] != n_patients:
            raise ValueError(
                "sampled_patient_indices must sample the same number of patients as the original test set: "
                f"expected {n_patients}, got {sampled_patient_indices.shape[1]}"
            )
        if np.any(sampled_patient_indices < 0) or np.any(sampled_patient_indices >= n_patients):
            raise ValueError("sampled_patient_indices contains out-of-range patient index")
        n_bootstrap = int(sampled_patient_indices.shape[0])
    counts = np.zeros((n_bootstrap, n_patients), dtype=np.float64)
    for row_idx, sampled in enumerate(sampled_patient_indices):
        counts[row_idx] = np.bincount(sampled, minlength=n_patients)
    return counts


def bootstrap_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    seed: int | None = 42,
    sampled_patient_indices: np.ndarray | None = None,
    return_bootstrap_values: bool = False,
) -> RegressionBootstrapResult | tuple[RegressionBootstrapResult, dict[str, np.ndarray]]:
    y_true, y_pred, patient_ids = _validate_inputs(y_true, y_pred, patient_ids)
    unique_patients, stats = _patient_sufficient_statistics(y_true, y_pred, patient_ids)
    n_patients = int(unique_patients.shape[0])
    if n_patients == 0:
        raise ValueError("cannot bootstrap without patients")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")
    rng = np.random.default_rng(seed)
    counts = _bootstrap_patient_counts(n_patients, n_bootstrap, rng, sampled_patient_indices)
    n_bootstrap = int(counts.shape[0])

    boot_n = counts @ stats["n"]
    boot_abs = counts @ stats["abs_err"]
    boot_sq = counts @ stats["sq_err"]
    boot_y_sum = counts @ stats["y_sum"]
    boot_y_sq_sum = counts @ stats["y_sq_sum"]

    boot_mae = boot_abs / boot_n
    boot_rmse = np.sqrt(boot_sq / boot_n)
    boot_ss_tot = boot_y_sq_sum - (boot_y_sum * boot_y_sum) / boot_n
    boot_r2 = np.full(n_bootstrap, np.nan, dtype=np.float64)
    valid_r2 = boot_ss_tot > np.finfo(np.float64).eps
    boot_r2[valid_r2] = 1.0 - boot_sq[valid_r2] / boot_ss_tot[valid_r2]

    point = regression_point_metrics(y_true, y_pred)
    r2_valid = int(np.isfinite(boot_r2).sum())
    r2_invalid = int(n_bootstrap - r2_valid)
    if r2_valid == 0:
        r2_lower = np.nan
        r2_upper = np.nan
    else:
        r2_lower, r2_upper = np.nanpercentile(boot_r2, [2.5, 97.5])
    result = RegressionBootstrapResult(
        mae=point["mae"],
        mae_ci_lower=float(np.percentile(boot_mae, 2.5)),
        mae_ci_upper=float(np.percentile(boot_mae, 97.5)),
        rmse=point["rmse"],
        rmse_ci_lower=float(np.percentile(boot_rmse, 2.5)),
        rmse_ci_upper=float(np.percentile(boot_rmse, 97.5)),
        r2=point["r2"],
        r2_ci_lower=float(r2_lower),
        r2_ci_upper=float(r2_upper),
        n_bootstrap=n_bootstrap,
        seed=seed,
        n_test_predictions=int(y_true.shape[0]),
        n_test_patients=n_patients,
        r2_valid_bootstrap_replicates=r2_valid,
        r2_invalid_bootstrap_replicates=r2_invalid,
    )
    if return_bootstrap_values:
        return result, {"mae": boot_mae, "rmse": boot_rmse, "r2": boot_r2, "patient_counts": counts}
    return result
