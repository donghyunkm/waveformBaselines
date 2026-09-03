#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_patchtst import NumpyWaveformDataset, SingleTargetDataset, TargetExtractor, TrainConfig

ANCHOR_TIME_DECIMALS = 6
TARGET_SENSITIVITY = 0.85


@dataclass
class PredictionSet:
    model: str
    sample_ids: np.ndarray
    patient_ids: np.ndarray
    anchor_times: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


def event_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "spec85": float(specificity_at_sensitivity(labels, scores)),
    }


def specificity_at_sensitivity(labels: np.ndarray, scores: np.ndarray, target_sensitivity: float = TARGET_SENSITIVITY) -> float:
    labels = labels.astype(np.int64)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    tp = np.cumsum(sorted_labels)[distinct_ends]
    fp = (distinct_ends + 1) - tp
    positives = max(int(labels.sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    sens = tp / positives
    spec = (negatives - fp) / negatives
    valid = sens >= target_sensitivity
    if not np.any(valid):
        return -1.0
    return float(np.max(spec[valid]))


def patchtst_ids(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = TrainConfig(
        task="event",
        event_name=args.event_name,
        horizon=args.horizon,
        target_path=args.target_path,
        splits_path=args.splits_path,
        waveform_dir=args.waveform_dir,
        run_tag="",
        channels=tuple(args.channels.split(",")),
        n_channels=len(args.channels.split(",")),
        seq_len=args.seq_len,
    )
    target_extractor = TargetExtractor(cfg, Path(args.target_path))
    numpy_ds = NumpyWaveformDataset(
        split="test",
        waveform_dir=Path(args.waveform_dir),
        splits_path=Path(args.splits_path),
        normalize=True,
        channels=tuple(args.channels.split(",")),
        seq_len=args.seq_len,
    )
    ds = SingleTargetDataset(numpy_ds, target_extractor)
    sample_ids = []
    patient_ids = []
    anchor_times = []
    for orig_idx in ds._valid_indices:
        pid, anchor_center = numpy_ds._windows[orig_idx]
        anchor_time = numpy_ds._patient_seg_start[pid] + anchor_center / float(numpy_ds.fs)
        patient_ids.append(str(pid))
        anchor_times.append(float(anchor_time))
        sample_ids.append(f"{pid}|{round(float(anchor_time), ANCHOR_TIME_DECIMALS):.6f}")
    return np.asarray(sample_ids), np.asarray(patient_ids), np.asarray(anchor_times, dtype=np.float64)


def load_patchtst(path: Path, args: argparse.Namespace) -> PredictionSet:
    data = np.load(path, allow_pickle=True)
    valid = data["masks"].astype(bool) & np.isfinite(data["predictions"]) & np.isfinite(data["targets"])
    sample_ids, patient_ids, anchor_times = patchtst_ids(args)
    if valid.sum() != sample_ids.size:
        raise ValueError(f"PatchTST valid prediction count {valid.sum()} does not match reconstructed IDs {sample_ids.size}")
    return PredictionSet(
        model="PatchTST v1 raw waveform",
        sample_ids=sample_ids,
        patient_ids=patient_ids,
        anchor_times=anchor_times,
        labels=data["targets"][valid].astype(int),
        scores=data["predictions"][valid].astype(float),
    )


def load_v7(model: str, path: Path) -> PredictionSet:
    data = np.load(path, allow_pickle=True)
    labels = data["targets"].astype(int)
    scores = data["predictions"].astype(float)
    patient_ids = data["patient_ids"].astype(str)
    anchor_times = data["anchor_times"].astype(np.float64)
    sample_ids = np.asarray([f"{pid}|{round(float(t), ANCHOR_TIME_DECIMALS):.6f}" for pid, t in zip(patient_ids, anchor_times)])
    return PredictionSet(model=model, sample_ids=sample_ids, patient_ids=patient_ids, anchor_times=anchor_times, labels=labels, scores=scores)


def align(base: PredictionSet, comp: PredictionSet) -> tuple[PredictionSet, PredictionSet]:
    base_index = {sid: i for i, sid in enumerate(base.sample_ids)}
    comp_index = {sid: i for i, sid in enumerate(comp.sample_ids)}
    if len(base_index) != len(base.sample_ids):
        raise ValueError("PatchTST sample IDs are not unique")
    if len(comp_index) != len(comp.sample_ids):
        raise ValueError(f"{comp.model} sample IDs are not unique")
    if set(base_index) != set(comp_index):
        missing = sorted(set(base_index) - set(comp_index))[:5]
        extra = sorted(set(comp_index) - set(base_index))[:5]
        raise ValueError(f"Sample ID mismatch for {comp.model}: missing={missing} extra={extra}")
    order = np.asarray([comp_index[sid] for sid in base.sample_ids], dtype=np.int64)
    comp_aligned = PredictionSet(
        model=comp.model,
        sample_ids=comp.sample_ids[order],
        patient_ids=comp.patient_ids[order],
        anchor_times=comp.anchor_times[order],
        labels=comp.labels[order],
        scores=comp.scores[order],
    )
    if not np.array_equal(base.sample_ids, comp_aligned.sample_ids):
        raise ValueError(f"Sample ID ordering mismatch after alignment for {comp.model}")
    if not np.array_equal(base.labels, comp_aligned.labels):
        raise ValueError(f"Label mismatch after alignment for {comp.model}")
    return base, comp_aligned


def grouped_indices(patient_ids: np.ndarray) -> list[np.ndarray]:
    groups = []
    for pid in np.unique(patient_ids):
        groups.append(np.flatnonzero(patient_ids == pid))
    return groups


def metric_delta(labels: np.ndarray, base_scores: np.ndarray, comp_scores: np.ndarray, metric: str) -> float:
    if metric == "auroc":
        return float(roc_auc_score(labels, comp_scores) - roc_auc_score(labels, base_scores))
    if metric == "auprc":
        return float(average_precision_score(labels, comp_scores) - average_precision_score(labels, base_scores))
    if metric == "spec85":
        return float(specificity_at_sensitivity(labels, comp_scores) - specificity_at_sensitivity(labels, base_scores))
    raise ValueError(metric)



def all_metric_deltas(labels: np.ndarray, base_scores: np.ndarray, comp_scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, comp_scores) - roc_auc_score(labels, base_scores)),
        "auprc": float(average_precision_score(labels, comp_scores) - average_precision_score(labels, base_scores)),
        "spec85": float(specificity_at_sensitivity(labels, comp_scores) - specificity_at_sensitivity(labels, base_scores)),
    }


def bootstrap_and_permutation_all(labels, base_scores, comp_scores, patient_ids, n_boot, n_perm, seed):
    rng = np.random.default_rng(seed)
    groups = grouped_indices(patient_ids)
    clustered = len(groups) < labels.size
    observed = all_metric_deltas(labels, base_scores, comp_scores)

    boot = {"auroc": [], "auprc": [], "spec85": []}
    skipped_boot = 0
    for _ in range(n_boot):
        if clustered:
            picks = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[i] for i in picks])
        else:
            idx = rng.integers(0, labels.size, size=labels.size)
        if np.unique(labels[idx]).size < 2:
            skipped_boot += 1
            continue
        deltas = all_metric_deltas(labels[idx], base_scores[idx], comp_scores[idx])
        for metric, delta in deltas.items():
            boot[metric].append(delta)

    perm_extreme = {metric: 0 for metric in observed}
    for _ in range(n_perm):
        b = base_scores.copy()
        c = comp_scores.copy()
        if clustered:
            swaps = rng.random(len(groups)) < 0.5
            for do_swap, idx in zip(swaps, groups):
                if do_swap:
                    b[idx], c[idx] = c[idx].copy(), b[idx].copy()
        else:
            swaps = rng.random(labels.size) < 0.5
            b[swaps], c[swaps] = c[swaps].copy(), b[swaps].copy()
        deltas = all_metric_deltas(labels, b, c)
        for metric, delta in deltas.items():
            if abs(delta) >= abs(observed[metric]):
                perm_extreme[metric] += 1

    out = {}
    for metric, observed_delta in observed.items():
        boot_values = np.asarray(boot[metric], dtype=float)
        ci = np.percentile(boot_values, [2.5, 97.5]) if boot_values.size else np.asarray([math.nan, math.nan])
        out[metric] = {
            "delta": float(observed_delta),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "p_permutation": float((1 + perm_extreme[metric]) / (n_perm + 1)),
            "bootstrap_successful": int(boot_values.size),
            "bootstrap_skipped_one_class": int(skipped_boot),
            "permutations": int(n_perm),
            "permutation_skipped": 0,
            "clustered_by_patient": bool(clustered),
            "n_patients": int(len(groups)),
        }
    return out


def bootstrap_and_permutation(labels, base_scores, comp_scores, patient_ids, metric, n_boot, n_perm, seed):
    rng = np.random.default_rng(seed)
    groups = grouped_indices(patient_ids)
    clustered = len(groups) < labels.size
    boot = []
    skipped_boot = 0
    for _ in range(n_boot):
        if clustered:
            picks = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[i] for i in picks])
        else:
            idx = rng.integers(0, labels.size, size=labels.size)
        if np.unique(labels[idx]).size < 2:
            skipped_boot += 1
            continue
        boot.append(metric_delta(labels[idx], base_scores[idx], comp_scores[idx], metric))
    boot = np.asarray(boot, dtype=float)
    ci = np.percentile(boot, [2.5, 97.5]) if boot.size else np.asarray([math.nan, math.nan])

    observed = metric_delta(labels, base_scores, comp_scores, metric)
    perm_extreme = 0
    skipped_perm = 0
    for _ in range(n_perm):
        b = base_scores.copy()
        c = comp_scores.copy()
        if clustered:
            swaps = rng.random(len(groups)) < 0.5
            for do_swap, idx in zip(swaps, groups):
                if do_swap:
                    b[idx], c[idx] = c[idx].copy(), b[idx].copy()
        else:
            swaps = rng.random(labels.size) < 0.5
            b[swaps], c[swaps] = c[swaps].copy(), b[swaps].copy()
        if np.unique(labels).size < 2:
            skipped_perm += 1
            continue
        delta = metric_delta(labels, b, c, metric)
        if abs(delta) >= abs(observed):
            perm_extreme += 1
    p = (1 + perm_extreme) / (n_perm + 1)
    return {
        "delta": observed,
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "p_permutation": float(p),
        "bootstrap_successful": int(boot.size),
        "bootstrap_skipped_one_class": int(skipped_boot),
        "permutations": int(n_perm),
        "permutation_skipped": int(skipped_perm),
        "clustered_by_patient": bool(clustered),
        "n_patients": int(len(groups)),
    }


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks


def delong_cov(labels: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(bool)
    pos = preds[:, labels]
    neg = preds[:, ~labels]
    m = pos.shape[1]
    n = neg.shape[1]
    tx = np.vstack([compute_midrank(row) for row in pos])
    ty = np.vstack([compute_midrank(row) for row in neg])
    tz = np.vstack([compute_midrank(row) for row in preds])
    aucs = (tz[:, labels].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, labels] - tx) / n
    v10 = 1 - (tz[:, ~labels] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    cov = np.atleast_2d(cov)
    return aucs, cov


def delong_p_ci(labels: np.ndarray, base_scores: np.ndarray, comp_scores: np.ndarray) -> dict[str, float]:
    aucs, cov = delong_cov(labels, np.vstack([base_scores, comp_scores]))
    diff = float(aucs[1] - aucs[0])
    contrast = np.asarray([-1.0, 1.0])
    var = float(contrast @ cov @ contrast.T)
    se = math.sqrt(max(var, 0.0))
    if se == 0:
        p = 1.0 if diff == 0 else 0.0
    else:
        z = abs(diff) / se
        p = math.erfc(z / math.sqrt(2.0))
    return {"p_delong": float(max(p, 0.0)), "ci_lower": diff - 1.96 * se, "ci_upper": diff + 1.96 * se, "se": se}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running = max(running, adj)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def fmt(x: float, digits: int = 4, signed: bool = False) -> str:
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def fmt_p(p: float) -> str:
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patchtst-predictions", default="outputs/patchtst/vasopressor_free_v1_events_5m_10m_anchor_horizon_filtered_es/event_hypotension_within_5m/test_predictions.npz")
    ap.add_argument("--v7-root", default="/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/classification")
    ap.add_argument("--target-path", default="outputs/targets/event_targets_vasopressor_free_anchor_horizon_filtered_5m_10m.npz")
    ap.add_argument("--splits-path", default="outputs/splits/vasopressor_free_splits.json")
    ap.add_argument("--waveform-dir", default="/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms")
    ap.add_argument("--event-name", default="hypotension")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--channels", default="ABP,II,PLETH")
    ap.add_argument("--seq-len", type=int, default=150000)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--out-json", default="outputs/feature_models/classification_patchtst_v7_significance_2026-08-29.json")
    ap.add_argument("--out-csv", default="outputs/feature_models/classification_patchtst_v7_significance_2026-08-29.csv")
    ap.add_argument("--out-md", default="outputs/feature_models/classification_patchtst_v7_significance_2026-08-29.md")
    ap.add_argument("--models", default="current_state_xgb,history_xgb,full_sequence_xgb,gru,transformer")
    args = ap.parse_args()

    model_catalog = {
        "current_state_xgb": ("Current-state XGBoost", "current_state_xgb_hypotension_within_5m_filtered_v7/test_predictions.npz"),
        "history_xgb": ("History XGBoost", "history_xgb_hypotension_within_5m_filtered_v7/test_predictions.npz"),
        "full_sequence_xgb": ("Full-sequence XGBoost", "full_sequence_xgb_hypotension_within_5m_filtered_v7/test_predictions.npz"),
        "full_sequence_mlp": ("Full-sequence MLP", "full_sequence_mlp_hypotension_within_5m_filtered_v7/test_predictions.npz"),
        "gru": ("GRU", "gru_hypotension_within_5m_filtered_v7/test_predictions.npz"),
        "tcn": ("TCN", "tcn_hypotension_within_5m_filtered_v7/test_predictions.npz"),
        "transformer": ("Transformer", "transformer_hypotension_within_5m_filtered_v7/test_predictions.npz"),
    }
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in requested_models if m not in model_catalog]
    if unknown:
        raise ValueError(f"Unknown models requested: {unknown}; choices={sorted(model_catalog)}")
    models = [model_catalog[m] for m in requested_models]
    missing = [str(Path(args.v7_root) / rel) for _, rel in models if not (Path(args.v7_root) / rel).exists()]
    if missing:
        raise FileNotFoundError("Missing v7 prediction files:\n" + "\n".join(missing))

    base = load_patchtst(Path(args.patchtst_predictions), args)
    base_metrics = event_metrics(base.labels, base.scores)
    results = []
    validation = {
        "patchtst_n": int(base.labels.size),
        "patchtst_positives": int(base.labels.sum()),
        "patchtst_metrics": base_metrics,
        "target_sensitivity": TARGET_SENSITIVITY,
    }
    for i, (name, rel) in enumerate(models):
        comp = load_v7(name, Path(args.v7_root) / rel)
        b, c = align(base, comp)
        metrics = event_metrics(c.labels, c.scores)
        row = {
            "model": name,
            "n": int(c.labels.size),
            "positives": int(c.labels.sum()),
            "metrics": metrics,
            "validation": {
                "sample_ids_identical": bool(np.array_equal(b.sample_ids, c.sample_ids)),
                "labels_identical": bool(np.array_equal(b.labels, c.labels)),
                "n_expected_3650": int(c.labels.size) == 3650,
                "positives_expected_194": int(c.labels.sum()) == 194,
                "n_patients": int(np.unique(c.patient_ids).size),
                "repeated_observations_per_patient": bool(np.unique(c.patient_ids).size < c.labels.size),
            },
            "tests": {},
        }
        seed = args.seed + i * 100
        delong = delong_p_ci(c.labels, b.scores, c.scores)
        row["tests"] = bootstrap_and_permutation_all(c.labels, b.scores, c.scores, c.patient_ids, args.n_bootstrap, args.n_permutations, seed + 1)
        row["tests"]["auroc"].update({
            "p_delong": delong["p_delong"],
            "delong_ci_lower": delong["ci_lower"],
            "delong_ci_upper": delong["ci_upper"],
            "primary_inference": "patient_clustered_permutation",
            "secondary_p_delong": delong["p_delong"],
        })
        results.append(row)

    for metric in ["auroc", "auprc", "spec85"]:
        adj = holm([r["tests"][metric]["p_permutation"] for r in results])
        for r, p_adj in zip(results, adj):
            r["tests"][metric]["p_holm"] = p_adj

    payload = {"metadata": vars(args), "validation": validation, "results": results}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n")

    headers = ["Model","N","Pos","AUROC","Delta AUROC","Delta AUROC 95% CI","AUROC p","AUROC p adj","AUPRC","Delta AUPRC","Delta AUPRC 95% CI","AUPRC p","AUPRC p adj","Spec @85%","Delta Spec","Delta Spec 95% CI","Spec p","Spec p adj"]
    rows = []
    rows.append(["PatchTST v1 raw waveform", base.labels.size, int(base.labels.sum()), fmt(base_metrics["auroc"]), "-", "-", "-", "-", fmt(base_metrics["auprc"]), "-", "-", "-", "-", fmt(base_metrics["spec85"]), "-", "-", "-", "-"])
    for r in results:
        t = r["tests"]
        m = r["metrics"]
        rows.append([
            r["model"], r["n"], r["positives"], fmt(m["auroc"]), fmt(t["auroc"]["delta"], signed=True), f"[{fmt(t['auroc']['ci_lower'], signed=True)}, {fmt(t['auroc']['ci_upper'], signed=True)}]", fmt_p(t["auroc"]["p_permutation"]), fmt_p(t["auroc"]["p_holm"]),
            fmt(m["auprc"]), fmt(t["auprc"]["delta"], signed=True), f"[{fmt(t['auprc']['ci_lower'], signed=True)}, {fmt(t['auprc']['ci_upper'], signed=True)}]", fmt_p(t["auprc"]["p_permutation"]), fmt_p(t["auprc"]["p_holm"]),
            fmt(m["spec85"]), fmt(t["spec85"]["delta"], signed=True), f"[{fmt(t['spec85']['ci_lower'], signed=True)}, {fmt(t['spec85']['ci_upper'], signed=True)}]", fmt_p(t["spec85"]["p_permutation"]), fmt_p(t["spec85"]["p_holm"]),
        ])
    csv_lines = [",".join(headers)]
    for row in rows:
        csv_lines.append(",".join('"'+str(x).replace('"','""')+'"' for x in row))
    Path(args.out_csv).write_text("\n".join(csv_lines) + "\n")
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] + ["---:"]*(len(headers)-1)) + " |"]
    for row in rows:
        md.append("| " + " | ".join(map(str, row)) + " |")
    Path(args.out_md).write_text("\n".join(md) + "\n")
    print(Path(args.out_md).read_text())


if __name__ == "__main__":
    main()
