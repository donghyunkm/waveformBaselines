#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_patchtst import NumpyWaveformDataset, SingleTargetDataset, TargetExtractor, TrainConfig

ANCHOR_TIME_DECIMALS = 6
FEATURES = [
    "HR", "RR", "SBP", "DBP", "PP", "MAP", "ABP_area", "PLETH_ACDC", "PLETH_amp", "ECG_Ramp",
    "HRV_RMSSD", "HR_range", "ShockIdx", "PPV", "PVI", "PTT", "dPdt_max", "ABP_tau", "RESP_amp",
    "PLETH_ACDC_PLETH_amp", "ABP_area_ABP_tau", "ABP_area_ShockIdx", "PLETH_amp_ShockIdx",
    "PLETH_ACDC_ShockIdx", "ShockIdx_ABP_tau", "PLETH_ACDC_ABP_tau",
]
SUPPORTED_MODELS = ["persistence", "current_state_xgb", "history_xgb", "full_sequence_xgb", "full_sequence_mlp", "gru", "tcn", "transformer"]
DEFAULT_MODELS = ["persistence", "current_state_xgb", "history_xgb", "full_sequence_xgb", "full_sequence_mlp", "gru", "transformer"]


@dataclass
class PredictionSet:
    sample_ids: np.ndarray
    patient_ids: np.ndarray
    anchor_times: np.ndarray
    y: np.ndarray
    pred: np.ndarray


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
    }


def metric_delta(y: np.ndarray, base: np.ndarray, comp: np.ndarray, metric: str) -> float:
    bm = metrics(y, base)[metric]
    cm = metrics(y, comp)[metric]
    return float(cm - bm)


def grouped_indices(patient_ids: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(patient_ids == pid) for pid in np.unique(patient_ids)]


def metric_from_sums(metric: str, n: np.ndarray, sum_y: np.ndarray, sum_y2: np.ndarray, sse: np.ndarray, sae: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    if metric == "rmse":
        return np.sqrt(sse / n)
    if metric == "mae":
        return sae / n
    if metric == "r2":
        sst = sum_y2 - (sum_y * sum_y / n)
        return 1.0 - sse / sst
    raise ValueError(metric)


def grouped_regression_stats(y: np.ndarray, base: np.ndarray, comp: np.ndarray, patient_ids: np.ndarray) -> dict[str, np.ndarray]:
    groups = grouped_indices(patient_ids)
    n = np.empty(len(groups), dtype=np.float64)
    sum_y = np.empty(len(groups), dtype=np.float64)
    sum_y2 = np.empty(len(groups), dtype=np.float64)
    sse_base = np.empty(len(groups), dtype=np.float64)
    sse_comp = np.empty(len(groups), dtype=np.float64)
    sae_base = np.empty(len(groups), dtype=np.float64)
    sae_comp = np.empty(len(groups), dtype=np.float64)
    for i, idx in enumerate(groups):
        yy = y[idx]
        n[i] = yy.size
        sum_y[i] = yy.sum()
        sum_y2[i] = np.square(yy).sum()
        sse_base[i] = np.square(base[idx] - yy).sum()
        sse_comp[i] = np.square(comp[idx] - yy).sum()
        sae_base[i] = np.abs(base[idx] - yy).sum()
        sae_comp[i] = np.abs(comp[idx] - yy).sum()
    return {
        "n": n,
        "sum_y": sum_y,
        "sum_y2": sum_y2,
        "sse_base": sse_base,
        "sse_comp": sse_comp,
        "sae_base": sae_base,
        "sae_comp": sae_comp,
        "n_groups": np.asarray([len(groups)]),
    }


def delta_from_sums(metric: str, n: np.ndarray, sum_y: np.ndarray, sum_y2: np.ndarray, sse_base: np.ndarray, sse_comp: np.ndarray, sae_base: np.ndarray, sae_comp: np.ndarray) -> np.ndarray:
    base_metric = metric_from_sums(metric, n, sum_y, sum_y2, sse_base, sae_base)
    comp_metric = metric_from_sums(metric, n, sum_y, sum_y2, sse_comp, sae_comp)
    return comp_metric - base_metric


def bootstrap_perm(y, base, comp, patient_ids, metric, n_boot, n_perm, seed):
    rng = np.random.default_rng(seed)
    stats = grouped_regression_stats(y, base, comp, patient_ids)
    n_groups = int(stats["n_groups"][0])
    clustered = n_groups < y.size

    observed = float(delta_from_sums(
        metric,
        stats["n"].sum(),
        stats["sum_y"].sum(),
        stats["sum_y2"].sum(),
        stats["sse_base"].sum(),
        stats["sse_comp"].sum(),
        stats["sae_base"].sum(),
        stats["sae_comp"].sum(),
    ))

    if clustered:
        counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_boot).astype(np.float64)
    else:
        counts = rng.multinomial(y.size, np.full(y.size, 1.0 / y.size), size=n_boot).astype(np.float64)
    boot_n = counts @ stats["n"]
    boot_sum_y = counts @ stats["sum_y"]
    boot_sum_y2 = counts @ stats["sum_y2"]
    boot_sse_base = counts @ stats["sse_base"]
    boot_sse_comp = counts @ stats["sse_comp"]
    boot_sae_base = counts @ stats["sae_base"]
    boot_sae_comp = counts @ stats["sae_comp"]
    if metric == "r2":
        valid = (boot_sum_y2 - (boot_sum_y * boot_sum_y / boot_n)) > 0
    else:
        valid = np.ones(n_boot, dtype=bool)
    boot = delta_from_sums(
        metric,
        boot_n[valid],
        boot_sum_y[valid],
        boot_sum_y2[valid],
        boot_sse_base[valid],
        boot_sse_comp[valid],
        boot_sae_base[valid],
        boot_sae_comp[valid],
    )
    ci = np.percentile(boot, [2.5, 97.5]) if boot.size else [math.nan, math.nan]

    swaps = rng.random((n_perm, n_groups)) < 0.5
    n_total = stats["n"].sum()
    sum_y_total = stats["sum_y"].sum()
    sum_y2_total = stats["sum_y2"].sum()
    sse_base_total = stats["sse_base"].sum()
    sse_comp_total = stats["sse_comp"].sum()
    sae_base_total = stats["sae_base"].sum()
    sae_comp_total = stats["sae_comp"].sum()
    perm_sse_base = sse_base_total + swaps @ (stats["sse_comp"] - stats["sse_base"])
    perm_sse_comp = sse_comp_total + swaps @ (stats["sse_base"] - stats["sse_comp"])
    perm_sae_base = sae_base_total + swaps @ (stats["sae_comp"] - stats["sae_base"])
    perm_sae_comp = sae_comp_total + swaps @ (stats["sae_base"] - stats["sae_comp"])
    perm_delta = delta_from_sums(
        metric,
        np.full(n_perm, n_total),
        np.full(n_perm, sum_y_total),
        np.full(n_perm, sum_y2_total),
        perm_sse_base,
        perm_sse_comp,
        perm_sae_base,
        perm_sae_comp,
    )
    extreme = int((np.abs(perm_delta) >= abs(observed)).sum())
    return {
        "delta": observed,
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "p": float((1 + extreme) / (n_perm + 1)),
        "bootstrap_successful": int(boot.size),
        "bootstrap_skipped_invalid": int(n_boot - boot.size),
        "clustered_by_patient": bool(clustered),
        "n_patients": int(n_groups),
    }

def patchtst_ids(feature, args):
    cfg = TrainConfig(
        task="feature", feature_name=feature, horizon=0, feature_horizon_mode="gap",
        target_path=args.target_path, splits_path=args.splits_path, waveform_dir=args.waveform_dir,
        run_tag="", channels=tuple(args.channels.split(",")), n_channels=len(args.channels.split(",")), seq_len=args.seq_len,
    )
    extractor = TargetExtractor(cfg, Path(args.target_path))
    numpy_ds = NumpyWaveformDataset(split="test", waveform_dir=Path(args.waveform_dir), splits_path=Path(args.splits_path), normalize=True, channels=tuple(args.channels.split(",")), seq_len=args.seq_len)
    ds = SingleTargetDataset(numpy_ds, extractor)
    sample_ids=[]; pids=[]; times=[]
    for orig_idx in ds._valid_indices:
        pid, anchor_center = numpy_ds._windows[orig_idx]
        t = numpy_ds._patient_seg_start[pid] + anchor_center / float(numpy_ds.fs)
        sample_ids.append(f"{pid}|{round(float(t), ANCHOR_TIME_DECIMALS):.6f}")
        pids.append(str(pid)); times.append(float(t))
    return np.asarray(sample_ids), np.asarray(pids), np.asarray(times)


def load_patchtst(feature, args):
    p = Path(args.patchtst_root) / f"feature_{feature}_t_plus_0m_gap" / "test_predictions.npz"
    d = np.load(p, allow_pickle=True)
    valid = d["masks"].astype(bool) & np.isfinite(d["predictions"]) & np.isfinite(d["targets"])
    ids, pids, times = patchtst_ids(feature, args)
    if valid.sum() != ids.size:
        raise ValueError(f"{feature}: PatchTST valid count {valid.sum()} != reconstructed IDs {ids.size}")
    return PredictionSet(ids, pids, times, d["targets"][valid].astype(float), d["predictions"][valid].astype(float))


def load_v7(path: Path):
    d = np.load(path, allow_pickle=True)
    pids = d["patient_ids"].astype(str)
    times = d["anchor_times"].astype(float)
    ids = np.asarray([f"{pid}|{round(float(t), ANCHOR_TIME_DECIMALS):.6f}" for pid,t in zip(pids,times)])
    return PredictionSet(ids, pids, times, d["targets"].astype(float), d["predictions"].astype(float))


def align(base, comp, label):
    bi = {sid:i for i,sid in enumerate(base.sample_ids)}
    ci = {sid:i for i,sid in enumerate(comp.sample_ids)}
    if len(bi) != len(base.sample_ids) or len(ci) != len(comp.sample_ids):
        raise ValueError(f"{label}: duplicate sample IDs")
    if set(bi) != set(ci):
        raise ValueError(f"{label}: sample ID mismatch missing={len(set(bi)-set(ci))} extra={len(set(ci)-set(bi))}")
    order = np.asarray([ci[sid] for sid in base.sample_ids])
    comp_y = comp.y[order]
    if not np.allclose(base.y, comp_y, rtol=0, atol=1e-6):
        raise ValueError(f"{label}: target values differ after alignment")
    return base.y, base.pred, comp.pred[order], base.patient_ids


def fmt(x, signed=False):
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def fmt_p(p):
    return "<0.0001" if p < 0.0001 else f"{p:.4g}"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v7-root", default="/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels/regression")
    ap.add_argument("--patchtst-root", default="outputs/patchtst/vasopressor_free_v1_es")
    ap.add_argument("--target-path", default="outputs/targets/feature_targets_gap_vasopressor_free.npz")
    ap.add_argument("--splits-path", default="outputs/splits/vasopressor_free_splits.json")
    ap.add_argument("--waveform-dir", default="/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms")
    ap.add_argument("--channels", default="ABP,II,PLETH")
    ap.add_argument("--seq-len", type=int, default=150000)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--out-json", default="outputs/feature_models/regression_patchtst_v7_significance_2026-08-29.json")
    ap.add_argument("--out-csv", default="outputs/feature_models/regression_patchtst_v7_significance_2026-08-29.csv")
    ap.add_argument("--out-md", default="outputs/feature_models/regression_patchtst_v7_significance_2026-08-29.md")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--features", default=",".join(FEATURES))
    args=ap.parse_args()
    requested_models=[m.strip() for m in args.models.split(",") if m.strip()]
    requested_features=[f.strip() for f in args.features.split(",") if f.strip()]
    unknown_models=[m for m in requested_models if m not in SUPPORTED_MODELS]
    unknown_features=[f for f in requested_features if f not in FEATURES]
    if unknown_models:
        raise ValueError(f"Unknown models requested: {unknown_models}; choices={SUPPORTED_MODELS}")
    if unknown_features:
        raise ValueError(f"Unknown features requested: {unknown_features}; choices={FEATURES}")
    results=[]; failures=[]
    for ti, feature in enumerate(requested_features):
        try:
            base = load_patchtst(feature, args)
        except Exception as e:
            failures.append({"target": feature, "model": "PatchTST", "error": str(e)})
            continue
        base_m = metrics(base.y, base.pred)
        for mi, model in enumerate(requested_models):
            pred_path = Path(args.v7_root)/f"{model}_feature_{feature}_t_plus_0m_gap_v7"/"test_predictions.npz"
            try:
                comp = load_v7(pred_path)
                y, bpred, cpred, pids = align(base, comp, f"{feature}/{model}")
                comp_m = metrics(y, cpred)
                tests={}
                for k, off in [("r2",1),("rmse",2),("mae",3)]:
                    tests[k] = bootstrap_perm(y,bpred,cpred,pids,k,args.n_bootstrap,args.n_permutations,args.seed + ti*1000 + mi*10 + off)
                results.append({"target":feature,"model":model,"n":int(y.size),"n_patients":int(np.unique(pids).size),"patchtst":base_m,"comparison":comp_m,"tests":tests,"prediction_file":str(pred_path)})
            except Exception as e:
                failures.append({"target":feature,"model":model,"prediction_file":str(pred_path),"error":str(e)})
    payload={"metadata":vars(args),"features":requested_features,"models":requested_models,"results":results,"failures":failures}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2)+"\n")
    headers=["Target","Model","N","Patients","PatchTST R2","v7 R2","Delta R2","Delta R2 95% CI","R2 p","PatchTST RMSE","v7 RMSE","Delta RMSE","Delta RMSE 95% CI","RMSE p","PatchTST MAE","v7 MAE","Delta MAE","Delta MAE 95% CI","MAE p"]
    with Path(args.out_csv).open('w', newline='') as f:
        w=csv.writer(f); w.writerow(headers)
        for r in results:
            t=r['tests']; w.writerow([r['target'],r['model'],r['n'],r['n_patients'],r['patchtst']['r2'],r['comparison']['r2'],t['r2']['delta'],f"[{t['r2']['ci_lower']}, {t['r2']['ci_upper']}]",t['r2']['p'],r['patchtst']['rmse'],r['comparison']['rmse'],t['rmse']['delta'],f"[{t['rmse']['ci_lower']}, {t['rmse']['ci_upper']}]",t['rmse']['p'],r['patchtst']['mae'],r['comparison']['mae'],t['mae']['delta'],f"[{t['mae']['ci_lower']}, {t['mae']['ci_upper']}]",t['mae']['p']])
    md=["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]+["---:"]*(len(headers)-1))+" |"]
    for r in results:
        t=r['tests']
        md.append("| "+" | ".join(map(str,[r['target'],r['model'],r['n'],r['n_patients'],fmt(r['patchtst']['r2']),fmt(r['comparison']['r2']),fmt(t['r2']['delta'], True),f"[{fmt(t['r2']['ci_lower'], True)}, {fmt(t['r2']['ci_upper'], True)}]",fmt_p(t['r2']['p']),fmt(r['patchtst']['rmse']),fmt(r['comparison']['rmse']),fmt(t['rmse']['delta'], True),f"[{fmt(t['rmse']['ci_lower'], True)}, {fmt(t['rmse']['ci_upper'], True)}]",fmt_p(t['rmse']['p']),fmt(r['patchtst']['mae']),fmt(r['comparison']['mae']),fmt(t['mae']['delta'], True),f"[{fmt(t['mae']['ci_lower'], True)}, {fmt(t['mae']['ci_upper'], True)}]",fmt_p(t['mae']['p'])]))+" |")
    Path(args.out_md).write_text("\n".join(md)+"\n")
    print(f"results={len(results)} failures={len(failures)}")
    print(args.out_md)

if __name__ == '__main__':
    main()
