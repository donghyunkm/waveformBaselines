#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.extract_full_data_vasofree_waveform_features import SegmentReader, extract_window  # noqa: E402
from waveform_baselines.wf_features.cache import load_feature_cache  # noqa: E402
from waveform_baselines.wf_features_v8.cache import feature_stats, load_v8_feature_cache, subset_feature_cache_by_anchor_ids, validate_v7_v8_alignment  # noqa: E402
from waveform_baselines.wf_features_v8.config import DEFAULT_V8_EXTRACTION_CONFIG  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v8 waveform feature cache distributions and v7 alignment.")
    parser.add_argument("--v8-cache", type=Path, required=True)
    parser.add_argument("--v7-cache", type=Path, default=None)
    parser.add_argument("--allow-prefix-v7-alignment", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--generate-overlays", action="store_true")
    parser.add_argument("--overlay-dir", type=Path, default=None)
    parser.add_argument("--max-overlays", type=int, default=16)
    return parser.parse_args()


def _flatten_valid(values: np.ndarray, mask: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values[:, :, idx], dtype=np.float64)
    m = np.asarray(mask[:, :, idx], dtype=bool) & np.isfinite(v)
    return v, m


def correlation_flags(values: np.ndarray, mask: np.ndarray, names: list[str], threshold: float = 0.95) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    for i in range(len(names)):
        xi, mi = _flatten_valid(values, mask, i)
        for j in range(i + 1, len(names)):
            xj, mj = _flatten_valid(values, mask, j)
            ok = mi & mj
            if int(np.sum(ok)) < 5:
                continue
            a = xi[ok]
            b = xj[ok]
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            if abs(corr) >= threshold:
                flags.append({"feature_a": names[i], "feature_b": names[j], "correlation": corr, "common_count": int(np.sum(ok))})
    return flags


def cross_cache_correlation_flags(v7, v8, threshold: float = 0.95) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    for i, v8_name in enumerate(v8.feature_names):
        xi, mi = _flatten_valid(np.asarray(v8.values), np.asarray(v8.mask), i)
        for j, v7_name in enumerate(v7.feature_names):
            xj, mj = _flatten_valid(np.asarray(v7.values), np.asarray(v7.mask), j)
            ok = mi & mj
            if int(np.sum(ok)) < 5:
                continue
            a = xi[ok]
            b = xj[ok]
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            if abs(corr) >= threshold:
                flags.append({"v8_feature": v8_name, "v7_feature": v7_name, "correlation": corr, "common_count": int(np.sum(ok))})
    return flags


def generate_overlays(v8, output_dir: Path, max_overlays: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    anchors_path = v8.cache_dir / "anchors.csv"
    if not anchors_path.exists():
        return []
    anchors = pd.read_csv(anchors_path)
    if anchors.empty:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_V8_EXTRACTION_CONFIG
    reader = SegmentReader(config.channel_order)
    priority = [
        "abp_ppv_pct",
        "abp_spv_mmhg",
        "pleth_resp_amplitude_variation_pct",
        "hrv_sampen_5m",
        "hrv_dfa_alpha1_5m",
        "hrv_lf_hf_ratio_5m",
        "baroreflex_gain_5m_ms_per_mmhg",
        "resp_pause_count",
        "resp_rr_max_abs_correlation",
    ]
    paths: list[str] = []
    seen: set[tuple[int, int, str]] = set()
    for feature in priority:
        if feature not in v8.feature_names or len(paths) >= max_overlays:
            continue
        feat_idx = v8.feature_names.index(feature)
        valid = np.asarray(v8.mask[:, :, feat_idx], dtype=bool) & np.isfinite(v8.values[:, :, feat_idx])
        candidates: list[tuple[str, int, int]] = []
        if valid.any():
            arr = np.asarray(v8.values[:, :, feat_idx])
            vals = arr[valid]
            q05, q50, q95 = np.percentile(vals, [5.0, 50.0, 95.0])
            for label, target in (("low", q05), ("median", q50), ("high", q95)):
                dist = np.where(valid, np.abs(arr - target), np.inf)
                sample_idx, minute_idx = np.unravel_index(int(np.argmin(dist)), dist.shape)
                candidates.append((label, int(sample_idx), int(minute_idx)))
        missing = np.argwhere(~valid)
        if missing.size:
            candidates.append(("missing", int(missing[0, 0]), int(missing[0, 1])))
        for label, sample_idx, minute_idx in candidates:
            key = (sample_idx, minute_idx, f"{feature}_{label}")
            if key in seen or len(paths) >= max_overlays:
                continue
            seen.add(key)
            row = anchors.iloc[sample_idx]
            waveform = extract_window(row, reader, config.input_samples, config.sampling_rate_hz)
            start = minute_idx * config.feature_window_samples
            end = start + config.feature_window_samples
            minute = waveform[:, start:end]
            t = np.arange(minute.shape[1], dtype=float) / config.sampling_rate_hz
            fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
            for ax, channel, values in zip(axes, config.channel_order, minute):
                ax.plot(t, values, linewidth=0.8)
                ax.set_ylabel(channel)
            value = v8.values[sample_idx, minute_idx, feat_idx]
            axes[0].set_title(f"{feature} {label}: sample={sample_idx}, minute={minute_idx}, value={value}")
            axes[-1].set_xlabel("seconds within token")
            fig.tight_layout()
            safe = f"{feature}_{label}_s{sample_idx}_m{minute_idx}.png"
            out = output_dir / safe
            fig.savefig(out, dpi=120)
            plt.close(fig)
            paths.append(str(out))
    return paths


def main() -> None:
    args = parse_args()
    v8 = load_v8_feature_cache(args.v8_cache, require_success=False)
    report = {
        "cache_dir": str(args.v8_cache),
        "shape": list(v8.values.shape),
        "feature_count": len(v8.feature_names),
        "feature_quality": feature_stats(np.asarray(v8.values), np.asarray(v8.mask), v8.feature_names),
        "disabled_features": [
            name
            for name, enabled in v8.metadata.get("feature_enabled_by_default", {}).items()
            if not bool(enabled)
        ],
    }
    report["near_zero_variance_features"] = [
        name
        for name, stats_row in report["feature_quality"].items()
        if stats_row.get("count", 0) > 0 and (stats_row.get("unique_finite_count", 0) <= 1 or stats_row.get("std", 0.0) < 1e-8)
    ]
    report["high_v8_correlations_abs_ge_0_95"] = correlation_flags(np.asarray(v8.values), np.asarray(v8.mask), v8.feature_names)
    if args.v7_cache is not None:
        v7 = load_feature_cache(args.v7_cache, require_success=True)
        if v8.metadata.get("shard_index") is not None:
            v7 = subset_feature_cache_by_anchor_ids(v7, v8.anchor_ids)
        elif args.allow_prefix_v7_alignment and v8.values.shape[0] < v7.values.shape[0]:
            from waveform_baselines.wf_features.cache import FeatureCache

            v7 = FeatureCache(
                values=v7.values[: v8.values.shape[0]],
                mask=v7.mask[: v8.values.shape[0]],
                patient_ids=v7.patient_ids[: v8.values.shape[0]],
                anchor_times=v7.anchor_times[: v8.values.shape[0]],
                anchor_ids=v7.anchor_ids[: v8.values.shape[0]],
                split_labels=v7.split_labels[: v8.values.shape[0]],
                feature_names=v7.feature_names,
                metadata=v7.metadata,
                cache_dir=v7.cache_dir,
                segment_ids=v7.segment_ids[: v8.values.shape[0]] if v7.segment_ids is not None else None,
                segment_names=v7.segment_names[: v8.values.shape[0]] if v7.segment_names is not None else None,
            )
        report["v7_alignment"] = validate_v7_v8_alignment(v7, v8)
        report["high_v7_v8_correlations_abs_ge_0_95"] = cross_cache_correlation_flags(v7, v8)
    if args.generate_overlays:
        overlay_dir = args.overlay_dir or (args.v8_cache / "overlays")
        report["overlay_paths"] = generate_overlays(v8, overlay_dir, args.max_overlays)
    out = args.output_json or (args.v8_cache / "feature_quality_audit.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"output_json": str(out), "shape": report["shape"], "feature_count": report["feature_count"]}, indent=2))


if __name__ == "__main__":
    main()
