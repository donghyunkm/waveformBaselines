"""
PatchTST baseline training script — optimized I/O pipeline.

Reads pre-extracted per-patient numpy waveforms (from scripts/extract_waveforms.py).
Uses patient-grouped batch sampling for minimal file switching, memory-mapped reads,
and optimized DataLoader settings for maximum GPU utilization.

Predicts a SINGLE target at a time:
  - One of the 26 waveform/correlation features
  - One of the 2 clinical event tasks (hypotension, tachycardia)

Usage:
    python scripts/train_patchtst.py --task feature --feature-name MAP --horizon 0
    python scripts/train_patchtst.py --task event --event-name hypotension --horizon 5
    python scripts/train_patchtst.py --list-targets
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from rotary_embedding_torch import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from waveform_baselines.numpy_dataset import NumpyWaveformDataset
from waveform_baselines.patient_sampler import PatientGroupedSampler
from waveform_baselines.task_specs import DEFAULT_EVENT_TASK, DEFAULT_FEATURE_TASK

# ── Target definitions ────────────────────────────────────────────────────────

WAVEFORM_FEATURE_NAMES = [
    "HR", "RR", "SBP", "DBP", "PP",
    "MAP", "ABP_area", "PLETH_ACDC", "PLETH_amp", "ECG_Ramp",
    "HRV_RMSSD", "HR_range", "ShockIdx", "PPV", "PVI",
    "PTT", "dPdt_max", "ABP_tau", "RESP_amp",
]

CORRELATION_FEATURE_NAMES = [
    "PLETH_ACDC_PLETH_amp", "ABP_area_ABP_tau", "ABP_area_ShockIdx",
    "PLETH_amp_ShockIdx", "PLETH_ACDC_ShockIdx", "ShockIdx_ABP_tau",
    "PLETH_ACDC_ABP_tau",
]

ALL_FEATURE_NAMES = WAVEFORM_FEATURE_NAMES + CORRELATION_FEATURE_NAMES  # 26 total

EVENT_NAMES = ["hypotension", "tachycardia"]

FEATURE_HORIZONS = list(DEFAULT_FEATURE_TASK.horizons_min)   # minutes
EVENT_HORIZONS = list(DEFAULT_EVENT_TASK.horizons_min)    # minutes


def parse_channel_list(channel_text: str) -> tuple[str, ...]:
    channels = tuple(part.strip() for part in channel_text.split(",") if part.strip())
    if not channels:
        raise ValueError("At least one channel must be specified.")
    if len(set(channels)) != len(channels):
        raise ValueError(f"Duplicate channels are not allowed: {channels}")
    return channels


def feature_target_name(feature_name: str, horizon: int, horizon_mode: str) -> str:
    suffix = f"t_plus_{horizon}m"
    if horizon_mode == "gap":
        suffix = f"{suffix}_gap"
    return f"{feature_name}_{suffix}"


PHYSIOJEPA_FIDELITY_PRESET = {
    "patch_len": 125,
    "stride": 125,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 3,
    "d_ff": 2048,
    "dropout": 0.1,
    "attn_dropout": 0.0,
    "qkv_bias": True,
    "pool_depth": 1,
    "pool_mlp_ratio": 4.0,
    "pool_num_queries": 1,
    "pool_complete_block": True,
    "pool_affine": False,
    "batch_size": 32,
}


def _cli_flag_set(argv: list[str] | None = None) -> set[str]:
    argv = sys.argv[1:] if argv is None else argv
    provided = set()
    for token in argv:
        if token.startswith("--"):
            provided.add(token.split("=", 1)[0])
    return provided


def apply_physiojepa_fidelity_preset(args: argparse.Namespace, provided_flags: set[str], parser: argparse.ArgumentParser) -> None:
    if not args.physiojepa_fidelity:
        return
    if args.model_variant != "patchtst_v1_5":
        parser.error("--physiojepa-fidelity requires --model-variant patchtst_v1_5")

    option_names = {
        "patch_len": {"--patch-len"},
        "stride": {"--stride"},
        "d_model": {"--d-model"},
        "n_heads": {"--n-heads"},
        "n_layers": {"--n-layers"},
        "d_ff": {"--d-ff"},
        "dropout": {"--dropout"},
        "attn_dropout": {"--attn-dropout"},
        "qkv_bias": {"--qkv-bias", "--no-qkv-bias"},
        "pool_depth": {"--pool-depth"},
        "pool_mlp_ratio": {"--pool-mlp-ratio"},
        "pool_num_queries": {"--pool-num-queries"},
        "pool_complete_block": {"--pool-complete-block", "--no-pool-complete-block"},
        "pool_affine": {"--pool-affine", "--no-pool-affine"},
        "batch_size": {"--batch-size"},
    }

    for attr, preset_value in PHYSIOJEPA_FIDELITY_PRESET.items():
        explicit = any(flag in provided_flags for flag in option_names[attr])
        current_value = getattr(args, attr)
        if explicit and current_value != preset_value:
            parser.error(
                f"--physiojepa-fidelity conflicts with explicit {sorted(option_names[attr])[0]}={current_value}; "
                f"expected {preset_value}."
            )
        if not explicit:
            setattr(args, attr, preset_value)


def log_v15_architecture(config: TrainConfig, n_patches: int) -> None:
    print("PatchTST v1.5 architecture:", flush=True)
    print(f"  patch_len: {config.patch_len}", flush=True)
    print(f"  stride: {config.stride}", flush=True)
    print(f"  n_patches: {n_patches}", flush=True)
    print(f"  d_model: {config.d_model}", flush=True)
    print(f"  n_heads: {config.n_heads}", flush=True)
    print(f"  n_layers: {config.n_layers}", flush=True)
    print(f"  d_ff: {config.d_ff}", flush=True)
    print(f"  dropout: {config.dropout}", flush=True)
    print(f"  attn_dropout: {config.attn_dropout}", flush=True)
    print(f"  qkv_bias: {config.qkv_bias}", flush=True)
    print(f"  pool_depth: {config.pool_depth}", flush=True)
    print(f"  pool_num_queries: {config.pool_num_queries}", flush=True)
    print(f"  pool_mlp_ratio: {config.pool_mlp_ratio}", flush=True)
    print(f"  pool_complete_block: {config.pool_complete_block}", flush=True)
    print(f"  pool_affine: {config.pool_affine}", flush=True)


def list_all_targets(feature_horizon_mode: str = "center"):
    """Print all available single targets."""
    print(
        f"=== Feature Regression Targets ({len(ALL_FEATURE_NAMES)} features × "
        f"{len(FEATURE_HORIZONS)} horizons = {len(ALL_FEATURE_NAMES) * len(FEATURE_HORIZONS)}) ==="
    )
    for h in FEATURE_HORIZONS:
        for name in ALL_FEATURE_NAMES:
            extra = ""
            if feature_horizon_mode == "gap":
                extra = " --feature-horizon-mode gap"
            print(f"  --task feature --feature-name {name} --horizon {h}{extra}")
    print()
    print(
        f"=== Event Classification Targets ({len(EVENT_NAMES)} events × "
        f"{len(EVENT_HORIZONS)} horizons = {len(EVENT_NAMES) * len(EVENT_HORIZONS)}) ==="
    )
    for h in EVENT_HORIZONS:
        for name in EVENT_NAMES:
            print(f"  --task event --event-name {name} --horizon {h}")


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """Training configuration for single-target PatchTST."""

    # Task
    task: str = "feature"  # "feature" or "event"
    feature_name: str = "MAP"  # which feature to predict
    event_name: str = "hypotension"  # which event to predict
    horizon: int = 0  # prediction horizon in minutes
    feature_horizon_mode: str = "center"
    run_tag: str = ""
    physiojepa_fidelity: bool = False

    # Data
    channels: tuple[str, ...] = ("ABP", "II", "PLETH")
    n_channels: int = 3
    seq_len: int = 150_000  # 20 min at 125 Hz
    fs: int = 125

    # Architecture
    model_variant: str = "patchtst_v1"
    patch_len: int = 250  # 2 seconds
    stride: int = 250  # non-overlapping patches
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    attn_dropout: float = 0.0
    qkv_bias: bool = True
    cross_channel_layers: int = 1
    cross_channel_heads: int = 4
    cross_channel_window: int = 1
    pooling_type: str = "mean"
    pool_depth: int = 1
    pool_mlp_ratio: float = 4.0
    pool_num_queries: int = 1
    pool_complete_block: bool = True
    pool_affine: bool = False

    # Training
    epochs: int = 50
    batch_size: int = 512  # L40S has 48GB — fits up to 1024, 512 leaves headroom
    lr: float = 1e-3  # scaled with sqrt(batch_size/64) from 3e-4 base
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    early_stopping_patience: int = 0
    early_stopping_min_epochs: int = 0
    early_stopping_min_delta: float = 0.0

    # Infrastructure
    num_workers: int = 4
    normalize: bool = True
    seed: int = 42
    output_dir: str = ""  # auto-generated if empty
    log_interval: int = 100

    # Paths
    waveform_dir: str = "/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms"
    splits_path: str = "outputs/splits/splits.json"
    target_path: str = ""  # path to .npz target bundle
    resume: bool = False  # resume from latest checkpoint

    @property
    def target_key(self) -> str:
        if self.task == "feature":
            return feature_target_name(self.feature_name, self.horizon, self.feature_horizon_mode)
        else:
            return f"{self.event_name}_within_{self.horizon}m"

    def resolve_output_dir(self) -> str:
        if self.output_dir:
            return self.output_dir
        parts = ["outputs/patchtst"]
        if self.run_tag:
            parts.append(self.run_tag)
        parts.append(f"{self.task}_{self.target_key}")
        return "/".join(parts)

    def __post_init__(self):
        self.channels = tuple(self.channels)
        if len(self.channels) != self.n_channels:
            raise ValueError(
                f"n_channels={self.n_channels} does not match channels={self.channels}"
            )
        if self.seq_len <= 0 or self.seq_len % 2 != 0:
            raise ValueError(f"seq_len must be a positive even integer, got {self.seq_len}")


# ── Model ─────────────────────────────────────────────────────────────────────


class PatchEmbedding(nn.Module):
    """Convert raw waveform into patches and project to d_model."""

    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> (B, C, n_patches, d_model)"""
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        return self.proj(patches)


class PatchEmbeddingV1(nn.Module):
    """Patch tokenizer for the simple channel-independent PatchTST baseline."""

    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        return self.proj(patches)


class AttentionPooling(nn.Module):
    """Pool a token sequence with a learned query."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(x.size(0), -1, -1)
        pooled, _ = self.attn(query, self.norm(x), self.norm(x), need_weights=False)
        return pooled.squeeze(1)


class LocalCrossChannelFusion(nn.Module):
    """Fuse channel-wise temporal features using only a local channel/time neighborhood."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        window: int,
    ):
        super().__init__()
        self.window = window
        self.fusion_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.offset_embed = nn.Embedding(2 * window + 1, d_model)
        self.pre_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.post_ffn_norm = nn.LayerNorm(d_model)

    def _local_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, D) -> (B*T, C*(2w+1), D)
        B, C, T, D = x.shape
        w = self.window
        padded = nn.functional.pad(x, (0, 0, w, w), mode="replicate")
        local_slices = []
        for offset in range(2 * w + 1):
            local_slices.append(padded[:, :, offset:offset + T, :])
        local = torch.stack(local_slices, dim=3)  # (B, C, T, 2w+1, D)
        offset_ids = torch.arange(2 * w + 1, device=x.device)
        local = local + self.offset_embed(offset_ids)[None, None, None, :, :]
        return local.permute(0, 2, 1, 3, 4).reshape(B * T, C * (2 * w + 1), D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, D = x.shape
        local_tokens = self._local_tokens(x)
        local_tokens = self.pre_norm(local_tokens)
        query = self.fusion_query.expand(B * T, -1, -1)
        fused, _ = self.cross_attn(query, local_tokens, local_tokens, need_weights=False)
        fused = fused + query
        fused = self.post_attn_norm(fused)
        fused = fused + self.ffn(fused)
        fused = self.post_ffn_norm(fused)
        return fused.reshape(B, T, D)


class PhysioJEPAPatchTokenizer(nn.Module):
    """Channel-specific grouped-Conv tokenizer with stride-aware zero end-padding."""

    def __init__(
        self,
        n_channels: int,
        patch_len: int,
        stride: int,
        d_model: int,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.proj = nn.Conv1d(
            in_channels=n_channels,
            out_channels=n_channels * d_model,
            kernel_size=patch_len,
            stride=stride,
            padding=0,
            groups=n_channels,
            bias=True,
        )

    def _pad_amount(self, seq_len: int) -> int:
        if seq_len <= self.patch_len:
            return self.patch_len - seq_len
        remainder = (seq_len - self.patch_len) % self.stride
        return 0 if remainder == 0 else self.stride - remainder

    def num_patches(self, seq_len: int) -> int:
        padded_len = seq_len + self._pad_amount(seq_len)
        return 1 + (padded_len - self.patch_len) // self.stride

    def _pad_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        pad = self._pad_amount(x.size(-1))
        if pad > 0:
            x = F.pad(x, (0, pad), mode="constant", value=0.0)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_if_needed(x)
        x = self.proj(x)
        bsz, _, n_patches = x.shape
        x = x.reshape(bsz, self.n_channels, self.d_model, n_patches)
        return x.permute(0, 1, 3, 2)


class PhysioJEPAMultiHeadAttention(nn.Module):
    """PhysioJEPA-style self-attention with separate Q/K/V and RoPE on Q/K."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        rotary_pes: bool,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"d_model ({dim}) must be divisible by n_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.W_Q = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_K = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_V = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop_prob = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rotary_pes = rotary_pes
        self.rotary_embed = None
        if rotary_pes:
            if RotaryEmbedding is None:
                raise ImportError(
                    "patchtst_v1_5 requires rotary_embedding_torch. "
                    "Install the dependency or use patchtst_v1/patchtst_v2."
                )
            self.rotary_embed = RotaryEmbedding(
                dim=self.head_dim,
                freqs_for="lang",
                theta=10000,
                learned_freq=False,
                seq_before_head_dim=False,
                use_xpos=False,
                cache_max_seq_len=29000,
            )

    def forward(self, x: torch.Tensor, key: torch.Tensor | None = None, value: torch.Tensor | None = None) -> torch.Tensor:
        key = x if key is None else key
        value = x if value is None else value

        q = self.W_Q(x).unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.W_K(key).unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        v = self.W_V(value).unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

        if self.rotary_embed is not None:
            if self.training:
                q = self.rotary_embed.rotate_queries_or_keys(q)
                k = self.rotary_embed.rotate_queries_or_keys(k)
            else:
                q, k = self.rotary_embed.rotate_queries_with_cached_keys(q, k)

        attn_dropout = self.attn_drop_prob if self.training else 0.0
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True):
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=attn_dropout,
                is_causal=False,
                scale=self.scale,
            )

        x = x.transpose(1, 2).flatten(-2)
        x = self.proj(x)
        return self.proj_drop(x)


class PhysioJEPATSTBlock(nn.Module):
    """PhysioJEPA TSTBlock with post-norm residual structure by default."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attn_dropout: float,
        qkv_bias: bool,
        rotary_pes: bool,
        pre_norm: bool = False,
    ):
        super().__init__()
        self.self_attn = PhysioJEPAMultiHeadAttention(
            dim=d_model,
            num_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_dropout,
            proj_drop=dropout,
            rotary_pes=rotary_pes,
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=qkv_bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=qkv_bias),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.pre_norm = pre_norm

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            src = self.norm_attn(src)
        attn_out = self.self_attn(src)
        src = src + self.dropout_attn(attn_out)
        if not self.pre_norm:
            src = self.norm_attn(src)
        if self.pre_norm:
            src = self.norm_ffn(src)
        ff_out = self.ff(src)
        src = src + self.dropout_ffn(ff_out)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        return src


class PhysioJEPACrossAttention(nn.Module):
    """PhysioJEPA attentive-pooler cross-attention without RoPE or attn dropout."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"d_model ({dim}) must be divisible by n_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.W_Q = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_K = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_V = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        q = self.W_Q(q).unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.W_K(x).unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        v = self.W_V(x).unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True):
            q = F.scaled_dot_product_attention(q, k, v)
        q = q.transpose(1, 2).flatten(-2)
        return self.proj(q)


class PhysioJEPACrossAttentionBlock(nn.Module):
    """PhysioJEPA complete attentive pooling block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.xattn = PhysioJEPACrossAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q


class PhysioJEPAAttentivePooler(nn.Module):
    """Shared learned-query pooler used independently for each channel sequence."""

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        init_std: float,
        qkv_bias: bool,
        complete_block: bool,
    ):
        super().__init__()
        if depth != 1:
            raise NotImplementedError('patchtst_v1_5 currently supports pool_depth=1 only')
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = PhysioJEPACrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
            )
        else:
            self.cross_attention_block = PhysioJEPACrossAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
            )
        self.init_std = init_std
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(_init_physiojepa_module)
        self._rescale_blocks()

    def _rescale_blocks(self) -> None:
        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp[2].weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query_tokens.repeat(len(x), 1, 1)
        return self.cross_attention_block(q, x)


def _init_physiojepa_module(module: nn.Module, init_std: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=init_std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Conv1d):
        torch.nn.init.trunc_normal_(module.weight, std=init_std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


class PatchTST(nn.Module):
    """
    PatchTST for single-target prediction from raw waveforms.
    Output is a single scalar (regression) or logit (classification).

    patchtst_v1_5

    Supervised adaptation of the PatchTST encoder and attentive
    classifier architecture used in benmfox/PhysioJEPA.

    Retained:
    - channel-specific grouped Conv1d patch embeddings
    - stride-aware zero end-padding; equivalent to PhysioJEPA's
      reference non-overlapping patch setup
    - channel-independent shared TST encoder
    - separate Q/K/V projections
    - rotary positional embeddings on encoder Q/K
    - post-norm LayerNorm
    - PhysioJEPA-style Transformer initialization/rescaling
    - per-channel shared learned-query attentive pooling
    - complete depth-1 cross-attention pooling block
    - no RoPE in pooling cross-attention
    - pooled channel embeddings concatenated before final output

    Intentional differences:
    - supervised training from random initialization
    - no masked reconstruction
    - no patch masking
    - no reconstruction head
    - no pretrained checkpoint
    - no frozen encoder
    - existing project dataset windows and training infrastructure

    Fidelity configuration:
    - patch_len=125
    - stride=125
    - d_model=512
    - n_heads=8
    - n_layers=3
    - d_ff=2048
    - dropout=0.1
    - attn_dropout=0
    - pool_depth=1
    - pool_num_queries=1
    - pool_mlp_ratio=4
    - pool_complete_block=True
    - pool_affine=False
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        if config.model_variant == "patchtst_v2" and config.cross_channel_layers != 1:
            raise ValueError("patchtst_v2 currently requires cross_channel_layers=1")

        self.encoder = None
        self.encoder_norm = None

        if config.model_variant == "patchtst_v1_5":
            self._validate_v15_config()
            self.v15_patch_embed = PhysioJEPAPatchTokenizer(
                n_channels=config.n_channels,
                patch_len=config.patch_len,
                stride=config.stride,
                d_model=config.d_model,
            )
            self.n_patches = self.v15_patch_embed.num_patches(config.seq_len)
            if self.n_patches <= 0:
                raise ValueError(
                    f"patchtst_v1_5 tokenizer produced invalid n_patches={self.n_patches} "
                    f"for seq_len={config.seq_len}, patch_len={config.patch_len}, stride={config.stride}"
                )
        else:
            self.n_patches = (config.seq_len - config.patch_len) // config.stride + 1

        if config.model_variant == "patchtst_v1":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
            self.encoder_norm = nn.LayerNorm(config.d_model)
            self.patch_embed = PatchEmbeddingV1(
                patch_len=config.patch_len,
                stride=config.stride,
                d_model=config.d_model,
            )
            self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, config.d_model) * 0.02)
            self.head = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, 1),
            )
        elif config.model_variant == "patchtst_v1_5":
            self.v15_embedding_dropout = nn.Dropout(config.dropout)
            self.v15_encoder = nn.ModuleList([
                PhysioJEPATSTBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    attn_dropout=config.attn_dropout,
                    qkv_bias=config.qkv_bias,
                    rotary_pes=True,
                    pre_norm=False,
                )
                for _ in range(config.n_layers)
            ])
            self.v15_pooler = PhysioJEPAAttentivePooler(
                num_queries=config.pool_num_queries,
                embed_dim=config.d_model,
                num_heads=config.n_heads,
                mlp_ratio=config.pool_mlp_ratio,
                depth=config.pool_depth,
                init_std=0.02,
                qkv_bias=config.qkv_bias,
                complete_block=config.pool_complete_block,
            )
            self.v15_head = nn.Linear(config.n_channels * config.d_model, 1, bias=True)
            self._init_v15_weights()
            self._rescale_v15_blocks()
        elif config.model_variant == "patchtst_v2":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
            self.encoder_norm = nn.LayerNorm(config.d_model)
            self.patch_embed = PatchEmbedding(
                patch_len=config.patch_len,
                stride=config.stride,
                d_model=config.d_model,
            )
            self.channel_embed = nn.Embedding(config.n_channels, config.d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, config.d_model) * 0.02)
            self.fusion_layers = nn.ModuleList(
                [
                    LocalCrossChannelFusion(
                        d_model=config.d_model,
                        n_heads=config.cross_channel_heads,
                        d_ff=config.d_ff,
                        dropout=config.dropout,
                        window=config.cross_channel_window,
                    )
                    for _ in range(config.cross_channel_layers)
                ]
            )
            self.pool = None
            if config.pooling_type == "attention":
                self.pool = AttentionPooling(
                    d_model=config.d_model,
                    n_heads=config.cross_channel_heads,
                    dropout=config.dropout,
                )
            self.final_norm = nn.LayerNorm(config.d_model)
            self.head = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, 1),
            )
        else:
            raise ValueError(f"Unknown model_variant: {config.model_variant}")

    def _validate_v15_config(self) -> None:
        if RotaryEmbedding is None:
            raise ImportError(
                "patchtst_v1_5 requires rotary_embedding_torch. "
                "Install the dependency or use patchtst_v1/patchtst_v2."
            )
        if self.config.patch_len <= 0:
            raise ValueError("patchtst_v1_5 requires patch_len > 0")
        if self.config.stride <= 0:
            raise ValueError("patchtst_v1_5 requires stride > 0")
        if self.config.n_heads <= 0:
            raise ValueError("patchtst_v1_5 requires n_heads > 0")
        if self.config.d_model % self.config.n_heads != 0:
            raise ValueError(
                f"patchtst_v1_5 requires d_model divisible by n_heads; got d_model={self.config.d_model}, n_heads={self.config.n_heads}"
            )
        if self.config.pool_depth != 1:
            raise ValueError(
                "patchtst_v1_5 currently requires pool_depth=1 for PhysioJEPA fidelity."
            )
        if self.config.pool_num_queries != 1:
            raise ValueError(
                "patchtst_v1_5 currently requires pool_num_queries=1 for PhysioJEPA fidelity."
            )
        if self.config.pool_affine:
            raise ValueError(
                "patchtst_v1_5 fidelity path requires pool_affine=False."
            )

    def _init_v15_weights(self) -> None:
        self.v15_patch_embed.apply(_init_physiojepa_module)
        self.v15_encoder.apply(_init_physiojepa_module)

    def _rescale_v15_blocks(self) -> None:
        for layer_id, layer in enumerate(self.v15_encoder, start=1):
            layer.self_attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            layer.ff[3].weight.data.div_(math.sqrt(2.0 * layer_id))

    def encode_channels(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> (B, C, T, D) with shared temporal encoder."""
        B, C, _ = x.shape
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed.unsqueeze(1)
        flat_tokens = tokens.reshape(B * C, self.n_patches, self.config.d_model)
        flat_tokens = self.encoder(flat_tokens)
        flat_tokens = self.encoder_norm(flat_tokens)
        return flat_tokens.reshape(B, C, self.n_patches, self.config.d_model)

    def fuse_channels(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, C, T, D) -> (B, T, D)."""
        if self.config.model_variant != "patchtst_v2":
            raise RuntimeError("fuse_channels is only available for patchtst_v2")
        fused_input = tokens + self.channel_embed(
            torch.arange(tokens.size(1), device=tokens.device)
        )[None, :, None, :]
        return self.fusion_layers[0](fused_input)

    def pool_sequence(self, fused: torch.Tensor) -> torch.Tensor:
        if self.config.pooling_type == "attention":
            pooled = self.pool(fused)
        else:
            pooled = fused.mean(dim=1)
        return self.final_norm(pooled)

    def forward_features(self, x: torch.Tensor, return_debug: bool = False):
        """Return pooled latent, optionally with shape metadata."""
        if self.config.model_variant == "patchtst_v1_5":
            tokens = self.v15_patch_embed(x)
            B, C, T, D = tokens.shape
            encoded = tokens.reshape(B * C, T, D)
            encoded = self.v15_embedding_dropout(encoded)
            for block in self.v15_encoder:
                encoded = block(encoded)
            encoded = encoded.reshape(B, C, T, D)

            pooler_raw = self.v15_pooler(encoded.reshape(B * C, T, D))
            pooled = pooler_raw.reshape(B, C, self.config.pool_num_queries, D).squeeze(2)
            latent = pooled.flatten(start_dim=1)

            if not return_debug:
                return latent
            debug = {
                "input": tuple(x.shape),
                "patch_tokens": (B, C, T, D),
                "encoder_input": (B * C, T, D),
                "encoder_output": (B, C, T, D),
                "pooler_raw_output": (B * C, 1, D),
                "pooled_channels": (B, C, D),
                "latent": (B, C * D),
            }
            return latent, debug

        channel_tokens = self.encode_channels(x)
        if self.config.model_variant == "patchtst_v1":
            latent = channel_tokens.mean(dim=(1, 2))
            if not return_debug:
                return latent
            debug = {
                "input": tuple(x.shape),
                "channel_tokens": tuple(channel_tokens.shape),
                "encoder_input": (x.size(0) * x.size(1), self.n_patches, self.config.d_model),
                "latent": tuple(latent.shape),
                "local_tokens_per_t": None,
            }
            return latent, debug

        fused_tokens = self.fuse_channels(channel_tokens)
        latent = self.pool_sequence(fused_tokens)
        if not return_debug:
            return latent
        debug = {
            "input": tuple(x.shape),
            "channel_tokens": tuple(channel_tokens.shape),
            "encoder_input": (x.size(0) * x.size(1), self.n_patches, self.config.d_model),
            "fused_tokens": tuple(fused_tokens.shape),
            "latent": tuple(latent.shape),
            "local_tokens_per_t": self.config.n_channels * (2 * self.config.cross_channel_window + 1),
        }
        return latent, debug

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> (B, 1) prediction"""
        latent = self.forward_features(x)
        if self.config.model_variant == "patchtst_v1_5":
            return self.v15_head(latent)
        return self.head(latent)


# ── Target extraction ─────────────────────────────────────────────────────────


class TargetExtractor:
    """
    Extracts a single target column from the full target bundle.

    The bundle has:
      - feature_targets: (N, *) — feature targets across configured horizons
      - feature_mask: (N, *)
      - event_targets: (N, *) — event targets across configured horizons
      - event_mask: (N, *)
    """

    def __init__(self, config: TrainConfig, target_path: Path):
        bundle = np.load(target_path, allow_pickle=True)
        metadata = {}
        metadata_path = target_path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as handle:
                metadata = json.load(handle)
        pids = bundle["anchor_patient_ids"].astype(str)
        times = bundle["anchor_times"].astype(np.float64)

        # Build lookup: (patient_id, anchor_time) -> row index
        self.index = {
            (str(pid), float(t)): idx
            for idx, (pid, t) in enumerate(zip(pids, times))
        }

        if config.task == "feature":
            all_targets = bundle["feature_targets"]  # (N, 78)
            all_masks = bundle["feature_mask"]  # (N, 78)
            target_names = metadata.get("feature_target_names")
            expected_mode = metadata.get("feature_spec", {}).get("horizon_mode")
            if expected_mode and expected_mode != config.feature_horizon_mode:
                raise ValueError(
                    f"Feature horizon mode mismatch: config={config.feature_horizon_mode}, "
                    f"bundle={expected_mode}"
                )
            col_idx = self._feature_col_index(
                config.feature_name,
                config.horizon,
                config.feature_horizon_mode,
                target_names,
            )
        else:
            all_targets = bundle["event_targets"].astype(np.float32)
            all_masks = bundle["event_mask"]
            target_names = metadata.get("event_target_names")
            col_idx = self._event_col_index(config.event_name, config.horizon, target_names)

        self.targets = all_targets[:, col_idx]  # (N,)
        self.masks = all_masks[:, col_idx]  # (N,)
        # Data-loader-level safeguard: never treat non-finite targets as valid.
        nonfinite_in_valid = ~np.isfinite(self.targets) & self.masks
        if nonfinite_in_valid.sum() > 0:
            print(f"  WARNING: {nonfinite_in_valid.sum()} non-finite values in valid targets — masking them out",
                  flush=True)
            self.masks = self.masks & np.isfinite(self.targets)
        self.task = config.task

    @staticmethod
    def _feature_col_index(
        feature_name: str,
        horizon: int,
        feature_horizon_mode: str,
        target_names: list[str] | None = None,
    ) -> int:
        if target_names:
            return target_names.index(feature_target_name(feature_name, horizon, feature_horizon_mode))
        horizon_idx = FEATURE_HORIZONS.index(horizon)
        feat_idx = ALL_FEATURE_NAMES.index(feature_name)
        return horizon_idx * len(ALL_FEATURE_NAMES) + feat_idx

    @staticmethod
    def _event_col_index(
        event_name: str,
        horizon: int,
        target_names: list[str] | None = None,
    ) -> int:
        if target_names:
            return target_names.index(f"{event_name}_within_{horizon}m")
        horizon_idx = EVENT_HORIZONS.index(horizon)
        event_idx = EVENT_NAMES.index(event_name)
        return horizon_idx * len(EVENT_NAMES) + event_idx

    def get(self, patient_id: str, anchor_time: float):
        """Returns (target_value, is_valid) for a single window."""
        row_idx = self.index.get((patient_id, anchor_time))
        if row_idx is None:
            return 0.0, False
        target = float(self.targets[row_idx])
        is_valid = bool(self.masks[row_idx]) and np.isfinite(target)
        if not is_valid:
            return 0.0, False
        return target, True


# ── Dataset wrapper with single-target ────────────────────────────────────────


class SingleTargetDataset(torch.utils.data.Dataset):
    """Wraps NumpyWaveformDataset to attach a single scalar target per window.

    Filters out windows that have no valid target (mask=False) at construction
    time, so the DataLoader never yields fully-masked batches. Patient boundaries
    are recomputed to reflect only valid windows.
    """

    def __init__(self, numpy_ds: NumpyWaveformDataset, target_extractor: TargetExtractor | None):
        self.numpy_ds = numpy_ds
        self.target_extractor = target_extractor

        if target_extractor is not None:
            # Pre-filter: keep only windows with valid targets
            # Build mapping from new (filtered) index -> original dataset index
            self._valid_indices: list[int] = []
            self._patient_ids_filtered: list[str] = []
            self._patient_boundaries_filtered: list[tuple[int, int]] = []

            orig_pids = numpy_ds.patient_ids
            orig_bounds = numpy_ds.patient_boundaries

            for p_idx, pid in enumerate(orig_pids):
                start, end = orig_bounds[p_idx]
                patient_valid_start = len(self._valid_indices)

                for orig_idx in range(start, end):
                    # Look up mask without loading waveform
                    window_pid, anchor_center = numpy_ds._windows[orig_idx]
                    seg_start = numpy_ds._patient_seg_start[window_pid]
                    anchor_time = seg_start + anchor_center / float(numpy_ds.fs)
                    _, is_valid = target_extractor.get(window_pid, anchor_time)
                    if is_valid:
                        self._valid_indices.append(orig_idx)

                patient_valid_end = len(self._valid_indices)
                if patient_valid_end > patient_valid_start:
                    self._patient_ids_filtered.append(pid)
                    self._patient_boundaries_filtered.append(
                        (patient_valid_start, patient_valid_end)
                    )

            n_orig = len(numpy_ds)
            n_valid = len(self._valid_indices)
            n_dropped = n_orig - n_valid
            print(f"  Filtered dataset: {n_valid}/{n_orig} windows have valid targets "
                  f"({n_dropped} dropped, {n_dropped/max(n_orig,1)*100:.1f}%)", flush=True)
        else:
            # No filtering (smoke test mode)
            self._valid_indices = None

    def __len__(self):
        if self._valid_indices is not None:
            return len(self._valid_indices)
        return len(self.numpy_ds)

    @property
    def patient_ids(self):
        if self._valid_indices is not None:
            return self._patient_ids_filtered
        return self.numpy_ds.patient_ids

    @property
    def patient_boundaries(self):
        if self._valid_indices is not None:
            return self._patient_boundaries_filtered
        return self.numpy_ds.patient_boundaries

    def __getitem__(self, index: int) -> dict:
        # Map filtered index to original dataset index
        orig_index = self._valid_indices[index] if self._valid_indices is not None else index

        sample = self.numpy_ds[orig_index]
        out = {
            "waveform": sample["waveform"],
        }

        if self.target_extractor is not None:
            target, mask = self.target_extractor.get(
                sample["patient_id"], sample["anchor_time"]
            )
            out["target"] = target
            out["mask"] = mask
        else:
            # Dummy target for smoke testing
            out["target"] = 0.0
            out["mask"] = True

        return out


# ── Collate ───────────────────────────────────────────────────────────────────


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate waveforms and single-target scalars. Minimal overhead."""
    waveforms = torch.stack([s["waveform"] for s in batch])
    waveforms = torch.nan_to_num(waveforms, nan=0.0, posinf=0.0, neginf=0.0)
    targets = torch.tensor([s["target"] for s in batch], dtype=torch.float32)
    masks = torch.tensor([s["mask"] for s in batch], dtype=torch.bool)

    finite_targets = torch.isfinite(targets)
    if not finite_targets.all():
        targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        masks = masks & finite_targets

    finite_waveforms = torch.isfinite(waveforms).all(dim=2).all(dim=1)
    if not finite_waveforms.all():
        masks = masks & finite_waveforms

    return {"waveform": waveforms, "target": targets, "mask": masks}


# ── Training utilities ────────────────────────────────────────────────────────


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE loss over valid samples only."""
    # Extra safety: exclude any NaN targets that slip through
    mask = mask & ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    diff = (pred.squeeze(-1) - target) ** 2
    return (diff * mask.float()).sum() / mask.float().sum()


def masked_bce_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BCE with logits over valid samples only."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    loss = nn.functional.binary_cross_entropy_with_logits(
        pred.squeeze(-1), target, reduction="none"
    )
    return (loss * mask.float()).sum() / mask.float().sum()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Main training loop ────────────────────────────────────────────────────────


def train(config: TrainConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.resolve_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task: {config.task}", flush=True)
    print(f"Target: {config.target_key}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(flush=True)

    # Save config
    (output_dir / "config.json").write_text(json.dumps(
        {k: getattr(config, k) for k in config.__dataclass_fields__},
        indent=2, default=str
    ))

    # Target extractor
    target_extractor = None
    if config.target_path:
        target_extractor = TargetExtractor(config, Path(config.target_path))
        print(f"Loaded targets from {config.target_path}", flush=True)

    # Datasets — using pre-extracted numpy files
    print("Loading datasets...", flush=True)
    train_numpy_ds = NumpyWaveformDataset(
        split="train",
        waveform_dir=Path(config.waveform_dir),
        splits_path=Path(config.splits_path),
        normalize=config.normalize,
        channels=config.channels,
        seq_len=config.seq_len,
    )
    val_numpy_ds = NumpyWaveformDataset(
        split="val",
        waveform_dir=Path(config.waveform_dir),
        splits_path=Path(config.splits_path),
        normalize=config.normalize,
        channels=config.channels,
        seq_len=config.seq_len,
    )

    train_ds = SingleTargetDataset(train_numpy_ds, target_extractor)
    val_ds = SingleTargetDataset(val_numpy_ds, target_extractor)
    print(f"  Train: {len(train_ds):,} windows ({len(train_ds.patient_ids)} patients)", flush=True)
    print(f"  Val:   {len(val_ds):,} windows ({len(val_ds.patient_ids)} patients)", flush=True)

    # Patient-grouped batch sampler for training
    train_sampler = PatientGroupedSampler(
        patient_boundaries=train_ds.patient_boundaries,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        seed=config.seed,
    )

    # Val sampler: patient-grouped but no shuffle (deterministic ordering)
    val_sampler = PatientGroupedSampler(
        patient_boundaries=val_ds.patient_boundaries,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        seed=config.seed,
    )

    # DataLoader with optimized settings
    # - batch_sampler: yields pre-formed batches (patient-grouped)
    # - num_workers: overlap I/O with GPU compute
    # - pin_memory: faster CPU->GPU transfer
    # - prefetch_factor: keep more batches ready
    # - persistent_workers: avoid worker respawn overhead
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    # Model
    model = PatchTST(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} params", flush=True)
    if config.model_variant == "patchtst_v1_5":
        log_v15_architecture(config, model.n_patches)

    compile_supported = hasattr(torch, "compile") and config.model_variant == "patchtst_v1"
    if compile_supported:
        try:
            model = torch.compile(model)
            print("  torch.compile enabled", flush=True)
        except Exception as e:
            print(f"  torch.compile failed ({e}), using eager mode", flush=True)
    elif config.model_variant in {"patchtst_v1_5", "patchtst_v2"}:
        print(f"  torch.compile disabled for {config.model_variant}", flush=True)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * config.epochs
    warmup_steps = len(train_loader) * config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss
    loss_fn = masked_bce_loss if config.task == "event" else masked_mse_loss

    # Enable TF32 for faster matmuls on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    if config.resume:
        ckpt_path = output_dir / "latest_model.pt"
        if ckpt_path.exists():
            print(f"Resuming from {ckpt_path}", flush=True)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            # Handle compiled model state dict
            state_dict = ckpt["model_state_dict"]
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                # Remove _orig_mod prefix from compiled model keys
                cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(cleaned, strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"]
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
            # Advance scheduler to correct step
            for _ in range(start_epoch * len(train_loader)):
                scheduler.step()
            print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}", flush=True)
        else:
            print("No checkpoint found, starting from scratch", flush=True)

    # Training
    n_steps_per_epoch = len(train_loader)
    print(f"\nTraining: epochs {start_epoch+1}–{config.epochs}, "
          f"{n_steps_per_epoch} steps/epoch, batch_size={config.batch_size}", flush=True)
    print(f"Total steps: {total_steps:,}, warmup: {warmup_steps:,}", flush=True)
    print("-" * 60, flush=True)

    # Use automatic mixed precision for faster training
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        n_valid = 0
        t0 = time.time()

        # Set epoch for sampler (ensures different shuffle each epoch)
        train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            if step == 0:
                step_t0 = time.time()

            waveform = batch["waveform"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            # Forward pass with AMP, but keep loss computation in fp32 for numerical stability
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(waveform)
            loss = loss_fn(pred.float(), target.float(), mask)

            # Skip backward/step if batch has no valid targets (avoids GradScaler assertion)
            n_valid_batch = mask.sum().item()
            if n_valid_batch == 0:
                scheduler.step()
                continue

            # Skip non-finite losses so a single bad batch does not poison epoch accounting
            if not torch.isfinite(loss):
                lr = scheduler.get_last_lr()[0]
                print(f"  WARNING: non-finite train loss at epoch {epoch+1} step {step+1}/{n_steps_per_epoch}; skipping batch (lr={lr:.2e})", flush=True)
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                continue

            # Backward pass with gradient scaling
            optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()
            scaler.scale(loss).backward()

            # Unscale gradients for clipping, then step
            scaler.unscale_(optimizer)
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item() * n_valid_batch
            n_valid += n_valid_batch

            if step == 0:
                elapsed_first = time.time() - step_t0
                print(f"  E{epoch+1} first step: {elapsed_first:.2f}s "
                      f"(includes data loading warmup)", flush=True)

            if (step + 1) % config.log_interval == 0:
                avg = epoch_loss / max(n_valid, 1)
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                steps_per_sec = (step + 1) / elapsed
                samples_per_sec = (step + 1) * config.batch_size / elapsed
                eta_epoch = (n_steps_per_epoch - step - 1) / steps_per_sec
                print(f"  E{epoch+1} step {step+1}/{n_steps_per_epoch} "
                      f"loss={avg:.6f} lr={lr:.2e} "
                      f"[{steps_per_sec:.1f} steps/s, {samples_per_sec:.0f} samples/s, "
                      f"ETA {eta_epoch/60:.1f}m]", flush=True)

        epoch_time = time.time() - t0
        train_loss = epoch_loss / max(n_valid, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_valid = 0
        with torch.no_grad():
            for batch in val_loader:
                waveform = batch["waveform"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred = model(waveform)
                loss = loss_fn(pred.float(), target.float(), mask)

                if not torch.isfinite(loss):
                    print("  WARNING: non-finite validation loss encountered; skipping batch", flush=True)
                    continue

                val_loss_sum += loss.item() * mask.sum().item()
                val_valid += mask.sum().item()

        val_loss = val_loss_sum / max(val_valid, 1)

        print(f"Epoch {epoch+1}/{config.epochs}: "
              f"train={train_loss:.6f} val={val_loss:.6f} "
              f"({epoch_time:.1f}s, {n_valid:,} valid samples, "
              f"{n_valid/epoch_time:.0f} samples/s)", flush=True)

        # Checkpoint — save model state without compile wrapper
        model_state = model.state_dict()
        # Strip _orig_mod prefix if compiled
        cleaned_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}

        improved = val_loss < (best_val_loss - config.early_stopping_min_delta)
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": cleaned_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "config": {k: getattr(config, k) for k in config.__dataclass_fields__},
                "target_key": config.target_key,
            }, output_dir / "best_model.pt")
            print(f"  -> Best (val_loss={val_loss:.6f})", flush=True)
        else:
            epochs_without_improvement += 1

        # Always save latest for resume
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": cleaned_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "config": {k: getattr(config, k) for k in config.__dataclass_fields__},
            "target_key": config.target_key,
        }, output_dir / "latest_model.pt")

        if (
            config.early_stopping_patience > 0
            and (epoch + 1) >= config.early_stopping_min_epochs
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                "Early stopping triggered: "
                f"no validation improvement greater than {config.early_stopping_min_delta:.6g} "
                f"for {epochs_without_improvement} epoch(s) after epoch {epoch + 1}.",
                flush=True,
            )
            break

    print(f"\nDone. Best val_loss={best_val_loss:.6f}", flush=True)
    print(f"Saved to {output_dir}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train PatchTST on a single target (optimized)")
    parser.add_argument("--list-targets", action="store_true",
                        help="Print all available targets and exit")

    # Task
    parser.add_argument("--task", type=str, default="feature",
                        choices=["feature", "event"])
    parser.add_argument("--feature-name", type=str, default="MAP",
                        help=f"One of: {ALL_FEATURE_NAMES}")
    parser.add_argument("--event-name", type=str, default="hypotension",
                        choices=EVENT_NAMES)
    parser.add_argument("--horizon", type=int, default=0,
                        help="Prediction horizon in minutes")
    parser.add_argument("--feature-horizon-mode", type=str, default="center",
                        choices=["center", "gap"])

    # Architecture
    parser.add_argument("--model-variant", type=str, default="patchtst_v1",
                        choices=["patchtst_v1", "patchtst_v1_5", "patchtst_v2"])
    parser.add_argument("--physiojepa-fidelity", action="store_true",
                        help="Apply the PhysioJEPA-fidelity patchtst_v1_5 architecture preset.")
    parser.add_argument("--channels", type=str, default="ABP,II,PLETH",
                        help="Comma-separated waveform channel order to load")
    parser.add_argument("--seq-len", type=int, default=150_000)
    parser.add_argument("--n-channels", type=int, default=3)
    parser.add_argument("--patch-len", type=int, default=250)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--qkv-bias", action="store_true", default=True)
    parser.add_argument("--no-qkv-bias", action="store_false", dest="qkv_bias")
    parser.add_argument("--cross-channel-layers", type=int, default=1)
    parser.add_argument("--cross-channel-heads", type=int, default=4)
    parser.add_argument("--cross-channel-window", type=int, default=1)
    parser.add_argument("--pooling-type", type=str, default="mean",
                        choices=["mean", "attention"])
    parser.add_argument("--pool-depth", type=int, default=1)
    parser.add_argument("--pool-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--pool-num-queries", type=int, default=1)
    parser.add_argument("--pool-complete-block", action="store_true", default=True)
    parser.add_argument("--no-pool-complete-block", action="store_false", dest="pool_complete_block")
    parser.add_argument("--pool-affine", action="store_true", default=False)
    parser.add_argument("--no-pool-affine", action="store_false", dest="pool_affine")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after this many epochs without validation improvement; 0 disables.")
    parser.add_argument("--early-stopping-min-epochs", type=int, default=0,
                        help="Do not early-stop before this many epochs have completed.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                        help="Minimum validation-loss improvement required to reset patience.")

    # Infra
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--target-path", type=str, default="")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint if available")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Optional output subdirectory tag")

    # Paths
    parser.add_argument("--waveform-dir", type=str,
                        default="/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/waveforms")
    parser.add_argument("--splits-path", type=str, default="outputs/splits/splits.json")

    args = parser.parse_args()
    provided_flags = _cli_flag_set()
    apply_physiojepa_fidelity_preset(args, provided_flags, parser)

    channels = parse_channel_list(args.channels)
    if len(channels) != args.n_channels:
        parser.error(
            f"--n-channels={args.n_channels} does not match parsed channel list {channels}"
        )

    if args.list_targets:
        list_all_targets(feature_horizon_mode=args.feature_horizon_mode)
        return

    # Validate
    if args.task == "feature" and args.feature_name not in ALL_FEATURE_NAMES:
        parser.error(f"Unknown feature: {args.feature_name}. Choose from: {ALL_FEATURE_NAMES}")
    if args.task == "feature" and args.horizon not in FEATURE_HORIZONS:
        parser.error(f"Feature horizon must be one of {FEATURE_HORIZONS}")
    if args.task == "event" and args.horizon not in EVENT_HORIZONS:
        parser.error(f"Event horizon must be one of {EVENT_HORIZONS}")

    config = TrainConfig(
        task=args.task,
        feature_name=args.feature_name,
        event_name=args.event_name,
        horizon=args.horizon,
        feature_horizon_mode=args.feature_horizon_mode,
        run_tag=args.run_tag,
        physiojepa_fidelity=args.physiojepa_fidelity,
        channels=channels,
        model_variant=args.model_variant,
        n_channels=args.n_channels,
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        qkv_bias=args.qkv_bias,
        cross_channel_layers=args.cross_channel_layers,
        cross_channel_heads=args.cross_channel_heads,
        cross_channel_window=args.cross_channel_window,
        pooling_type=args.pooling_type,
        pool_depth=args.pool_depth,
        pool_mlp_ratio=args.pool_mlp_ratio,
        pool_num_queries=args.pool_num_queries,
        pool_complete_block=args.pool_complete_block,
        pool_affine=args.pool_affine,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        early_stopping_min_delta=args.early_stopping_min_delta,
        num_workers=args.num_workers,
        normalize=args.normalize,
        seed=args.seed,
        output_dir=args.output_dir,
        target_path=args.target_path,
        log_interval=args.log_interval,
        resume=args.resume,
        waveform_dir=args.waveform_dir,
        splits_path=args.splits_path,
    )

    train(config)


if __name__ == "__main__":
    main()
