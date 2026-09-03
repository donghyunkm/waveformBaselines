from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SequenceModelConfig:
    input_dim: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    output_dim: int = 1
    use_cls_token: bool = True
    max_seq_len: int = 20
    gru_hidden_dim: int = 128
    gru_layers: int = 1
    tcn_hidden_dim: int = 128
    tcn_blocks: int = 3
    tcn_kernel_size: int = 3

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {self.dropout}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be at least 1, got {self.max_seq_len}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be at least 1, got {self.n_layers}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.gru_hidden_dim <= 0:
            raise ValueError(f"gru_hidden_dim must be positive, got {self.gru_hidden_dim}")
        if self.gru_layers < 1:
            raise ValueError(f"gru_layers must be at least 1, got {self.gru_layers}")
        if self.tcn_hidden_dim <= 0:
            raise ValueError(f"tcn_hidden_dim must be positive, got {self.tcn_hidden_dim}")
        if self.tcn_blocks < 1:
            raise ValueError(f"tcn_blocks must be at least 1, got {self.tcn_blocks}")
        if self.tcn_kernel_size < 1:
            raise ValueError(f"tcn_kernel_size must be at least 1, got {self.tcn_kernel_size}")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.embedding.size(1):
            raise ValueError(
                f"Input sequence length {x.size(1)} exceeds positional capacity {self.embedding.size(1)}"
            )
        return x + self.embedding[:, : x.size(1)]


class TransformerSequenceModel(nn.Module):
    def __init__(self, config: SequenceModelConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.input_dim, config.d_model)
        positional_len = config.max_seq_len + (1 if config.use_cls_token else 0)
        self.pos = PositionalEncoding(config.d_model, max_len=positional_len)
        self.cls = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02) if config.use_cls_token else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.cls is not None:
            cls = self.cls.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm(x)
        pooled = x[:, 0] if self.cls is not None else x.mean(dim=1)
        return self.head(pooled)


class GRUSequenceModel(nn.Module):
    def __init__(self, config: SequenceModelConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(config.input_dim, config.gru_hidden_dim)
        self.gru = nn.GRU(
            input_size=config.gru_hidden_dim,
            hidden_size=config.gru_hidden_dim,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(config.gru_hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(config.gru_hidden_dim, config.gru_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.gru_hidden_dim, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x, _ = self.gru(x)
        pooled = self.norm(x[:, -1])
        return self.head(pooled)

class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TCNResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = ChannelLayerNorm(channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm2 = ChannelLayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual


class TCNSequenceModel(nn.Module):
    def __init__(self, config: SequenceModelConfig) -> None:
        super().__init__()
        self.config = config
        self.dilations = [2**i for i in range(config.tcn_blocks)]
        self.proj = nn.Linear(config.input_dim, config.tcn_hidden_dim)
        self.blocks = nn.Sequential(
            *[
                TCNResidualBlock(
                    channels=config.tcn_hidden_dim,
                    kernel_size=config.tcn_kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                )
                for dilation in self.dilations
            ]
        )
        self.norm = nn.LayerNorm(config.tcn_hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(config.tcn_hidden_dim, config.tcn_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tcn_hidden_dim, config.output_dim),
        )

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.config.tcn_kernel_size - 1) * sum(self.dilations)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).transpose(1, 2)
        x = self.blocks(x).transpose(1, 2)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_sequence(x)
        pooled = x[:, -1]
        return self.head(pooled)

