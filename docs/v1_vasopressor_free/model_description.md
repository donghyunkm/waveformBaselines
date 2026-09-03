# PatchTST Model Variants

This document records the main architectural differences between the current PatchTST variants in this repo and the configuration values that define them.

## Shared Context

All three variants:

- operate on waveform input shaped `(B, C, L)`
- use the existing supervised training pipeline in `scripts/train_patchtst.py`
- train one target at a time
- use `masked_bce_loss` for event classification and `masked_mse_loss` for feature regression
- keep dataset-level waveform normalization external to the model

The differences are in tokenization, temporal encoding, cross-channel handling, pooling, and the intended configuration scale.

## `patchtst_v1`

### Architecture

- patch tokenizer: per-channel `nn.Linear(patch_len, d_model)` applied after `unfold`
- temporal encoder: shared `nn.TransformerEncoder` over each channel sequence independently
- positional encoding: learned absolute positional embedding
- cross-channel modeling: none
- pooling: mean over channels and time
- output head:
  `Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, 1)`

### Representation Path

```text
(B,C,L)
-> unfold + linear patch projection
-> (B,C,T,D)
-> learned absolute positional embedding
-> reshape to (B*C,T,D)
-> shared Transformer encoder
-> reshape to (B,C,T,D)
-> mean over C and T
-> (B,D)
-> MLP head
-> (B,1)
```

### Typical / current baseline config

- channels: `ABP,II,PLETH`
- `patch_len=250`
- `stride=250`
- `d_model=128`
- `n_heads=8`
- `n_layers=4`
- `d_ff=256`
- `dropout=0.1`
- `batch_size=512`

## `patchtst_v1_5`

### Architecture

- supervised adaptation of the PatchTST encoder and attentive classifier architecture from `benmfox/PhysioJEPA`
- patch tokenizer: grouped `Conv1d` with `groups=C`, so each waveform channel has its own patch projection filters
- padding: stride-aware zero right-padding before tokenization when needed
- temporal encoder: shared channel-independent TST stack
- attention: separate `W_Q`, `W_K`, `W_V`
- positional encoding: rotary embeddings on encoder `Q/K`
- encoder block style: post-norm
- cross-channel modeling: none inside the encoder
- pooling: shared learned-query attentive pooler applied independently to each channel
- channel handling after pooling: concatenate pooled channel embeddings
- output head: final linear classifier/regressor `Linear(C*d_model, 1)`

### Representation Path

```text
(B,C,L)
-> grouped Conv1d tokenizer
-> (B,C,T,D)
-> reshape to (B*C,T,D)
-> embedding dropout
-> shared PhysioJEPA-style TST encoder
-> reshape to (B,C,T,D)
-> shared attentive pooler per channel
-> (B,C,D)
-> flatten channels
-> (B,C*D)
-> Linear(C*D,1)
-> (B,1)
```

### PhysioJEPA-fidelity config

Use `--physiojepa-fidelity` with `--model-variant patchtst_v1_5`.

- channels: `ABP,II,PLETH`
- `patch_len=125`
- `stride=125`
- `d_model=512`
- `n_heads=8`
- `n_layers=3`
- `d_ff=2048`
- `dropout=0.1`
- `attn_dropout=0.0`
- `qkv_bias=True`
- `pool_depth=1`
- `pool_num_queries=1`
- `pool_mlp_ratio=4.0`
- `pool_complete_block=True`
- `pool_affine=False`
- `batch_size=32`

### Important differences from original PhysioJEPA training

- trained end-to-end from random initialization
- no masked reconstruction
- no patch masking
- no reconstruction head
- no pretrained checkpoint loading
- no frozen encoder

## `patchtst_v2`

### Architecture

- patch tokenizer: same basic `unfold` + linear patch projection family as the local repo baseline path
- temporal encoder: shared `nn.TransformerEncoder` over each channel sequence independently
- positional encoding: learned absolute positional embedding
- cross-channel modeling: explicit local cross-channel fusion after temporal encoding
- local fusion mechanism:
  learned fusion query attends to a local channel/time neighborhood
- pooling: mean or attention pooling over fused temporal tokens
- output head:
  `Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, 1)`

### Representation Path

```text
(B,C,L)
-> unfold + linear patch projection
-> (B,C,T,D)
-> learned absolute positional embedding
-> reshape to (B*C,T,D)
-> shared Transformer encoder
-> reshape to (B,C,T,D)
-> add channel embeddings
-> local cross-channel fusion
-> (B,T,D)
-> mean or attention pooling
-> (B,D)
-> MLP head
-> (B,1)
```

### Typical / pilot config

The code supports a range of experimental `v2` settings. The current intended pilot-style configuration is:

- channels: often `II,PLETH,ABP,RESP`
- `patch_len=64`
- `stride=64`
- `d_model=128`
- `n_heads=4`
- `n_layers=3`
- `d_ff=256`
- `dropout=0.1`
- `cross_channel_layers=1`
- `cross_channel_heads=4`
- `cross_channel_window=1`
- `pooling_type=attention`

## Summary Table

| Variant | Tokenizer | Positional encoding | Encoder | Cross-channel modeling | Pooling | Final head | Typical scale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `patchtst_v1` | `unfold` + per-channel linear | learned absolute | generic shared Transformer | none | mean over channels and time | MLP | `250 / 128 / 4 / 256` |
| `patchtst_v1_5` | grouped `Conv1d`, channel-specific filters | RoPE on encoder `Q/K` | PhysioJEPA-style TST blocks | none | per-channel attentive pooler | `Linear(C*d_model,1)` | `125 / 512 / 3 / 2048` |
| `patchtst_v2` | `unfold` + linear | learned absolute | generic shared Transformer | local post-encoder fusion | mean or attention over fused tokens | MLP | `64 / 128 / 3 / 256` pilot-style |

### Interpreting the shorthand

In the summary table:

- first number = `patch_len` and usually `stride`
- second = `d_model`
- third = `n_layers`
- fourth = `d_ff`
