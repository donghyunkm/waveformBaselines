# Extracted Feature Model Description

This document describes the model families used for the v7 extracted-feature experiments summarized in `docs/v7_extracted_features/extractedFeaturesRegression.md` and `docs/v7_extracted_features/extractedFeaturesClassification.md`. These models do not operate on raw waveforms. They consume the engineered v7 physiological feature cache produced by the waveform-feature extraction pipeline.

## Shared Inputs

All extracted-feature models start from the same production v7 feature cache:

`/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtraction/v7/vasopressor_free_waveform_features_v7`

Important shape convention: `N` means the number of examples currently being passed to a model. It is not always the full cache size. For example, `N` can mean all cache rows before target filtering, valid training rows for one target, valid validation rows for one target, or valid test rows for one target. The exact `N` changes by split and by target because invalid/missing target rows are removed.

The raw feature cache before preprocessing has:

- `values`: shape `(N_cache, 20, 93)`
- `mask`: shape `(N_cache, 20, 93)`
- `N_cache = 334833` rows in the merged v7 vasopressor-free cache
- `20` time steps per row: one token per minute over the 20-minute input history
- `93` physiological feature values per time step

Each cache row corresponds to one waveform window/anchor, identified by `(patient_id, anchor_time)`. Target alignment uses rounded `(patient_id, anchor_time)` keys with `6` decimal places. Regression uses `outputs/targets/feature_targets_gap_vasopressor_free.npz`; filtered `5m` hypotension classification uses `outputs/targets/event_targets_vasopressor_free_anchor_horizon_filtered_5m_10m.npz`.

For XGBoost and neural extracted-feature models, preprocessing is fit only on the training split. Each of the `93` physiological values is imputed with its train-split median. Continuous features are train-split z-scored when their feature definition has `normalize=True`; bounded or indicator-like quality features keep identity scaling. The binary validity mask is concatenated to the imputed/scaled values at every time step.

After preprocessing, one example has:

- value channels: `93`
- mask channels: `93`
- total channels per time step: `186`
- full sequence shape for one example: `(20, 186)`
- model batch shape for sequence models: `(N, 20, 186)`

The persistence baseline is the exception: it does not use the v7 `(20, 93)` feature sequence or the `(20, 186)` preprocessed sequence. It directly uses the current-window value from the target-generation source table for the same target definition.

## Persistence

Persistence is a regression-only baseline. It has no learned input matrix and no fitted parameters. For each valid test example, it predicts the future target with that same example's current-window value for the same target definition. Its effective input is one scalar per example, so the prediction vector has shape `(N,)`.

For the all-target persistence path, current values are loaded from the legacy target-generation source table at `/gpfs/data/eh3828lab/derived_datasets/baselines/output_v2`, specifically `X_stats.npy` plus `corr_features_focused.npy`.

This choice is intentional: the 26 regression targets were generated from that same 26-column source table, so persistence uses exact current-window target semantics rather than approximate v7 feature-name mappings. A prediction is evaluated only when both the current source value and future target value are finite and valid.

Current status: all 26 persistence baselines are complete; MAP was already present and the remaining 25 jobs ran as `26918444`-`26918468`.

## Current-State XGBoost

Current-state XGBoost is a tabular model using only the final time step from each preprocessed sequence. For each valid example, the trainer takes `seq_x[:, -1, :]`, where `seq_x` has shape `(N, 20, 186)`. The resulting model input is `(N, 186)`.

Here, each row is one valid window/anchor for the current task and split. The `186` columns are the latest minute's `93` imputed/scaled physiological values concatenated with the latest minute's `93` validity indicators. This model discards the first `19` minutes of the sequence after preprocessing.

For regression, the trainer fits an `xgboost.XGBRegressor` with squared-error objective. For classification, it fits an `xgboost.XGBClassifier` with logistic objective and log-loss evaluation. The small hyperparameter grid is selected on validation performance:

- `max_depth`: `3`, `5`
- `n_estimators`: `100`, `200`
- `learning_rate`: `0.05`

Regression selects by validation RMSE. Classification selects by validation AUPRC.

## History XGBoost

History XGBoost is a tabular history-summary baseline. It consumes the original v7 value/mask sequence for each valid example, shaped `(20, 93)`, and converts that sequence into one flat feature vector. It summarizes each of the `93` physiological features across the 20-minute input history. For each physiological feature, it computes:

- mean
- median
- standard deviation
- minimum
- maximum
- first valid value
- last valid value
- last minus first
- linear slope over valid token positions
- valid fraction

The resulting input has shape `(N, 930)` because there are `93 * 10 = 930` summary columns per valid example. Unlike current-state XGBoost, this model uses information from all `20` one-minute tokens, but only after compressing the sequence into summary statistics. The same XGBoost regressor/classifier setup and validation selection rules are used as in current-state XGBoost.

This model is designed to test how much of the temporal signal can be captured by compact, interpretable summary statistics rather than a neural sequence model. In completed v7 results, history XGBoost is the strongest overall regression family so far and the best filtered `5m` hypotension classifier.

## Full-Sequence XGBoost

Full-sequence XGBoost is a tabular model that uses the same preprocessed input information as the GRU and Transformer, but without a learned sequence architecture. For each valid example, it starts with the full preprocessed sequence tensor `(20, 186)` and flattens it into one tabular row with `20 * 186 = 3720` columns. The model input shape is therefore `(N, 3720)`.

The flattened columns preserve time order by concatenating all channels from minute 1, then all channels from minute 2, through minute 20. Each minute contributes `93` imputed/scaled physiological values and `93` validity indicators. This means the model can use any current or historical feature value directly, but it sees time only through column position rather than through recurrence, attention, or explicit summary features.

This model is useful as a direct control for the neural sequence models. If full-sequence XGBoost performs close to GRU or Transformer, then much of the value may come from exposing the full 20-minute feature history rather than from learned sequence dynamics. If GRU or Transformer clearly outperform it, that supports the value of ordered temporal modeling beyond a high-dimensional tabular representation.

The XGBoost regressor/classifier setup and validation selection rules are the same as current-state XGBoost and history XGBoost. Prediction-export runs save `test_predictions.npz` with aligned test predictions/targets plus sample identifiers, and tabular replay artifacts are saved as `model.pkl`.

## Full-Sequence MLP

Full-sequence MLP is a neural tabular model over the same flattened input used by full-sequence XGBoost. For each valid example, the preprocessed sequence `(20, 186)` is flattened to one vector with `20 * 186 = 3720` inputs, so the model batch shape is `(N, 3720)`.

Unlike GRU and Transformer, this model does not keep an explicit sequence axis after flattening. Time is represented by fixed column position: the first block of `186` columns is minute 1, the next block is minute 2, and so on through minute 20. Unlike XGBoost, the mapping is learned with a feed-forward neural network trained by AdamW. The default architecture is:

```text
(N,3720)
-> Linear(3720, mlp_hidden_dim)
-> LayerNorm
-> GELU
-> Dropout
-> Linear/LayerNorm/GELU/Dropout repeated mlp_layers times
-> Linear(1)
```

Current default configuration in `RunConfig`:

- `mlp_hidden_dim=512`
- `mlp_layers=2`
- `dropout=0.1`
- `epochs=20` unless overridden
- optimizer: AdamW, `lr=1e-3`, `weight_decay=1e-4`

For regression, the loss is MSE. For classification, the loss is `BCEWithLogitsLoss`; test scores are sigmoid probabilities. Prediction-export runs save `test_predictions.npz` with aligned test predictions/targets plus sample identifiers, and the fitted neural weights are saved as `model.pt`.

## GRU

The GRU model is a neural sequence baseline over the full preprocessed feature sequence. Its input is the full `(N, 20, 186)` tensor: `N` valid examples, `20` one-minute time steps per example, and `186` channels per time step.

For each time step, the GRU first projects the `186` channels to `gru_hidden_dim`. It then processes the ordered 20-token sequence with an `nn.GRU`. The output from the final time step is layer-normalized and passed through a small MLP head:

```text
(N,20,186)
-> Linear(186, gru_hidden_dim)
-> GRU
-> final token output
-> LayerNorm
-> Linear -> GELU -> Dropout -> Linear(1)
```

Current default configuration in `RunConfig`:

- `gru_hidden_dim=128`
- `gru_layers=1`
- `dropout=0.1`
- `epochs=20` unless overridden
- optimizer: AdamW, `lr=1e-3`, `weight_decay=1e-4`

For regression, the loss is MSE. For classification, the loss is `BCEWithLogitsLoss`; models return logits during training and sigmoid is applied only for metric/output probabilities.

## TCN

The TCN model is a causal temporal convolutional sequence model over the full preprocessed feature sequence. Its input is the same `(N, 20, 186)` tensor used by GRU and Transformer: `N` valid examples, `20` ordered one-minute time steps, and `186` channels per time step.

The implementation projects each time step to `tcn_hidden_dim`, transposes to channel-first form for 1D temporal convolutions, applies stacked residual causal convolution blocks, then uses the final time-step representation for the task head:

```text
(N,20,186)
-> Linear(186, tcn_hidden_dim)
-> transpose to (N,tcn_hidden_dim,20)
-> residual causal Conv1d blocks with dilations 1, 2, 4
-> transpose back to (N,20,tcn_hidden_dim)
-> final token output
-> LayerNorm
-> Linear -> GELU -> Dropout -> Linear(1)
```

Current default configuration in `RunConfig`:

- `tcn_hidden_dim=128`
- `tcn_blocks=3`
- `tcn_kernel_size=3`
- dilations: `[1, 2, 4]`
- two causal convolutions per residual block
- `dropout=0.1`
- `epochs=20` unless overridden
- optimizer: AdamW, `lr=1e-3`, `weight_decay=1e-4`

The default receptive field is `1 + 2 * (kernel_size - 1) * sum(dilations) = 29` time steps. Because the input history has `20` one-minute steps, the final representation can cover the entire available history. Causality is enforced with left-only padding before every convolution; the model never pads on the right or uses centered convolution. Normalization is per-time-step channel LayerNorm, so future positions do not affect earlier causal representations through normalization.

For regression, the loss is MSE. For classification, the loss is `BCEWithLogitsLoss`; test scores are sigmoid probabilities. Prediction-export runs save `test_predictions.npz` with aligned test predictions/targets plus sample identifiers, and the fitted neural weights are saved as `model.pt`.

## Transformer

The Transformer model is the primary neural sequence model over extracted features. Its input is also the full `(N, 20, 186)` tensor: `N` valid examples, `20` ordered one-minute time steps, and `186` channels per time step.

It projects each 186-channel time step to `d_model`, prepends a learned CLS token, adds learned positional embeddings, and applies a standard `nn.TransformerEncoder`. The CLS output is layer-normalized and passed through an MLP head:

```text
(N,20,186)
-> Linear(186, d_model)
-> prepend CLS token
-> learned positional embedding
-> TransformerEncoder
-> CLS pooling
-> LayerNorm
-> Linear -> GELU -> Dropout -> Linear(1)
```

Current default configuration in `RunConfig`:

- `d_model=128`
- `n_heads=4`
- `n_layers=2`
- feed-forward width: `4 * d_model`
- `dropout=0.1`
- `max_seq_len=20` plus CLS capacity
- optimizer: AdamW, `lr=1e-3`, `weight_decay=1e-4`

For regression, the loss is MSE. For classification, the loss is `BCEWithLogitsLoss`; test scores are sigmoid probabilities.

## Regression Usage

Regression jobs train one target at a time for the `t+0m_gap` horizon. For a given target, rows with invalid target values are removed separately for train, validation, and test. Therefore each regression run has its own train/validation/test `N`, although all runs start from the same `334833` cache rows. The input window is the 20-minute current history, and the target is the adjacent future target window defined by `feature_horizon_mode=gap`. Metrics are reported in original target units:

- MAE
- RMSE
- R2

Completed all-target v7 regression families currently include current-state XGBoost, history XGBoost, full-sequence XGBoost, full-sequence MLP, GRU, Transformer, and TCN for all 26 targets. All-target persistence support has been implemented and all 26 persistence metrics are present.

## Classification Usage

The completed v7 classification comparison uses filtered `5m` hypotension labels. For this task, the documented PatchTST/v7 comparison test set has `N=3650` valid test examples with `194` positives. Models are trained as binary classifiers and evaluated on the test split with:

- AUROC
- AUPRC
- threshold at 85% sensitivity
- specificity at 85% sensitivity

The filtered task has low test prevalence (`194/3650`, about `5.3%`), so AUPRC and specificity at fixed sensitivity are especially important. In the completed v7 classification runs, history XGBoost is best on all reported metrics, followed closely by current-state XGBoost; GRU and Transformer improve over the raw-waveform PatchTST baseline but trail the tabular XGBoost models. Full-sequence XGBoost, full-sequence MLP, and TCN are complete for the same filtered `5m` hypotension task.

## Interpretation

The main empirical pattern is that explicit physiological features plus tabular nonlinear models are very strong for these tasks. History XGBoost usually outperforms current-state XGBoost, showing that recent temporal context matters. However, its advantage over GRU and Transformer on most completed tasks suggests that much of the useful temporal information is captured by summary statistics rather than requiring learned sequence dynamics.

The neural sequence models still have target-specific value. TCN was added to test whether local/dilated temporal structure helps on the same sequence input. After completed TCN significance/results insertion, TCN is best for `DBP`, GRU is best for `ABP_area` and `ABP_tau`, and Transformer is best for targets such as `dPdt_max`, `MAP`, and `PLETH_ACDC_ShockIdx`. These wins should be interpreted target by target rather than as a broad sequence-model advantage.

All downstream results should still be interpreted with the v7 ECG/ABP rate-disagreement caveat documented in `docs/v7_extracted_features/extractedFeatures.md`.

## Sequence-Model Correctness Audit - 2026-08-29

Scope: `TransformerSequenceModel`, `GRUSequenceModel`, `TCNSequenceModel`, and the shared extracted-feature training path in `scripts/train_feature_models.py`.

Confirmed input semantics:

- The production v7 cache is `(334833, 20, 93)` before preprocessing.
- `20` is exactly `input_window_seconds / feature_window_seconds = 1200 / 60`: one genuine one-minute feature window per temporal position.
- Preprocessing concatenates the feature-level validity mask, so sequence models receive `(N, 20, 186)`.
- Positions are generated by `minute_window_slices` in increasing sample order, so position `0` is the oldest minute and position `19` is the newest minute in the current 20-minute input window.
- There is no temporal padding or variable-length sequence path in the extracted-feature cache. Missingness is feature-level missingness inside genuine minute windows and is represented by the appended validity indicators. Because temporal padding is absent, Transformer `src_key_padding_mask`, GRU packed sequences/last-valid indexing, and TCN temporal-padding masks are not needed for the current framework.
- The input window is centered on the repository anchor (`anchor_time - 600s` to `anchor_time + 600s`). Regression `t+0m_gap` targets begin after the input window; filtered classification targets also start at the input end. Thus the sequence inputs are historical/current measurements available before the prediction outcome window, and Transformer bidirectional attention over the observed 20-minute input does not introduce target leakage.
- GRU, Transformer, TCN, full-sequence MLP, and full-sequence XGBoost all start from the same train-only imputed/scaled sequence representation. Current-state XGBoost uses the last token of that representation, and history XGBoost uses summaries of the same original value/mask cache.

Confirmed model behavior:

- Transformer positional encoding is learned, despite the historical `PositionalEncoding` class name. This is acceptable and was not renamed to avoid unnecessary churn. Positional capacity remains `max_seq_len + 1` when the CLS token is enabled, and the existing runtime error still catches longer-than-configured inputs.
- Transformer non-causal self-attention is acceptable here because all 20 tokens are observed input-history tokens. No causal attention mask was added.
- Transformer activation remains PyTorch's historical default rather than switching to GELU, because changing it would alter completed Transformer-run comparability and is not a correctness fix.
- GRU remains unidirectional and pools `x[:, -1]`, which is correct for oldest-to-newest fixed-length sequences with no temporal padding.
- GRU `dropout=0.1` still affects the prediction head for one-layer GRU runs. PyTorch recurrent dropout is only active for `gru_layers > 1`; this is documented behavior and was not changed because adding projection/input dropout would be an architecture change requiring reruns.
- TCN uses left-only `CausalConv1d` padding and final-timestep pooling. With defaults `kernel_size=3`, `dilations=[1,2,4]`, and two convolutions per residual block, the receptive field is `1 + 2 * (3 - 1) * (1 + 2 + 4) = 29` time steps, covering the full 20-minute sequence for the final representation.
- `ChannelLayerNorm` normalizes channels independently at each time step by transposing `(B,C,T) -> (B,T,C)`, applying `LayerNorm(C)`, and transposing back. It does not mix across time and does not leak future positions backward.
- Regression heads return raw scalar predictions. Classification training uses raw logits with `BCEWithLogitsLoss`; sigmoid probabilities are used only for metrics and saved probability predictions. Specificity at 85% sensitivity is computed by sweeping thresholds on the held-out test scores, matching the existing table convention. New Torch classification prediction exports also include raw `logits`.

Code hardening added:

- `SequenceModelConfig` now raises clear `ValueError`s for invalid dimensions, layer counts, dropout, sequence length, and Transformer head divisibility.
- The extracted-feature trainer validates sequence arrays as `(N, 20, F)` before GRU/Transformer/TCN training.
- Torch train DataLoader shuffling now uses a seeded `torch.Generator` in addition to the existing NumPy/Torch/CUDA RNG seeding. This improves reproducibility but does not claim bitwise GPU determinism.
- Metrics metadata for Torch models now records trainable parameter counts. TCN runs additionally record dilation schedule, receptive field, and whether the receptive field covers the input sequence.
- `test_predictions.npz` now includes stable `sample_ids` plus patient IDs, anchor times, row indices, target key, model type, and optional Torch logits; duplicate test sample IDs raise before saving. Existing patient IDs remain available for patient-clustered paired significance testing.

Default trainable parameter counts for `(N,20,186)` sequence inputs:

| Model | Default trainable parameters |
| --- | ---: |
| Transformer | 440193 |
| GRU | 139905 |
| TCN | 338049 |
| Full-sequence MLP | 2170369 |

Validation added/run:

- Added tests for sequence-model batch-size-1 output shape, finite outputs, gradient propagation, invalid config failures, Transformer positional capacity, GRU one-layer and multi-layer construction, TCN sequence-length preservation, TCN receptive-field metadata, TCN causality, `ChannelLayerNorm` per-timestep behavior, sequence-array shape validation, MLP invalid config failures, and prediction-export sample IDs/logits.
- Passed: `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m py_compile scripts/train_feature_models.py waveform_baselines/wf_features/models.py tests/test_waveform_feature_pipeline.py`.
- Passed: `/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python -m unittest tests.test_waveform_feature_pipeline` (`70` tests).

Changes that affect future run artifacts:

- New runs will have richer `metrics.json` and `test_predictions.npz` metadata.
- Existing completed GRU/Transformer/MLP metrics are not silently reinterpreted; their architectures were not changed.
- The only behavior changes are clearer invalid-config failures, seeded Torch DataLoader shuffling, parameter metadata, TCN metadata, and optional saved logits for Torch classification runs. Existing reported comparisons do not need reruns for architecture comparability, but rerunning is useful if the new metadata/logit fields are required for every historical artifact.

## Jobs

- Completed TCN-only PatchTST-relative significance jobs: regression `26919471` and classification `26919472`; outputs are `outputs/feature_models/*_tcn_only_2026-08-29.{json,csv,md}`.

- Completed `tcn` GPU training/export jobs `26919397`-`26919423` for all 26 regression targets plus filtered `5m` hypotension classification; outputs are under `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/featureExtractedModels`.

- Submitted `full_sequence_mlp` GPU jobs `26919162`-`26919188` to generate regression/classification prediction exports for downstream significance testing.

- Submitted `full_sequence_xgb` jobs `26919125`-`26919151` to generate regression/classification prediction exports for downstream significance testing.
