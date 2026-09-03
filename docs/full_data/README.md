# Full-Data Segment-Aware Cohort

This folder tracks the full-data `data_m3_120s_prediction` workstream. It covers the segment-aware vasopressor-free manifest, full-data feature cache, full-data regression/classification targets, extracted-feature model results, and the newer full-data raw-waveform PatchTST jobs.

Use this folder when working with `anchor_id`/`segment_id` aligned full-data artifacts or the large full-data SLURM batches.

Key docs:

- `full_data_new_windows.md`: full-data cohort construction and feature-cache workflow.
- `record_level_vasopressor_free_manifest.md`: segment-level vasopressor-free manifest reconstruction and QC.
- `extractedFeaturesRegressionFullData.md`: full-data extracted-feature regression workflow and results.
- `extractedFeaturesClassificationFullData.md`: full-data extracted-feature classification targets, submissions, and results.
- `full_data_raw_waveform_models.md`: full-data 4-channel raw-waveform `patchtst_v1` regression/classification jobs.
- `full_data_raw_waveform_models_vasopressor_present.md`: full-data 4-channel raw-waveform `patchtst_v1` setup for confirmed vasopressor-overlap segments.
- `full_data_30s_waveform_models.md`: full-data 4-channel trailing-30-second raw-waveform `patchtst_v1_5` setup and experiments.
