# Segment-Level Vasopressor-Free Waveform Manifest

This manifest labels MIMIC-III matched waveform segments, not patients, ICU stays, or parent WFDB records. Parent WFDB records are used only to validate/reconstruct segment timing.

## Outputs

- Complete manifest: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/full_data_segment_level_vasopressor_free_waveform_manifest.csv`
- QC report: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/full_data_segment_level_vasopressor_free_waveform_manifest.qc.json`
- Convenience free-segment list: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/full_data_segment_level_vasopressor_free_free_segments.txt`
- Confirmed continuous intervals: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/full_data_segment_level_vasopressor_free_vasopressor_intervals.csv`
- Uncertain vasopressor evidence: `/gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/full_data_segment_level_vasopressor_free_uncertain_vasopressor_evidence.csv`

Older `full_data_record_level_*` outputs also exist in the manifest directory from intermediate runs. Use the `full_data_segment_level_*` outputs above for downstream full-data feature extraction.

## Completed Full-Data Manifest Build (2026-08-29)

Command pattern:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python \
  scripts/build_record_level_vasopressor_free_manifest.py \
  --segment-metadata-json /gpfs/data/eh3828lab/derived_datasets/physionet_restricted/mimic_derived_data/data_m3_120s_prediction/segment_metadata.json \
  --output /gpfs/data/eh3828lab/derived_datasets/baselines/waveformBaselines/manifests/full_data_segment_level_vasopressor_free_waveform_manifest.csv
```

The generated QC reports:

- total waveform segments: `32044`
- valid timestamp segments: `32044`
- metadata/WFDB timestamp mismatches: `0`
- ICU matches: `30492` `matched_full`, `120` `matched_partial`, `1432` `unmatched`
- segments with confirmed vasopressor overlap: `8658`
- segments classified high-confidence vasopressor-free: `21833`
- unknown-classification segments: `1553`
- percentage vasopressor-free among classified segments: `71.6047%`
- CareVue raw rows: `1724783`
- CareVue clinical-drug episodes: `34846`
- CareVue any-vasopressor merged episodes: `24696`
- MetaVision raw rows: `215200`
- MetaVision retained intervals: `40349`
- continuous confirmed interval rows exported: `38577`
- uncertain CareVue rows exported: `3165`
- uncertain MetaVision rows exported: `75`
- CareVue same-drug timestamp stop+active conflicts resolved: `967`
- CareVue same-drug timestamp zero+positive conflicts resolved: `141`

Focused tests passed after the final hardening pass:

```bash
/gpfs/data/eh3828lab/derived_datasets/baselines/conda/myenv/bin/python \
  -m unittest tests.test_record_level_vasopressor_free_manifest
```

Result: `30` tests passed.

## Definition

`vasopressor_free=True` means:

- segment timing is valid;
- metadata and WFDB-derived timestamps agree within the configured tolerance when both are available;
- the segment is fully contained within one unambiguous ICU stay using the configured absolute containment tolerance;
- no confirmed reconstructed vasopressor interval overlaps the segment;
- no unresolved CareVue or MetaVision evidence prevents confident classification.

The interval overlap convention is half-open:

```python
start_a < end_b and end_a > start_b
```

`matched_partial`, unmatched ICU linkage, ambiguous ICU linkage, timestamp mismatch, uncertain CareVue evidence, uncertain MetaVision evidence, untimed same-ICU evidence, untimed same-admission evidence, and conservative patient-only untimed evidence all remain `vasopressor_free=NA`.

## Vasopressor Reconstruction

The configured drug set intentionally includes norepinephrine, epinephrine, phenylephrine, dopamine, dobutamine, milrinone, and vasopressin. This is broader than some generic vasopressor-duration examples and should not be silently narrowed.

MetaVision uses the established ITEMIDs, excludes `STATUSDESCRIPTION == "Rewritten"`, reconstructs valid LINKORDERID intervals from `STARTTIME`/`ENDTIME`, and preserves unresolved retained rows as uncertain evidence.

CareVue maps raw ITEMIDs to clinical drug names before reconstruction. Rate/stop transitions are reconstructed per:

```text
ICUSTAY_ID + vasopressor_name
```

not per raw ITEMID. This prevents artificial gaps when charting switches between ITEMIDs for the same drug, such as `30047 -> 30120` for norepinephrine or `30044 -> 30119 -> 30309` for epinephrine. Contributing raw ITEMIDs are retained in `source_itemids`. Duplicate same-drug rows at the same timestamp use max RATE as the representative episode state, matching the existing MIMIC-derived reconstruction style; rates are not summed. If a same-ICU, same-drug timestamp has both a stop marker and a non-stopped positive-rate row, the clinical drug state remains active. A zero-rate row also does not stop the drug when another same-drug row at that timestamp has active positive-rate evidence. This conflict resolution is scoped to `ICUSTAY_ID + CHARTTIME + vasopressor_name`, not across different vasopressor drugs.

Candidate CareVue episodes with positive-rate evidence but unreliable duration are exported as uncertain evidence rather than converted to arbitrary fixed-duration intervals.

## Future Window Filtering

The segment-level label is a coarse prefilter/QC field. Final feature-window filtering should classify each actual extraction window using both exported files:

```text
confirmed interval overlaps window -> vasopressor_free=False
no confirmed overlap, but uncertain evidence could correspond -> vasopressor_free=NA / exclude
no confirmed overlap and no relevant uncertainty -> vasopressor_free=True
```

Do not determine window-level vasopressor freedom from confirmed continuous intervals alone.
