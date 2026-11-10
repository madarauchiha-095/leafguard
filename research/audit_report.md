# Phase 1 Reproducibility Audit

**Stage:** current implementation

No training was performed.

## Overall checks

- **VERIFIED** `plantvillage_splits/train/count` — Found 7647 images; expected 7647.
- **VERIFIED** `plantvillage_splits/train/class_folders` — ImageFolder class order compared to the model's fixed output order.
- **VERIFIED** `plantvillage_splits/val/count` — Found 845 images; expected 845.
- **VERIFIED** `plantvillage_splits/val/class_folders` — ImageFolder class order compared to the model's fixed output order.
- **VERIFIED** `plantvillage_splits/test/count` — Found 2054 images; expected 2054.
- **VERIFIED** `plantvillage_splits/test/class_folders` — ImageFolder class order compared to the model's fixed output order.
- **VERIFIED** `plantvillage_splits/test/per_class_support` — Counts are folder-label supports; semantic image-label correctness was not independently adjudicated.
- **NOT TESTED** `plantvillage_splits/test/semantic_label_review` — Folder labels were audited mechanically; no independent agronomist relabeling was performed.
- **WARNING** `original_training_seeds` — Training-script seed configuration inspected without launching training. Unseeded legacy runs cannot be reconstructed from checkpoint files alone.
- **VERIFIED** `plantdoc_inventory/sample_count` — Prepared ImageFolder count and both source manifests agree.
- **VERIFIED** `plantdoc_inventory/source_split` — Materialization manifest split declaration.
- **VERIFIED** `plantdoc_inventory/per_class_counts` — Prepared folder counts compared with preparation manifest.
- **VERIFIED** `plantdoc_inventory/provenance_hashes` — All prepared images compared byte-for-byte with pinned source SHA-256 entries.
- **VERIFIED** `plantdoc_inventory/label_mapping_code_agreement` — Manifest mappings compared with the current explicit label map.
- **WARNING** `plantdoc_inventory/semantic_label_review` — Mapping rules are explicit and reproducible, but PlantDoc labels were not independently expert-adjudicated.
- **VERIFIED** `plantdoc_inventory/plantdoc_training_images_used` — The source split declaration and every recorded source path must be test-only.
- **VERIFIED** `plantdoc_inventory/duplicate_within_external` — Exact bytewise SHA-256 duplicate scan within mapped PlantDoc subset.
- **VERIFIED** `external_preprocessing_and_pairing/checkpoint_specific_clahe` — PlantDoc evaluator must apply each MODEL_SPECS CLAHE flag. False means the CLAHE checkpoint is receiving baseline preprocessing.
- **VERIFIED** `external_preprocessing_and_pairing/same_external_samples` — Runner compares each per-model ImageFolder sample list to one shared source inventory.
- **VERIFIED** `external_preprocessing_and_pairing/plantvillage_checkpoint_preprocessing` — PlantVillage verification runner applies the preprocessing flag stored for each checkpoint.
- **VERIFIED** `checkpoint_and_resource_integrity/four_checkpoints` — All four named model specs were strictly loaded without training.
- **VERIFIED** `checkpoint_and_resource_integrity/hybrid_expected_sha256` — Hybrid digest compared with configs/model_integrity.json.
- **VERIFIED** `checkpoint_and_resource_integrity/checkpoint_associations` — Saved evaluation records identify the same checkpoint path loaded from each MODEL_SPECS entry.
- **VERIFIED** `checkpoint_and_resource_integrity/parameter_count_and_file_size` — Recomputed parameter counts and file byte sizes compared with final_results.json.
- **VERIFIED** `checkpoint_and_resource_integrity/latency_repeatability` — Three 60-iteration batch-one model-forward timing trials after eight warmups; CUDA events measure GPU execution. Saved median and trial coefficient of variation are validated. CV <10% is the repeatability criterion.
- **WARNING** `checkpoint_and_resource_integrity/latency_cross_run_consistency` — Comparison of independent audit-run medians with the saved report-run medians; >10% deviation is reported as GPU/runtime variability even when each run's internal trials are stable.
- **VERIFIED** `paper_numeric_values` — Accuracy, macro-F1, domain gaps, parameter counts, checkpoint sizes, and displayed latency cells are checked against current artifacts; results are valid only when checkpoint-specific preprocessing also passes.
- **VERIFIED** `bootstrap` — 10,000 stratified multinomial resamples regenerated from saved confusion matrices and compared with recorded percentile intervals.
- **VERIFIED** `gradcampp/stored_outputs` — Existing Grad-CAM++ PNG dimensions and non-empty files checked.
- **VERIFIED** `gradcampp/fresh_smoke` — Fresh hybrid attribution forward pass; verifies both branch maps, finite/nonconstant maps, and prediction confidence parity.
- **VERIFIED** `confusion_matrices_and_metrics` — PV and PlantDoc supports, full 10-output confusion matrix dimensions, and accuracy/macro P/R/F1 reconstructed independently from matrix counts.
- **VERIFIED** `unsupported_predictions` — Full ten-column external confusion matrices retain unsupported model outputs; they are not discarded from accuracy or mapped-class precision.

## Checkpoint and model resource audit

| Model | SHA-256 | Bytes | Parameters | Repeated latency trials (ms/image) | Saved latency delta |
|---|---|---:|---:|---|---:|
| mobilenet | `33194968c50eb807a02fab2fe46088de71a99100d61bbcb0ce63b3023b6c9fcc` | 18599019 | 1,528,106 | 6.541, 7.172, 6.872 | 0.0016 |
| mobilenet_clahe | `2060a5a75cac87eaf307452ba6b28f52386a9288dd4dab413694eaf397347966` | 18603267 | 1,528,106 | 7.707, 6.647, 7.443 | 0.0487 |
| deit_tiny | `c64dac118ca113edfc2f2f23af6d7f578a01e53087d285ccf460187995f1592c` | 22167777 | 5,526,346 | 7.620, 6.944, 6.286 | 0.1206 |
| hybrid | `116be826ae3ccb601de4a7259ca06ed137c6f413879edc5e9438e9a9f25abb00` | 28132099 | 6,980,586 | 14.552, 13.407, 14.592 | 0.0755 |

## Independently reconstructed metrics

Metrics below are regenerated from the saved confusion matrices. Accuracy uses all samples; macro P/R/F1 are averaged over ten PlantVillage classes or nine mapped PlantDoc true classes, respectively.

| Model | PV accuracy / macro F1 | PlantDoc n | PlantDoc accuracy / macro F1 | Same-class accuracy / F1 gap |
|---|---:|---:|---:|---:|
| mobilenet | 99.5131% / 99.5952% | 83 | 49.3976% / 42.3196% | 50.0783% / 57.2307% |
| mobilenet_clahe | 99.5618% / 99.6089% | 83 | 40.9639% / 38.8300% | 58.5644% / 60.7355% |
| deit_tiny | 99.6592% / 99.7155% | 83 | 50.6024% / 46.1512% | 49.0307% / 53.5327% |
| hybrid | 99.6592% / 99.7420% | 83 | 49.3976% / 42.3859% | 50.2355% / 57.3274% |

## Discrepancies and limits

- **ERROR FOUND THEN; CORRECTION STATUS: VERIFIED:** pre-correction PlantDoc evaluation skipped CLAHE for `mobilenet_clahe`. Its old scores (30.1205% accuracy, 23.5108% macro F1) are preserved in `results/audit/pre_correction_domain_shift/` and superseded. The manuscript and active result files must use only the corrected rerun.
- Image-folder class consistency and manifests were verified mechanically; original PlantVillage annotations and PlantDoc label semantics were not independently adjudicated.
- Cross-dataset duplication was checked by exact SHA-256 only. Perceptual near-duplicates were not tested.
- Latency was measured in three short repeated trials on the recorded device. This does not establish hardware-independent or long-run timing repeatability.
- Saved-run and independent audit-run latency medians are compared separately; >10% drift indicates runtime/device variability and is not hidden by within-run trial stability.
- Grad-CAM++ maps were checked for finite, nonconstant branch outputs and confidence parity; this validates execution, not explanation faithfulness or localization.
- The PlantDoc subset remains only 83 images. Bootstrap intervals condition on these samples and do not resolve dataset selection or mapping uncertainty.

Machine-readable details: `results/audit/reproducibility_audit.json`.
