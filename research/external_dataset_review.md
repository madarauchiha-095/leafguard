# External Dataset Review

**Review date:** 2026-09-24  
**Scope:** Public/research-accessible image datasets that could support a second external test of the four frozen LeafGuard checkpoints. The initial review used public documentation only; following that review, the official PlantSeg v7 archive was downloaded for metadata, integrity, licensing, and overlap checks.

## Decision

**PlantSeg v7 was selected for a second external evaluation.** The official archive is 1,057,281,724 bytes and passed both the official Zenodo MD5 check (`9358a66dff88cdd15c4fe009763c40a3`) and ZIP CRC verification. `Metadata.csv` contains 7,774 images (5,367 train, 846 validation, 1,561 test), consistent with the latest paper. Seven exact disease labels overlap; their source counts total 735. The frozen test split contains 145 target-class candidates; two are marked `CC0`. Before evaluating any checkpoint, the project froze a CC-BY-NC-only cohort to keep one per-image license rule, leaving 143 images in seven classes. The exclusion was made on license metadata, before model outputs were generated.

This is suitable for a *mapped-class external stress test*, not a representative estimate of farm performance. PlantSeg images were collected from web sources and are not all necessarily farm photographs. The selected image rows are marked `CC-BY-NC` in the v7 metadata, and the paper documents non-commercial dataset terms and per-image source URLs/licenses. The general PlantWild landing page also mentions CC BY-NC-ND, so the scope difference between the PlantSeg v7 data record and the broader PlantWild statement must remain documented; the study uses the data only for non-commercial evaluation and does not redistribute source images. SHA-256 and perceptual-hash screening found no exact duplicates and no pHash flags at Hamming distance <=4 against PlantVillage test or PlantDoc. This supports distinct image samples under the checks used; it cannot prove source-population independence. A local visual review found both leaf and fruit/whole-plant views, varied framing/resolution, watermarks, and some overlaid text/graphics. Therefore, describe PlantSeg as web-sourced heterogeneous plant-disease imagery, not as a leaf-only field sample. Keep the derived source images out of distributed figures unless reuse rights are confirmed.

## Candidate comparison

| Dataset | Public/research access and license | Size and label overlap | Image conditions and independence | Decision |
|---|---|---|---|---|
| **PlantSeg v7 (selected)** | Official Zenodo v7, 1,057,281,724 bytes. The current paper identifies CC BY-NC 4.0 dataset terms and metadata contains per-image `CC-BY-NC`/`CC0` entries. The PlantWild landing page has a broader CC BY-NC-ND statement; do not redistribute any images. Evaluation is limited to rows marked `CC-BY-NC` for non-commercial research. | Verified `Metadata.csv`: 7,774 images (5,367 train / 846 validation / 1,561 test). Exact compatible full-source counts: apple scab 168, apple black rot 63, grape black rot 88, potato early blight 68, potato late blight 78, tomato early blight 153, tomato late blight 117 (735). Frozen test candidates: 145; after excluding two `CC0` rows, 143 remain: apple scab 34, apple black rot 12, grape black rot 17, potato early blight 9, potato late blight 18, tomato early blight 34, tomato late blight 19. Corn rust is excluded as ambiguous; healthy mappings are absent. | Web-sourced imagery, with complex backgrounds, varied viewpoints, lighting, and resolution described by the authors. Official source URLs and expert-reviewed annotations are available. SHA-256 and pHash checks found 0 exact duplicates and 0 pHash<=4 flags against PlantVillage test and PlantDoc. This cannot establish full source-population independence. | **Proceed with the mapped-class evaluation.** Archive checksum and split verified; same 143-image inventory will be used for all four checkpoints. |
| **FieldPlant** | Original peer-reviewed dataset is documented as a public field-image resource; an official-looking Roboflow Universe mirror lists CC BY 4.0. The available mirror's current class taxonomy/version is not an exact, stable reproduction of the original dataset and shows all 5,156 images in its training split. | Published original: 5,170 images, 27 classes, three crop species (corn, cassava, tomato), 8,629 annotated leaves. This suggests possible overlap on corn rust and some tomato classes, but the ready mirror's classes are detection labels (e.g., `Corn_Rust`, `Tomato_Mildiou`, `Tomato_healthy`) and do not provide classwise counts or a test split suitable for this LeafGuard classification protocol. | Authors report direct plantation capture in Cameroon and expert-supervised leaf annotations: strong visual and geographic distinction from PlantVillage. Images can contain multiple leaves; image-level classification would require a pre-registered crop/label protocol and careful grouping to prevent crops from the same source image crossing splits. | **Not selected for this run.** Strong future candidate, but current accessible artifact's label/split lineage needs resolving before it can serve as a reproducible fixed test set. |
| **Plant Pathology 2020 / FGVC7** | Official Kaggle rules allow academic research use and identify Apache 2.0 terms, but prohibit transmitting/duplicating/publishing/redistributing competition data outside participants; Kaggle registration and acceptance are required. | Source authors report 3,651 real-life apple foliar images; the challenge subset has apple scab, cedar apple rust, and healthy labels. Only Apple scab maps exactly to LeafGuard. Kaggle's challenge test labels are hidden, so its official leaderboard data do not provide a locally auditable labeled test set. | Images were manually captured in apple orchards under variable illumination, angles, surfaces, and noise. Distinct from PlantVillage, but one compatible labeled class and hidden challenge labels prevent a robust independent comparison. | **Do not use.** Insufficient labeled class overlap for the stated multi-class analysis, with redistribution and label-access constraints. |
| **PlantWild / PlantWild v2** | Official project download page provides access links. The page states CC BY-NC-ND 4.0; v2 is expert-refined and expanded to 115 classes. | The 2024 paper documents v1 as 18,542 images/89 classes (56 diseased and 33 healthy); v2 has 115 classes, but the official landing page does not publish a classwise count table. PlantSeg is an associated segmentation dataset built on PlantWild imagery, so it is not an independent second source relative to PlantSeg. | In-the-wild visual conditions; internet-source and possible cross-dataset overlap concerns remain. No official stable classification test split and class counts were established during this review. | **Do not add as another dataset.** PlantSeg is the stronger, documented artifact; PlantWild may overlap it and has the more restrictive ND terms. |
| **AI Challenger 2018** | A public/research license applicable to the exact image release was not established from a primary source during this review. | Reported versions and class counts vary across secondary sources; exact LeafGuard overlap and usable held-out image totals were not established. | Acquisition provenance and overlap with common web datasets remain uncertain. | **Do not use.** Access, license, split, and mapping are not sufficiently documented. |

## PlantSeg mapping and pre-evaluation gates

| PlantSeg source label | Full source count | LeafGuard output | Mapping decision |
|---|---:|---|---|
| apple scab | 168 | `Apple___Apple_scab` | Include if image-level license supports non-commercial research and sample is in official test split. |
| apple black rot | 63 | `Apple___Black_rot` | Include under the same gates. |
| grape black rot | 88 | `Grape___Black_rot` | Include under the same gates. |
| potato early blight | 68 | `Potato___Early_blight` | Include under the same gates. |
| potato late blight | 78 | `Potato___Late_blight` | Include under the same gates. |
| tomato early blight | 153 | `Tomato___Early_blight` | Include under the same gates. |
| tomato late blight | 117 | `Tomato___Late_blight` | Include under the same gates. |
| corn rust | 120 | `Corn_(maize)___Common_rust_` | **Exclude unless expert review establishes the label means common rust specifically.** |
| grape healthy / tomato healthy | Not established in the current table | corresponding healthy output | **Do not infer a mapping from the disease-only class table; inspect official metadata.** |

Full-source counts above come from the latest paper's disease table. Exact test supports come from the frozen v7 `Metadata.csv`; no counts were approximated from percentages. Training and validation images are not used.

Before evaluating:

1. Use only Zenodo DOI `10.5281/zenodo.17719108` (v7); the downloaded archive is verified by official MD5 and ZIP CRC checks.
2. Use only `CC-BY-NC` target test records; two `CC0` target records are excluded conservatively. Retain each row's URL, license, filename, split, and SHA-256 in the audit JSON.
3. Use the frozen 143-image manifest after exact and perceptual-hash overlap checks; all four checkpoints receive the same IDs and checkpoint-specific preprocessing.
4. Keep the archive local and evaluate from ZIP members in memory. Do not copy or redistribute images. Publish the DOI, checksums, mapping, and scripts so others can obtain the source under its terms.
5. Review representative imagery for leaf/fruit/whole-plant content before interpreting the scores. Do not change mappings based on model outputs. A temporary contact-sheet review was used, then its image-bearing sheets were removed to avoid retaining or redistributing image derivatives; only the archive audit manifest remains.

## Completed second-dataset evaluation

All four existing checkpoints were evaluated once on the same 143-image frozen PlantSeg subset. No model was retrained or fine-tuned. The evaluator read images directly from ZIP members in memory and applied checkpoint-specific preprocessing, including LAB-lightness CLAHE only for the CLAHE checkpoint. Per-image predictions, confidence, correctness, ten-output confusion matrices, per-class metrics, CSV/JSON/Markdown summaries, and conditional bootstrap intervals are saved under `results/domain_shift/plantseg_v7/`. The separate prediction audit recomputed summary and per-class metrics from the 572 model-image rows, verified the 143 identical sample IDs and labels for each model, verified confusion matrices and checkpoint hashes, and passed as **VERIFIED**.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Accuracy 95% CI | Macro-F1 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV3-Small | 21.68% | 21.83% | 18.77% | 15.10% | 19.23% | [15.38%, 27.97%] | [10.55%, 19.66%] |
| MobileNetV3-Small + CLAHE | 23.08% | 28.70% | 19.19% | 15.89% | 17.14% | [18.18%, 28.67%] | [9.51%, 21.41%] |
| DeiT-Tiny | 23.08% | 33.95% | 22.19% | 18.62% | 19.20% | [17.48%, 29.37%] | [11.91%, 25.03%] |
| MobileNetV3-Small + DeiT-Tiny hybrid | 20.98% | 32.31% | 20.52% | 15.55% | 16.81% | [15.38%, 26.57%] | [10.01%, 21.35%] |

The class supports are apple scab 34, apple black rot 12, grape black rot 17, potato early blight 9, potato late blight 18, tomato early blight 34, and tomato late blight 19. Macro measures average over these seven mapped true classes; predictions to any of the other three model outputs count as errors. The intervals are stratified image-bootstrap intervals conditional on this selected sample and observed class supports; they do not account for dataset selection, mapping uncertainty, or source dependence.

Paired comparisons used the same image IDs for every model. DeiT-Tiny minus hybrid accuracy was 2.10 percentage points (paired stratified bootstrap 95% CI [-4.20, 8.39]); macro-F1 difference was 3.07 points (95% CI [-2.75, 9.39]). Exact McNemar comparison had 13 DeiT-only correct and 10 hybrid-only correct discordant cases (Holm-adjusted p=1.00 across the six exploratory comparisons). These observations do not establish a statistically reliable model ranking.

At a descriptive maximum-softmax threshold of 0.90, incorrect fractions among predictions at or above the threshold were 74.3% for MobileNetV3 (52/70), 67.6% for CLAHE (48/71), 73.3% for DeiT-Tiny (44/60), and 80.0% for the hybrid (52/65). Maximum softmax is not calibrated uncertainty; these fractions apply only to this sample and threshold.

## Visual and interpretation limits

The 143 PlantSeg samples visibly mix close leaf views with fruit-centered and whole-plant scenes; several contain watermarks, text, collages, or very small source resolution. This supports a description of heterogeneous web imagery, but does not support a causal subgroup analysis. We did not add subjective PlantSeg annotations or claim that any visual factor caused prediction errors. The single-reviewer PlantDoc factor annotations remain draft-only and need author review before paper use. The temporary PlantSeg contact sheets were deleted after review; no PlantSeg examples are embedded in the Grad-CAM figure because of the project's no-redistribution rule and the broader PlantWild license statement.

The selected Grad-CAM++ figure uses deterministic PlantVillage/PlantDoc examples for Groups A-D and is stored under `results/explainability/paired_examples/`. The examples show class-specific maps that vary by model and branch. They do not establish that the model attended to lesions; there are no verified lesion masks for these classification samples. Figure metadata records image IDs, true/predicted classes, confidence, source revision, branch maps, and the selection rule. PlantDoc image reuse in any public paper figure still requires appropriate dataset attribution and license review.

## Provenance and sources

- PlantSeg project, collection method and official download links: <https://github.com/tqwei05/PlantSeg> and <https://tqwei05.github.io/PlantWild/>.
- PlantSeg v7 archive, DOI `10.5281/zenodo.17719108`, 1.1 GB: <https://zenodo.org/records/17719108>.
- Machine-readable archive, split, license, hash, and overlap audit: `results/audit/plantseg_v7_archive_audit.json`; prediction recomputation and checkpoint integrity check: `results/audit/plantseg_v7_evaluation_audit.json`.
- Evaluation summary, per-image predictions, metrics CSVs, per-class CSV, confidence analysis, and bootstrap intervals: `results/domain_shift/plantseg_v7/`.
- Latest PlantSeg paper, source conditions, class counts, split and license discussion: <https://www.nature.com/articles/s41597-025-06513-4> (DOI: 10.1038/s41597-025-06513-4).
- FieldPlant paper: <https://doi.org/10.1109/ACCESS.2023.3263042>; available mirror: <https://universe.roboflow.com/project-59cph/fieldplant-cksi2-se8bt>.
- Plant Pathology 2020 source paper: <https://arxiv.org/abs/2004.11958>; official competition terms: <https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/rules>.
- PlantWild project page and license: <https://tqwei05.github.io/PlantWild/>; benchmark paper: <https://arxiv.org/abs/2409.04038>.

## Status

**PHASES 3-5 complete for PlantSeg; Phase 6 visual review is descriptive only.** The official archive passed MD5/CRC checks, the 143-image seven-class test inventory is frozen, all four frozen checkpoints completed, and prediction metrics were independently recomputed. PlantSeg is a heterogeneous web-image stress test, not a representative field benchmark. No training occurred. PlantDoc visual subgroup summaries remain exploratory and draft-only pending author review. Source images and image-bearing PlantSeg derivatives are not retained for redistribution.
