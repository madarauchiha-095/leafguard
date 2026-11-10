# Reproducibility Environment Report

Manifest generated at 2026-09-24T15:13:21.627789+00:00.

## Runtime

- Python: `3.14.3` (CPython)
- OS/architecture: Windows 11 / AMD64
- PyTorch / torchvision / timm: 2.11.0+cu128 / 0.26.0+cu128 / 1.0.30
- CUDA: `12.8`; cuDNN: `91900`; GPU: `NVIDIA GeForce RTX 3050 Laptop GPU`
- Dependency pins checked: 77/81 match the installed environment.
- Mismatches and missing packages: `{"datasets": {"requirements_entry": "datasets==5.0.1", "installed_version": "3.6.0", "requirement_status": "MISMATCH", "requirement_matches_installed": false}, "dill": {"requirements_entry": "dill==0.4.1", "installed_version": "0.3.8", "requirement_status": "MISMATCH", "requirement_matches_installed": false}, "fsspec": {"requirements_entry": "fsspec==2026.6.0", "installed_version": "2025.3.0", "requirement_status": "MISMATCH", "requirement_matches_installed": false}, "multiprocess": {"requirements_entry": "multiprocess==0.70.19", "installed_version": "0.70.16", "requirement_status": "MISMATCH", "requirement_matches_installed": false}}`.
- The complete declared and installed package versions are recorded in `research/reproducibility_manifest.json`.
- Workspace Git revision is unavailable because the project directory has no `.git` metadata.

## Data and model provenance

- PlantVillage-derived source provenance/license were not available in project metadata; this is an unresolved reproducibility warning.
- PlantDoc: commit `5467f6012d78d1c446145d5f582da6096f852ae8`, test split only, 83 mapped images in 9 classes; prepared file hashes match the pinned source manifest.
- PlantSeg v7: official Zenodo record, 143 CC-BY-NC-only mapped test images in 7 classes; archive and prediction audit statuses are VERIFIED / VERIFIED.
- All four checkpoint digests, byte sizes, parameter counts, and preprocessing settings are listed in the JSON manifest.
- The two DeiT-containing training scripts set PyTorch/CUDA seeds to 42. The MobileNet scripts did not set explicit seeds; no checkpoint was retrained.

## Evaluation reproducibility

- PlantVillage verification uses batch 32; PlantDoc and PlantSeg use batch 16; resource measurement is batch 1.
- Inputs resize to 224 by 224 and use ImageNet normalization. CLAHE is applied only for the CLAHE checkpoint (LAB L, clip 2.0, 8x8 grid).
- Resource latency is median CUDA-event model-forward time from three 60-forward trials after eight warmups; preprocessing and decode are excluded.
- Reproduction commands are included in the manifest. Training commands are intentionally omitted: the verified study reuses frozen checkpoints.

## Limits

The requirements file does not exactly match the active environment for four pinned packages (`datasets`, `dill`, `fsspec`, and `multiprocess`); the exact required and installed versions are listed in the manifest. The evaluated scripts completed in this environment, but rebuilding from requirements alone would not recreate the audited runtime exactly.
The exact PlantVillage source cannot be reconstructed from the current metadata, two training runs lack recorded seeds, external labels require expert review, PlantSeg's mapped subset has mixed plant-organ/graphic content and only seven disease classes, and GPU timing varies across separate runs. These limits do not invalidate the recorded held-out inference calculations, but they constrain exact training reproduction and broad generalization claims.
