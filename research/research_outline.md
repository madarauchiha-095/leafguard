# Research outline

## Working title

Cross-domain evaluation and visual attribution for a lightweight heterogeneous CNN–Transformer plant disease classifier

## Research question

Does feature fusion between MobileNetV3-Small and DeiT-Tiny improve generalization from controlled PlantVillage imagery to field-oriented images, and do Grad-CAM++ maps provide useful visual evidence about model predictions?

## Scope

The work compares four already-trained checkpoints: MobileNetV3-Small, MobileNetV3-Small with CLAHE preprocessing, DeiT-Tiny, and a MobileNetV3-Small/DeiT-Tiny feature-fusion classifier. PlantVillage remains the training and in-domain test source. PlantDoc is considered as a separate out-of-domain evaluation set; it does not replace or enter the PlantVillage test split.

## Planned sections

1. Motivation and the limits of controlled-background leaf classification.
2. Related work on CNN and Transformer classifiers, feature fusion, plant disease datasets, and visual attribution.
3. Dataset construction and explicit cross-dataset label mapping.
4. Frozen-checkpoint evaluation and model resource measurements.
5. PlantVillage-to-PlantDoc accuracy and macro-F1 gaps, if the field evaluation is completed.
6. Grad-CAM++ examples from both hybrid branches and failure cases.
7. Limitations, reproducibility details, and next experiments.

## Claims boundary

This study does not claim that CNN–Transformer fusion is novel by itself. It will claim increased robustness only if the cross-domain comparison supports that statement. PlantVillage scores alone do not support field deployment claims. Grad-CAM++ maps will be described as qualitative attribution unless valid region annotations are available.

## Current status

The checkpoints and PlantVillage test split are present and all four PlantVillage results have been re-verified. The mapped PlantDoc test subset (83 images, nine classes) has been prepared and evaluated. The observed cross-domain results do not support a claim that the hybrid is more robust than DeiT-Tiny.
