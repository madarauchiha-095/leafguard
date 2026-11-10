# Methodology

## Datasets

The in-domain dataset is a ten-class PlantVillage-derived subset at 224×224 input resolution. The project records 7,647 train, 845 validation, and 2,054 held-out test images. The existing test split is preserved.

PlantDoc is the field-oriented external test source. The original paper reports 2,598 images, 13 plant species, and 27 categories; repository variants differ in image availability and label count. For pinned source commit `5467f6012d78d1c446145d5f582da6096f852ae8`, the official test split contains 236 images across 27 folders. The upstream repository states CC BY 4.0. Only the 83 test images in nine reliably mapped classes (20,510,778 bytes) were materialized; no training images were obtained. These internet-sourced, curated/cropped samples have more varied capture conditions than PlantVillage but are not a representative field survey.

## Label mapping and external split

Only explicit disease labels with compatible crop and symptom names are mapped. Generic “leaf” classes for grape and tomato are treated as healthy because those are PlantDoc's unqualified leaf categories; generic apple leaf is left unmapped because the class is not sufficiently specific for a reliable health label. Labels with other diseases, crops, pests, or ambiguous naming are excluded. The full mapping and exclusions are recorded in `research/class_mapping.md` and implemented in `src/data/prepare_plantdoc.py`.

The materialization and preparation scripts select only the upstream PlantDoc `test/` split and copy eligible images into `data/external/plantdoc_prepared/test/<PlantVillage class>/`. They do not read or modify PlantVillage data. The untouched ten-class checkpoint is evaluated on the mapped samples; predictions into other output classes count as errors. Macro precision, recall, and F1 average over mapped true classes only, and the confusion matrix retains the model's full ten-class output space.

## Models and preprocessing

The four existing checkpoints are loaded without retraining. Inputs are resized to 224×224 and normalized with ImageNet mean and standard deviation. The CLAHE checkpoint receives LAB lightness CLAHE after resize (clip limit 2.0; 8×8 tile grid), matching the legacy evaluation preprocessing. No preprocessing or class definitions are changed for the original results.

## Evaluation

The verification script evaluates each checkpoint on the full existing PlantVillage test set and saves new outputs under `results/audit/`, then compares those metrics with the saved legacy result JSONs. Cross-domain outputs, when available, are written to `results/domain_shift/` as per-model JSON and confusion-matrix PNG files. Accuracy, macro precision, macro recall, macro F1, per-class metrics, and confusion matrices are recorded.

Model resources are reported as total parameter count, checkpoint file size, and batch-one forward latency measured on the available device after warm-up. PyTorch profiler FLOPs are approximate and may omit operators. No additional hybrid+CLAHE training is conducted.

## Grad-CAM++

For CNN feature maps, Grad-CAM++ targets the last MobileNet feature block. For DeiT-Tiny, it targets the final transformer block's `norm1` output, drops the class token, and reshapes the 196 patch tokens into a 14×14 spatial grid. In the hybrid, each branch's target-class map is calculated from gradients of the shared output logit; each branch map is normalized separately and their mean is used for a combined display. Separate branch maps are retained for inspection. Examples include correct and difficult/confused PlantVillage predictions and correct and failed PlantDoc predictions. A separate PlantDoc object-detection release provides leaf bounding boxes, but this work did not establish correspondence between those annotations and the selected cropped-classification images; no localization score is valid here.

## Reproducibility

The inspected DeiT-Tiny and hybrid training scripts set PyTorch and CUDA seeds to 42. The MobileNetV3 and MobileNetV3+CLAHE scripts do not set an explicit Python, NumPy, or PyTorch random seed, so their original stochastic training runs cannot be exactly reconstructed from the available code and checkpoints. This work does not retrain any model. Evaluation uses deterministic, non-shuffled test loaders. The hybrid checkpoint SHA-256 is recorded for change detection; this is not a digital signature.
