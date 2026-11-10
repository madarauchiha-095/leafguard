# Robustness comparison

| Frozen model / external set | PV accuracy (same classes) | External accuracy | Accuracy gap | PV macro F1 (same classes) | External macro F1 | Macro-F1 gap |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV3-Small on PlantDoc | 99.48% | 49.40% | 50.08 pp | 99.55% | 42.32% | 57.23 pp |
| MobileNetV3-Small on PlantSeg v7 | 99.36% | 21.68% | 77.68 pp | 99.46% | 15.10% | 84.37 pp |
| MobileNetV3-Small + CLAHE on PlantDoc | 99.53% | 40.96% | 58.56 pp | 99.57% | 38.83% | 60.74 pp |
| MobileNetV3-Small + CLAHE on PlantSeg v7 | 99.36% | 23.08% | 76.28 pp | 99.44% | 15.89% | 83.55 pp |
| DeiT-Tiny on PlantDoc | 99.63% | 50.60% | 49.03 pp | 99.68% | 46.15% | 53.53 pp |
| DeiT-Tiny on PlantSeg v7 | 99.50% | 23.08% | 76.43 pp | 99.59% | 18.62% | 80.97 pp |
| MobileNetV3-Small + DeiT-Tiny Hybrid on PlantDoc | 99.63% | 49.40% | 50.24 pp | 99.71% | 42.39% | 57.33 pp |
| MobileNetV3-Small + DeiT-Tiny Hybrid on PlantSeg v7 | 99.50% | 20.98% | 78.52 pp | 99.63% | 15.55% | 84.08 pp |

These are frozen-checkpoint comparisons, not intervention results. The PlantVillage baseline is restricted to the same mapped classes as each external test. Gaps are descriptive; independent stratified-bootstrap intervals are available in `results/metrics/master_results.csv`.

No controlled augmentation or training intervention was performed because both available external datasets are now evaluation evidence. Using their observed test failures to select a training modification would convert them into development data and leave no independent external evaluation set.
