# Master results across controlled and external test sets

External metrics average over each dataset's mapped classes. Domain gaps compare each external result with PlantVillage restricted to the same true classes. All entries are from frozen checkpoints; no retraining was performed.

| Model | PV accuracy (10) | PV macro F1 (10) | PlantDoc accuracy / F1 (9) | PlantSeg accuracy / F1 (7) | PlantDoc acc / F1 gap | PlantSeg acc / F1 gap | Parameters | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV3-Small | 99.51% | 99.60% | 49.40% / 42.32% | 21.68% / 15.10% | 50.08 / 57.23 pp | 77.68 / 84.37 pp | 1,528,106 | 6.88 |
| MobileNetV3-Small + CLAHE | 99.56% | 99.61% | 40.96% / 38.83% | 23.08% / 15.89% | 58.56 / 60.74 pp | 76.28 / 83.55 pp | 1,528,106 | 7.82 |
| DeiT-Tiny | 99.66% | 99.72% | 50.60% / 46.15% | 23.08% / 18.62% | 49.03 / 53.53 pp | 76.43 / 80.97 pp | 5,526,346 | 7.90 |
| MobileNetV3-Small + DeiT-Tiny Hybrid | 99.66% | 99.74% | 49.40% / 42.39% | 20.98% / 15.55% | 50.24 / 57.33 pp | 78.52 / 84.08 pp | 6,980,586 | 15.74 |

Accuracy and macro-F1 95% bootstrap intervals are in `results/metrics/master_results.csv` as low/high columns; domain-gap intervals are also recorded there. The bootstrap conditions on each dataset's observed class supports and does not address source selection.

Point estimates alone do not establish a ranking. Paired bootstrap intervals and exact McNemar tests for the same external images are in `results/metrics/paired_model_comparisons.csv` and the full JSON artifacts.

PlantVillage values in the first two columns use all ten classes. The same-class PlantVillage baselines used to calculate each domain gap are kept separately in the CSV and JSON. PlantDoc and PlantSeg overlap on different classes and have different sample supports, so their external scores should not be compared as if they were the same test distribution.
