# LeafGuard final results

## Dataset

PlantVillage: 10 selected classes; reported split counts 7,647 train, 845 validation, and 2,054 test. The held-out test set was re-evaluated from the existing checkpoints. PlantDoc: 236 upstream test images were listed, of which 83 mapped images from nine compatible classes were prepared from pinned commit 5467f6012d78. The mapped subset totals 20,510,778 bytes.

## Experimental setup

All four frozen checkpoints were loaded with their original 224×224 preprocessing; the CLAHE checkpoint uses LAB lightness CLAHE (clip limit 2.0, 8×8 tiles). No training was run. Verified outputs are saved in `results/audit/`.

## Baseline, CLAHE, DeiT, and hybrid results

| Model | PV accuracy (10) | PV macro P (10) | PV macro R (10) | PV macro F1 (10) | PV accuracy (mapped 9) | PV macro F1 (mapped 9) | PlantDoc accuracy (9) | PlantDoc macro P (9) | PlantDoc macro R (9) | PlantDoc macro F1 (9) | Accuracy gap* | Macro F1 gap* | Parameters | Checkpoint MiB | Forward latency ms/img | Approx FLOPs/img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV3-Small | 0.9951 | 0.9967 | 0.9952 | 0.9960 | 0.9948 | 0.9955 | 0.4940 | 0.6445 | 0.4691 | 0.4232 | 0.5008 | 0.5723 | 1,528,106 | 17.74 | 6.88 | 110995576 |
| MobileNetV3-Small + CLAHE | 0.9956 | 0.9961 | 0.9961 | 0.9961 | 0.9953 | 0.9957 | 0.4096 | 0.5826 | 0.3991 | 0.3883 | 0.5856 | 0.6074 | 1,528,106 | 17.74 | 7.82 | 110995576 |
| DeiT-Tiny | 0.9966 | 0.9974 | 0.9969 | 0.9972 | 0.9963 | 0.9968 | 0.5060 | 0.5556 | 0.4799 | 0.4615 | 0.4903 | 0.5353 | 5,526,346 | 21.14 | 7.90 | 2150268096 |
| MobileNetV3-Small + DeiT-Tiny Hybrid | 0.9966 | 0.9982 | 0.9967 | 0.9974 | 0.9963 | 0.9971 | 0.4940 | 0.5118 | 0.4617 | 0.4239 | 0.5024 | 0.5733 | 6,980,586 | 26.83 | 15.74 | 2261113400 |

PlantDoc mapped test subset: 83 images across 9 classes. *Domain gaps use PlantVillage metrics recomputed on the same nine classes; predictions into the model's other output classes count as errors. FLOPs are profiler estimates and may omit operators. Latency is the median of three batch-one model-forward trials after eight warmups on cuda; image decoding and preprocessing are excluded.

## External-domain results and domain shift

Accuracy and macro-F1 domain gaps compare the same nine mapped classes (PlantVillage n=1908 vs PlantDoc n=83) and are calculated as PlantVillage minus PlantDoc. For the hybrid, the gaps are 0.5024 accuracy and 0.5733 macro F1. DeiT-Tiny has the best external accuracy (0.5060) and macro F1 (0.4615) among these four checkpoints. These results do not support a claim that the hybrid is more robust than DeiT-Tiny; class supports are small.

## Explainability analysis

Grad-CAM++ is implemented for the MobileNet convolutional branch and DeiT patch-token branch. Branch maps are normalized independently and averaged for the combined display. The output is qualitative attribution; PlantDoc's classification split does not provide pixel masks for a localization metric.

## Ablation and limitations

The four checkpoints form the available model comparison. An additional CLAHE hybrid was not trained. Parameter count, file size, batch-one latency and approximate profiler FLOPs appear above. The external mapped subset is limited to 83 images and sparse classes; results should not be generalized beyond this evaluation.

## Conclusions supported by current data

The saved PlantVillage scores were reproduced. On the mapped PlantDoc subset, DeiT-Tiny outperforms the hybrid in both accuracy and macro F1; therefore current evidence does not show superior hybrid robustness.
