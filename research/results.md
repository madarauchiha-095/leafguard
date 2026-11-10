# Results

## Controlled-domain results

The project includes saved PlantVillage test metrics for all four checkpoints. The script `python -m src.evaluation.finalize_results` re-evaluates the current checkpoints on the unchanged test split and writes new records to `results/audit/`; it also generates `results/final_results.json`, `results/final_results.md`, and comparison figures. The verified metric values and the saved-versus-verified consistency checks are maintained in those generated files. A result is treated as verified only when the re-run completes and the consistency record is available.

## External-domain results

The pinned PlantDoc source test split contributed 83 reliably mapped images from nine classes. The exact class supports, per-class precision/recall/F1, and 10×10 confusion matrices are in `results/domain_shift/`.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Correct/total |
|---|---:|---:|---:|---:|---:|
| MobileNetV3 | 0.4940 | 0.6445 | 0.4691 | 0.4232 | 41/83 |
| MobileNetV3 + CLAHE | 0.4096 | 0.5826 | 0.3991 | 0.3883 | 34/83 |
| DeiT-Tiny | 0.5060 | 0.5556 | 0.4799 | 0.4615 | 42/83 |
| Hybrid | 0.4940 | 0.5118 | 0.4617 | 0.4239 | 41/83 |

Domain gaps are computed on the same nine mapped classes (1,908 PlantVillage images and 83 PlantDoc images), as PlantVillage minus PlantDoc, for accuracy and macro F1 respectively: MobileNetV3 0.5008/0.5723; MobileNetV3+CLAHE 0.5856/0.6074; DeiT-Tiny 0.4903/0.5353; hybrid 0.5024/0.5733. Lower gaps indicate less measured degradation on this particular subset. DeiT-Tiny has the best external accuracy and macro F1; the hybrid does not demonstrate superior robustness. CLAHE values use the corrected, checkpoint-specific preprocessing.

## Model resources

Parameter count, checkpoint size, inference latency, and approximate profiler FLOPs are produced by `finalize_results.py`. Latency is batch-one measurement on the recorded hardware. FLOPs are incomplete estimates when the profiler does not support every operator.

## Explainability

Grad-CAM++ targets both hybrid branches and saves combined and branch-specific maps with metadata. The selected examples are qualitative. There are no verified pixel-level disease or leaf masks for this classification evaluation, so localization overlap or activation-region accuracy is not calculated.
