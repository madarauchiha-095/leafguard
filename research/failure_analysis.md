# External failure summary

High-confidence error fraction is the fraction incorrect among predictions with max-softmax ≥0.90; it is descriptive and is not a calibration metric.

| Model | Dataset | n | Worst class (F1; n) | Most common confusion | ≥0.90 predictions / errors | High-confidence error fraction | Macro F1 |
|---|---|---:|---|---|---:|---:|---:|
| MobileNetV3-Small | PlantDoc | 83 | Potato___Early_blight (0.000; 8) | Potato___Late_blight → Tomato___Late_blight (n=6) | 58 / 28 | 48.3% | 0.423 |
| MobileNetV3-Small | PlantSeg v7 | 143 | Apple___Black_rot (0.000; 12) | Apple___Apple_scab → Tomato___Late_blight (n=13) | 70 / 52 | 74.3% | 0.151 |
| MobileNetV3-Small + CLAHE | PlantDoc | 83 | Potato___Early_blight (0.000; 8) | Grape___healthy → Tomato___Late_blight (n=7) | 53 / 28 | 52.8% | 0.388 |
| MobileNetV3-Small + CLAHE | PlantSeg v7 | 143 | Apple___Black_rot (0.000; 12) | Apple___Apple_scab → Tomato___Early_blight (n=14) | 71 / 48 | 67.6% | 0.159 |
| DeiT-Tiny | PlantDoc | 83 | Tomato___healthy (0.000; 8) | Potato___Late_blight → Tomato___Late_blight (n=6) | 50 / 22 | 44.0% | 0.462 |
| DeiT-Tiny | PlantSeg v7 | 143 | Apple___Black_rot (0.000; 12) | Tomato___Early_blight → Tomato___Late_blight (n=14) | 60 / 44 | 73.3% | 0.186 |
| MobileNetV3-Small + DeiT-Tiny Hybrid | PlantDoc | 83 | Tomato___healthy (0.000; 8) | Potato___Late_blight → Tomato___Late_blight (n=6) | 50 / 21 | 42.0% | 0.424 |
| MobileNetV3-Small + DeiT-Tiny Hybrid | PlantSeg v7 | 143 | Apple___Black_rot (0.000; 12) | Tomato___Early_blight → Tomato___Late_blight (n=14) | 65 / 52 | 80.0% | 0.156 |
