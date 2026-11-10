# Limitations

1. The original training and held-out results use a PlantVillage-derived dataset with controlled capture conditions; that evidence alone does not establish performance on farms or in deployment.
2. Cross-domain evaluation used only 83 mapped PlantDoc test images from nine classes. This is a small subset of the reported source and class-specific estimates have substantial uncertainty.
3. Published PlantDoc counts differ by paper/repository version and by missing image files. The preparation manifest must be used for the exact evaluated sample count.
4. PlantDoc's mapped test subset is smaller and more imbalanced than PlantVillage. Per-class metrics can have substantial uncertainty and should be reported with support.
5. Some generic PlantDoc leaf labels are treated as healthy reference classes; mappings are documented and should be revisited if source annotations indicate a different interpretation.
6. The model is closed-set. PlantDoc classes outside the ten PlantVillage labels are not evaluated as recognized categories and may be assigned an incorrect in-scope class.
7. Grad-CAM++ is qualitative attribution. The separate [PlantDoc object-detection release](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset) provides leaf boxes, but no correspondence was established between those source images and this pinned cropped classification test subset; the boxes also do not mark disease lesions. No valid paired regions were available, so localization performance is not measured. Branch-normalized CAM averaging does not represent relative branch importance.
8. Batch-one latency depends on GPU/CPU, driver, thermal state, warm-up, and software versions. Profiler FLOPs are approximate and may omit operations.
9. SHA-256 comparison detects whether a checkpoint differs from a stored expected digest. It does not authenticate the author or provide a signed chain of trust.
10. No additional CLAHE-hybrid model was trained. No real-world deployment validation or prospective agronomic assessment was conducted.
