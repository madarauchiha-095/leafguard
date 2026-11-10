# PlantVillage–PlantDoc class mapping

The mapping is intentionally conservative and applies to PlantDoc's class-folder labels. Folder strings are normalized for case, spaces, and underscores by the preparation script. Only images in PlantDoc's official test split are considered.

| PlantVillage class | PlantDoc class | Status | Reason |
|---|---|---|---|
| Apple___Apple_scab | Apple Scab Leaf | Mapped | Same crop and explicitly named disease. |
| Apple___Black_rot | — | Excluded | No exact apple black rot class in the selected PlantDoc classification labels; apple rust is not equivalent. |
| Corn_(maize)___Common_rust_ | Corn Rust Leaf | Mapped | Same crop and explicit common rust label. |
| Grape___Black_rot | Grape Leaf Black Rot | Mapped | Same crop and explicit black rot label. |
| Grape___healthy | Grape leaf | Mapped | Unqualified grape leaf class denotes the healthy reference class in the PlantDoc label set. |
| Potato___Early_blight | Potato Leaf Early Blight | Mapped | Same crop and explicit early blight label. |
| Potato___Late_blight | Potato Leaf Late Blight | Mapped | Same crop and explicit late blight label. |
| Tomato___Early_blight | Tomato Early Blight Leaf | Mapped | Same crop and explicit early blight label. |
| Tomato___Late_blight | Tomato Leaf Late Blight | Mapped | Same crop and explicit late blight label. |
| Tomato___healthy | Tomato leaf | Mapped | Unqualified tomato leaf class denotes the healthy reference class in the PlantDoc label set. |

Other PlantDoc classes (including apple rust, corn leaf blight/gray spot, potato leaf diseases outside the selected set, tomato bacterial/viral/Septoria/mold classes, and other crops) are excluded because they have no compatible output class in this fixed ten-class model. They are not coerced into the nearest available disease category.

## Complete pinned test-split class disposition

This table accounts for all 236 images in the pinned upstream `test/` split. Counts are from Git tree metadata; only mapped images were downloaded. A source category absent from the PlantVillage output taxonomy remains excluded even if it is a healthy leaf.

| PlantDoc test class | Images | Disposition | Reason if excluded |
|---|---:|---|---|
| Apple Scab Leaf | 10 | Mapped → Apple___Apple_scab | — |
| Apple leaf | 9 | Excluded | Generic leaf label; healthy interpretation not reliable enough for this crosswalk. |
| Apple rust leaf | 10 | Excluded | Apple rust is not one of the trained apple classes. |
| Bell_pepper leaf | 8 | Excluded | Crop is outside the model label set. |
| Bell_pepper leaf spot | 9 | Excluded | Crop is outside the model label set. |
| Blueberry leaf | 11 | Excluded | Crop is outside the model label set. |
| Cherry leaf | 10 | Excluded | Crop is outside the model label set. |
| Corn Gray leaf spot | 4 | Excluded | Different corn disease from Common Rust. |
| Corn leaf blight | 12 | Excluded | Different corn disease from Common Rust. |
| Corn rust leaf | 10 | Mapped → Corn_(maize)___Common_rust_ | — |
| Peach leaf | 9 | Excluded | Crop is outside the model label set. |
| Potato leaf early blight | 8 | Mapped → Potato___Early_blight | — |
| Potato leaf late blight | 8 | Mapped → Potato___Late_blight | — |
| Raspberry leaf | 7 | Excluded | Crop is outside the model label set. |
| Soyabean leaf | 8 | Excluded | Crop is outside the model label set. |
| Squash Powdery mildew leaf | 6 | Excluded | Crop is outside the model label set. |
| Strawberry leaf | 8 | Excluded | Crop is outside the model label set. |
| Tomato Early blight leaf | 9 | Mapped → Tomato___Early_blight | — |
| Tomato Septoria leaf spot | 11 | Excluded | Different tomato disease from the trained classes. |
| Tomato leaf | 8 | Mapped → Tomato___healthy | — |
| Tomato leaf bacterial spot | 9 | Excluded | Different tomato disease from the trained classes. |
| Tomato leaf late blight | 10 | Mapped → Tomato___Late_blight | — |
| Tomato leaf mosaic virus | 10 | Excluded | Different tomato disease from the trained classes. |
| Tomato leaf yellow virus | 6 | Excluded | Different tomato disease from the trained classes. |
| Tomato mold leaf | 6 | Excluded | Different tomato disease from the trained classes. |
| grape leaf | 12 | Mapped → Grape___healthy | — |
| grape leaf black rot | 8 | Mapped → Grape___Black_rot | — |

`Tomato two spotted spider mites leaf` appears in the upstream training tree but has no test images in the pinned revision; it contributes zero to the 236-image test split and is not evaluated. The upstream test tree has no `Potato leaf` generic class. The exact counts and per-image file hashes are in `data/external/plantdoc_prepared/preparation_manifest.json` and `data/external/plantdoc_source_subset/source_manifest.json`.
