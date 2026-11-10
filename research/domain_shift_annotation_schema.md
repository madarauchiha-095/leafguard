# Domain-Shift Visual Annotation Schema

**Status:** preliminary one-reviewer annotations of the 83 mapped PlantDoc samples. The values are descriptive image-level observations, not verified causal factors. A project author should review and approve them before using subgroup results in a manuscript.

## Fields and decision rules

| Field | Allowed values | Rule |
|---|---|---|
| `background` | `simple`, `complex`, `unclear` | `simple` means a mostly uniform/plain background with little visible scene context. `complex` means multiple scene elements such as other foliage, soil, containers, buildings, or patterned scenery. Use `unclear` when the image is too small/ambiguous. |
| `framing` | `close`, `medium`, `distant`, `unclear` | Estimate the target leaf/plant occupancy: `close` if the focal leaf or plant fills roughly half or more of the frame; `distant` if it occupies roughly one fifth or less; `medium` otherwise. This is an approximate visual bin, not a pixel mask. |
| `foliage_count` | `single_dominant`, `multiple`, `whole_plant_or_cluster`, `unclear` | `single_dominant` if one leaf/leaflet is the obvious subject; `multiple` if two or more distinct leaves dominate; `whole_plant_or_cluster` if the frame shows a plant/cluster rather than a selected leaf. |
| `lighting` | `even`, `adverse`, `unclear` | `adverse` if strong shadows, glare, clipping, or visibly uneven/low illumination affects the leaf. Otherwise use `even`. |
| `blur` | `not_visible`, `visible`, `unclear` | `visible` only when leaf edges/texture are visibly smeared or out of focus at native size. |
| `occlusion` | `none`, `partial`, `major`, `unclear` | `partial` when some target leaf area is hidden/cropped by overlap or another object; `major` when a substantial part of the target is obscured. Do not infer lesion visibility from model maps. |
| `minimum_dimension_px` | non-negative integer | Read directly from the decoded original image; `below_224_px` is derived as `minimum_dimension_px < 224`, which flags images upscaled to the model's 224-pixel input size. |

## Workflow and limits

The review contact sheets are generated deterministically from the prepared PlantDoc test subset and list source-relative image IDs and original dimensions. The draft CSV records one annotation row per image. A single reviewer can assign subjective scene attributes inconsistently; categories may overlap, and the 83-image sample is too small for strong subgroup conclusions. Report counts and exploratory metrics with denominators. Do not claim that a visual factor caused model error. The `below_224_px` field is objective but still only a proxy for image detail; compression and focus are not captured by pixel count.

Regenerate the sheets with:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.create_review_contact_sheets
```

Contact sheets and the machine-readable image dimension manifest are stored under `results/plots/domain_shift/visual_review/`.
