"""Export draft, image-level visual factor labels for the mapped PlantDoc subset."""
import csv
import json
import re
from pathlib import Path

from src.inference.common import ROOT


# Labels were assigned by reviewing the deterministic contact sheets against the
# written schema. They are exploratory draft annotations, not expert ground truth.
ANNOTATIONS = {
    1: "simple,close,single_dominant,even,not_visible,none",
    2: "simple,close,single_dominant,even,not_visible,none",
    3: "complex,medium,multiple,even,not_visible,partial",
    4: "simple,close,multiple,even,not_visible,none",
    5: "simple,close,single_dominant,adverse,not_visible,none",
    6: "complex,medium,multiple,adverse,not_visible,none",
    7: "complex,close,single_dominant,even,not_visible,partial",
    8: "complex,close,single_dominant,even,not_visible,none",
    9: "simple,close,single_dominant,unclear,visible,none",
    10: "simple,close,single_dominant,even,not_visible,partial",
    11: "complex,close,single_dominant,even,not_visible,none",
    12: "simple,close,single_dominant,even,not_visible,none",
    13: "simple,close,single_dominant,even,not_visible,none",
    14: "simple,close,single_dominant,even,not_visible,none",
    15: "simple,close,single_dominant,even,not_visible,none",
    16: "simple,close,single_dominant,even,not_visible,none",
    17: "complex,close,single_dominant,adverse,not_visible,none",
    18: "complex,close,single_dominant,even,not_visible,none",
    19: "simple,close,single_dominant,even,not_visible,none",
    20: "complex,close,multiple,even,not_visible,none",
    21: "complex,distant,whole_plant_or_cluster,even,not_visible,none",
    22: "complex,medium,multiple,even,not_visible,none",
    23: "simple,close,single_dominant,unclear,not_visible,none",
    24: "simple,close,single_dominant,unclear,not_visible,none",
    25: "complex,close,single_dominant,even,not_visible,none",
    26: "simple,close,single_dominant,even,not_visible,none",
    27: "simple,close,single_dominant,adverse,not_visible,none",
    28: "simple,close,single_dominant,unclear,visible,none",
    29: "simple,close,single_dominant,even,not_visible,none",
    30: "complex,close,single_dominant,even,not_visible,none",
    31: "complex,close,single_dominant,even,not_visible,none",
    32: "complex,close,single_dominant,adverse,not_visible,none",
    33: "simple,close,single_dominant,even,not_visible,none",
    34: "complex,medium,multiple,adverse,not_visible,partial",
    35: "simple,close,single_dominant,even,not_visible,none",
    36: "simple,close,multiple,even,not_visible,none",
    37: "complex,medium,multiple,adverse,not_visible,none",
    38: "complex,medium,multiple,adverse,visible,none",
    39: "complex,close,single_dominant,even,not_visible,none",
    40: "simple,close,single_dominant,even,not_visible,none",
    41: "complex,distant,whole_plant_or_cluster,even,not_visible,none",
    42: "complex,medium,multiple,even,not_visible,none",
    43: "simple,close,single_dominant,unclear,visible,none",
    44: "complex,close,single_dominant,even,not_visible,none",
    45: "complex,medium,multiple,even,not_visible,none",
    46: "simple,close,single_dominant,even,not_visible,none",
    47: "simple,close,single_dominant,even,not_visible,none",
    48: "complex,close,single_dominant,even,not_visible,partial",
    49: "complex,medium,multiple,even,not_visible,none",
    50: "simple,close,single_dominant,unclear,not_visible,none",
    51: "complex,medium,multiple,even,not_visible,none",
    52: "complex,close,multiple,even,not_visible,none",
    53: "simple,close,single_dominant,even,not_visible,none",
    54: "complex,medium,multiple,even,not_visible,none",
    55: "simple,close,multiple,even,not_visible,none",
    56: "simple,medium,multiple,even,not_visible,none",
    57: "complex,medium,whole_plant_or_cluster,adverse,not_visible,none",
    58: "complex,close,single_dominant,even,not_visible,partial",
    59: "complex,medium,whole_plant_or_cluster,even,not_visible,none",
    60: "complex,close,single_dominant,even,not_visible,none",
    61: "complex,distant,whole_plant_or_cluster,adverse,not_visible,none",
    62: "complex,close,single_dominant,adverse,not_visible,none",
    63: "simple,close,single_dominant,even,not_visible,none",
    64: "complex,medium,multiple,even,not_visible,none",
    65: "simple,close,single_dominant,even,not_visible,none",
    66: "simple,close,single_dominant,even,not_visible,none",
    67: "complex,close,single_dominant,even,not_visible,none",
    68: "complex,close,single_dominant,even,not_visible,partial",
    69: "complex,medium,multiple,even,not_visible,partial",
    70: "simple,close,multiple,even,not_visible,none",
    71: "simple,close,single_dominant,even,not_visible,none",
    72: "simple,medium,multiple,even,not_visible,none",
    73: "simple,close,single_dominant,even,not_visible,none",
    74: "complex,close,single_dominant,even,not_visible,none",
    75: "complex,medium,single_dominant,even,not_visible,partial",
    76: "simple,medium,multiple,even,not_visible,none",
    77: "simple,close,single_dominant,even,not_visible,none",
    78: "simple,close,single_dominant,even,not_visible,none",
    79: "simple,close,single_dominant,even,not_visible,none",
    80: "complex,close,multiple,even,not_visible,none",
    81: "simple,close,single_dominant,even,not_visible,none",
    82: "simple,close,single_dominant,even,not_visible,none",
    83: "complex,medium,multiple,even,not_visible,none",
}


def main():
    manifest_path = ROOT / "results/plots/domain_shift/visual_review/contact_sheet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fields = [
        "image_id", "true_label", "width_px", "height_px", "minimum_dimension_px",
        "below_224_px", "background", "framing", "foliage_count", "lighting",
        "blur", "occlusion", "annotation_status",
    ]
    rows = []
    for item in manifest:
        filename = Path(item["image_id"]).name
        match = re.search(r"plantdoc_(\d+)\.", filename)
        if not match:
            raise ValueError(f"Unexpected PlantDoc file name: {filename}")
        number = int(match.group(1))
        if number not in ANNOTATIONS:
            raise ValueError(f"Missing draft visual labels for sample {number}")
        values = ANNOTATIONS[number].split(",")
        rows.append({
            "image_id": item["image_id"],
            "true_label": item["image_id"].split("/")[0],
            "width_px": item["width"],
            "height_px": item["height"],
            "minimum_dimension_px": item["minimum_dimension_px"],
            "below_224_px": item["minimum_dimension_px"] < 224,
            **dict(zip(fields[6:12], values)),
            "annotation_status": "assistant_visual_draft_requires_author_review",
        })
    if len(rows) != 83 or len({row["image_id"] for row in rows}) != 83 or len(ANNOTATIONS) != 83:
        raise RuntimeError(f"Expected 83 one-to-one annotation rows, found {len(rows)}")
    output = ROOT / "results/domain_shift/plantdoc_visual_annotations.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} draft image annotations to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
