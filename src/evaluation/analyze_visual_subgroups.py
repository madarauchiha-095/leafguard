"""Compute descriptive external metrics by draft visual-shift annotations."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from src.inference.common import ROOT


FACTORS = ("background", "framing", "foliage_count", "lighting", "blur", "occlusion", "below_224_px")


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ROOT / "results/domain_shift/plantdoc_visual_annotations.csv")
    parser.add_argument("--predictions", type=Path, default=ROOT / "results/domain_shift/plantdoc_predictions.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "results/domain_shift/visual_subgroup_analysis.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results/domain_shift/visual_subgroup_metrics.csv")
    args = parser.parse_args()
    annotations = read_csv(args.annotations)
    predictions = read_csv(args.predictions)
    ann_by_id = {row["image_id"]: row for row in annotations}
    if len(ann_by_id) != len(annotations):
        raise ValueError("Duplicate image IDs in visual annotations")
    grouped = {}
    for row in predictions:
        grouped.setdefault(row["model"], []).append(row)
    for key, rows in grouped.items():
        ids = {row["image_id"] for row in rows}
        if ids != set(ann_by_id):
            raise ValueError(f"Annotation inventory does not match prediction inventory for {key}")
        for row in rows:
            if row["true_label"] != ann_by_id[row["image_id"]]["true_label"]:
                raise ValueError(f"True-label mismatch for {key}/{row['image_id']}")

    result_rows = []
    output = {
        "dataset": "PlantDoc mapped test subset",
        "sample_count": len(annotations),
        "annotation_status": "Single-reviewer assistant draft; author validation required before confirmatory claims.",
        "interpretation": "Exploratory subgroup descriptions only. Groups are small, factors are subjective and correlated, and no causal or inferential claims are supported.",
        "models": {},
    }
    for model, rows in grouped.items():
        by_id = {row["image_id"]: row for row in rows}
        model_output = {}
        for factor in FACTORS:
            levels = sorted({ann_by_id[image_id][factor] for image_id in ann_by_id})
            model_output[factor] = {}
            for level in levels:
                ids = [image_id for image_id in ann_by_id if ann_by_id[image_id][factor] == level]
                y_true = [ann_by_id[image_id]["true_label"] for image_id in ids]
                y_pred = [by_id[image_id]["predicted_label"] for image_id in ids]
                labels = sorted(set(y_true))
                score = {
                    "sample_count": len(ids),
                    "correct": int(sum(a == b for a, b in zip(y_true, y_pred))),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "macro_f1_over_true_classes_present": float(f1_score(
                        y_true, y_pred, labels=labels, average="macro", zero_division=0
                    )),
                    "class_support": dict(Counter(y_true)),
                    "small_sample_note": "n<10; unstable descriptive subgroup" if len(ids) < 10 else "descriptive only; no confidence interval",
                }
                model_output[factor][level] = score
                result_rows.append({"model": model, "factor": factor, "level": level, **score})
        output["models"][model] = model_output

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        fields = ["model", "factor", "level", "sample_count", "correct", "accuracy",
                  "macro_f1_over_true_classes_present", "class_support", "small_sample_note"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result_rows:
            row = dict(row)
            row["class_support"] = json.dumps(row["class_support"], ensure_ascii=False)
            writer.writerow({key: row.get(key) for key in fields})
    print(f"Wrote {len(result_rows)} subgroup metric rows across {len(grouped)} models.")


if __name__ == "__main__":
    main()
