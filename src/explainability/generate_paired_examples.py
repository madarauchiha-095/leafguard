"""Generate deterministic cross-domain Grad-CAM++ examples from saved predictions."""
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from zipfile import ZipFile

from src.explainability.gradcampp import explain_pil, overlay_cam
from src.inference.common import ROOT, build_model

OUTPUT = ROOT / "results/explainability/paired_examples"
PLANTDOC = ROOT / "data/external/plantdoc_prepared/test"
PLANTVILLAGE = ROOT / "data/processed/test"
PLANTDOC_COMMIT = "5467f6012d78d1c446145d5f582da6096f852ae8"


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["confidence"] = float(row["confidence"])
        row["correct"] = row["correct"].lower() == "true"
    return rows


def by_model(rows):
    return {key: {row["image_id"]: row for row in rows if row["model"] == key}
            for key in ("deit_tiny", "hybrid")}


def select_groups(pv_rows, external_rows):
    pv, ext = by_model(pv_rows), by_model(external_rows)
    deit_pv_correct = {}
    both_pv_correct = {}
    for row in pv["deit_tiny"].values():
        if row["correct"]:
            deit_pv_correct.setdefault(row["true_label"], []).append(row)
            if pv["hybrid"].get(row["image_id"], {}).get("correct", False):
                both_pv_correct.setdefault(row["true_label"], []).append(row)
    for values in deit_pv_correct.values():
        values.sort(key=lambda row: row["image_id"])
    for values in both_pv_correct.values():
        values.sort(key=lambda row: row["image_id"])
    candidates_a = sorted(
        (row for row in ext["deit_tiny"].values()
         if row["correct"] and row["true_label"] in both_pv_correct
         and ext["hybrid"].get(row["image_id"], {}).get("correct", False)),
        key=lambda row: (row["true_label"], row["image_id"]),
    )
    candidates_b = sorted(
        (row for row in ext["deit_tiny"].values()
         if not row["correct"] and row["true_label"] in deit_pv_correct),
        key=lambda row: (row["true_label"], row["image_id"]),
    )
    c_rows = sorted(
        (row for row in ext["deit_tiny"].values()
         if row["correct"] and not ext["hybrid"].get(row["image_id"], {}).get("correct", True)),
        key=lambda row: (row["true_label"], row["image_id"]),
    )
    d_rows = sorted(
        (row for row in ext["hybrid"].values()
         if row["correct"] and not ext["deit_tiny"].get(row["image_id"], {}).get("correct", True)),
        key=lambda row: (row["true_label"], row["image_id"]),
    )
    groups = {}
    if candidates_a:
        row = candidates_a[0]
        groups["A"] = {
            "description": "DeiT-Tiny and hybrid correct on PlantVillage and on the external image; the two images share the mapped true class.",
            "items": [
                {"source": "plantvillage", "row": both_pv_correct[row["true_label"]][0]},
                {"source": "plantdoc", "row": row},
            ],
        }
    if candidates_b:
        row = candidates_b[0]
        groups["B"] = {
            "description": "PlantVillage correct and PlantDoc incorrect for DeiT-Tiny, on separate images with the same mapped true class.",
            "items": [
                {"source": "plantvillage", "row": deit_pv_correct[row["true_label"]][0]},
                {"source": "plantdoc", "row": row},
            ],
        }
    if c_rows:
        groups["C"] = {
            "description": "DeiT-Tiny correct and hybrid incorrect on the same PlantDoc image.",
            "items": [{"source": "plantdoc", "row": c_rows[0]}],
        }
    if d_rows:
        groups["D"] = {
            "description": "Hybrid correct and DeiT-Tiny incorrect on the same PlantDoc image.",
            "items": [{"source": "plantdoc", "row": d_rows[0]}],
        }
    if set(groups) != {"A", "B", "C", "D"}:
        raise RuntimeError(f"Could not find every requested representative group; found {sorted(groups)}")
    return groups, ext


def image_path(source, row):
    root = PLANTVILLAGE if source == "plantvillage" else PLANTDOC
    path = root / Path(row["image_id"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def save_cam(model, model_key, image, output_dir, expected):
    explanation = explain_pil(model, image, model_key=model_key)
    if explanation["class_name"] != expected["predicted_label"]:
        raise RuntimeError(
            f"Fresh {model_key} prediction differs from stored result for {expected['image_id']}: "
            f"{explanation['class_name']} vs {expected['predicted_label']}"
        )
    confidence_delta = abs(explanation["confidence"] - expected["confidence"])
    if confidence_delta > 5e-3:
        raise RuntimeError(
            f"Fresh confidence differs for {expected['image_id']}: "
            f"fresh={explanation['confidence']:.8f}, stored={expected['confidence']:.8f}"
        )
    Image.fromarray(explanation["original"]).save(output_dir / f"{model_key}_input.png")
    Image.fromarray(explanation["overlay"]).save(output_dir / f"{model_key}_overlay.png")
    branch_paths = {}
    for branch, cam in explanation["branch_cams"].items():
        image_cam = np.uint8(np.clip(cam, 0, 1) * 255)
        path = output_dir / f"{model_key}_{branch}_gradcampp.png"
        Image.fromarray(image_cam).save(path)
        overlay_path = output_dir / f"{model_key}_{branch}_overlay.png"
        Image.fromarray(overlay_cam(explanation["original"], cam)).save(overlay_path)
        branch_paths[branch] = {"heatmap": path.name, "overlay": overlay_path.name}
    combined_map = f"{model_key}_combined_gradcampp.png"
    combined_overlay = f"{model_key}_combined_overlay.png"
    Image.fromarray(np.uint8(np.clip(explanation["combined_cam"], 0, 1) * 255)).save(output_dir / combined_map)
    Image.fromarray(explanation["overlay"]).save(output_dir / combined_overlay)
    return {
        "model_key": model_key,
        "true_label": expected["true_label"],
        "predicted_label": explanation["class_name"],
        "confidence": explanation["confidence"],
        "confidence_absolute_difference_from_batched_saved_run": confidence_delta,
        "correct": explanation["class_name"] == expected["true_label"],
        "branch_maps": branch_paths,
        "combined_map": combined_map,
        "combined_overlay": combined_overlay,
    }


def main():
    pv_rows = read_csv(ROOT / "results/domain_shift/plantvillage_per_image/plantvillage_test_predictions.csv")
    plantdoc_rows = read_csv(ROOT / "results/domain_shift/plantdoc_predictions.csv")
    groups, ext = select_groups(pv_rows, plantdoc_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        key: build_model(key, device=device)
        for key in ("deit_tiny", "hybrid")
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    montage_rows = []
    summary = {
        "selection_rule": "Deterministic first image_id after sorting by mapped true class then image_id; groups A/B require the same mapped true class across separately sampled PlantVillage and PlantDoc images; groups C/D use the same PlantDoc image for both models.",
        "external_source": f"PlantDoc test split at pinned commit {PLANTDOC_COMMIT}",
        "device": str(device),
        "gradcam_method": "Grad-CAM++ for DeiT patch tokens and, for the hybrid, final MobileNet feature maps plus DeiT patch tokens. Each branch is normalized separately; the hybrid combined map averages those normalized maps.",
        "interpretation_limit": "Qualitative attribution only. Maps do not establish lesion localization, faithfulness, or causal feature use; no verified paired masks are used.",
        "groups": {},
    }
    for group, details in groups.items():
        summary["groups"][group] = {"description": details["description"], "items": []}
        for item_index, item in enumerate(details["items"], start=1):
            source, row = item["source"], item["row"]
            path = image_path(source, row)
            directory = OUTPUT / f"group_{group}" / source / re.sub(r"[^A-Za-z0-9._-]+", "_", row["image_id"])
            directory.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as image:
                rgb = image.convert("RGB").copy()
            rgb.save(directory / "original.jpg", quality=95)
            model_outputs = {}
            for model_key in ("deit_tiny", "hybrid"):
                expected = ext[model_key][row["image_id"]] if source == "plantdoc" else {
                    **row,
                    **{"predicted_label": next(r["predicted_label"] for r in pv_rows
                                                  if r["model"] == model_key and r["image_id"] == row["image_id"]),
                       "confidence": next(r["confidence"] for r in pv_rows
                                           if r["model"] == model_key and r["image_id"] == row["image_id"])}
                }
                model_outputs[model_key] = save_cam(models[model_key], model_key, rgb, directory, expected)
            record = {
                "source": source,
                "image_id": row["image_id"],
                "true_label": row["true_label"],
                "source_dataset": (
                    "PlantVillage-derived test split" if source == "plantvillage"
                    else f"PlantDoc, pinned source commit {PLANTDOC_COMMIT}"
                ),
                "source_image_path": path.relative_to(ROOT).as_posix(),
                "models": model_outputs,
                "output_directory": directory.relative_to(ROOT).as_posix(),
            }
            (directory / "metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            summary["groups"][group]["items"].append(record)
            montage_rows.append((group, source, row, directory))

    # Recompute PlantDoc prediction lookups from the shared CSV to make the
    # figure's exact values explicit and avoid using display-only assumptions.
    pred_maps = {key: {row["image_id"]: row for row in ext[key].values()} for key in ext}
    fig, axes = plt.subplots(len(montage_rows), 5, figsize=(13.5, 2.7 * len(montage_rows)))
    if len(montage_rows) == 1:
        axes = np.asarray([axes])
    column_titles = ["Original", "DeiT-Tiny", "Hybrid CNN", "Hybrid transformer", "Hybrid combined"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=10)
    for row_index, (group, source, row, directory) in enumerate(montage_rows):
        files = [
            "original.jpg",
            "deit_tiny_deit_overlay.png",
            "hybrid_mobilenet_overlay.png",
            "hybrid_deit_overlay.png",
            "hybrid_combined_overlay.png",
        ]
        for col, filename in enumerate(files):
            with Image.open(directory / filename) as picture:
                axes[row_index, col].imshow(picture.convert("RGB"))
            axes[row_index, col].axis("off")
        if source == "plantdoc":
            deit_row = pred_maps["deit_tiny"][row["image_id"]]
            hybrid_row = pred_maps["hybrid"][row["image_id"]]
        else:
            deit_row = next(r for r in pv_rows if r["model"] == "deit_tiny" and r["image_id"] == row["image_id"])
            hybrid_row = next(r for r in pv_rows if r["model"] == "hybrid" and r["image_id"] == row["image_id"])
        axes[row_index, 0].axis("on")
        axes[row_index, 0].set_xticks([])
        axes[row_index, 0].set_yticks([])
        for spine in axes[row_index, 0].spines.values():
            spine.set_visible(False)
        axes[row_index, 0].set_xlabel(
            f"Group {group} · {source}\nTrue: {row['true_label']}\nDeiT: {deit_row['predicted_label']} ({deit_row['confidence']:.2f})\nHybrid: {hybrid_row['predicted_label']} ({hybrid_row['confidence']:.2f})",
            fontsize=7, labelpad=2,
        )
    fig.suptitle("Deterministically selected Grad-CAM++ examples", fontsize=14)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.98))
    figure_path = OUTPUT / "selected_gradcam_examples.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    summary["figure"] = figure_path.relative_to(ROOT).as_posix()
    summary["max_fresh_vs_saved_confidence_difference"] = max(
        entry["confidence_absolute_difference_from_batched_saved_run"]
        for group_data in summary["groups"].values()
        for item in group_data["items"]
        for entry in item["models"].values()
    )
    (OUTPUT / "selection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote paired Grad-CAM++ examples for groups A-D under {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
