"""Select representative hybrid predictions and save Grad-CAM++ research artifacts."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.explainability.gradcampp import explain_pil
from src.inference.common import CLASS_NAMES, ROOT, build_model, make_transform

OUTPUT = ROOT / "results/explainability"


def score_dataset(model, dataset, device, labels_are_local=False):
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    rows = []
    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = logits.softmax(dim=1).cpu().numpy()
            for i, row in enumerate(probs):
                sample_index = offset + i
                true_name = dataset.classes[int(labels[i])] if labels_are_local else CLASS_NAMES[int(labels[i])]
                true_id = CLASS_NAMES.index(true_name)
                predicted = int(row.argmax())
                order = np.argsort(row)[::-1]
                rows.append({
                    "index": sample_index,
                    "path": dataset.samples[sample_index][0],
                    "true_class": true_name,
                    "true_id": true_id,
                    "predicted_id": predicted,
                    "predicted_class": CLASS_NAMES[predicted],
                    "confidence": float(row[predicted]),
                    "margin": float(row[order[0]] - row[order[1]]),
                    "correct": predicted == true_id,
                })
            offset += len(labels)
    return rows


def save_example(model, row, category, device):
    directory = OUTPUT / category
    directory.mkdir(parents=True, exist_ok=True)
    with Image.open(row["path"]) as source:
        source = source.convert("RGB")
        explanation = explain_pil(model, source, model_key="hybrid")
        source.save(directory / "original.jpg", quality=95)
    Image.fromarray(explanation["overlay"]).save(directory / "gradcampp_overlay.png")
    Image.fromarray(np.uint8(explanation["combined_cam"] * 255)).save(directory / "gradcampp_heatmap.png")
    for branch, cam in explanation["branch_cams"].items():
        Image.fromarray(np.uint8(np.clip(cam, 0, 1) * 255)).save(directory / f"{branch}_gradcampp.png")
    metadata = {
        "source_image": row["path"],
        "true_class": row["true_class"],
        "predicted_class": explanation["class_name"],
        "correct": row["correct"],
        "confidence": explanation["confidence"],
        "top_3": explanation["top3"],
        "example_selection": category,
        "method": "Grad-CAM++ on MobileNet final convolution and DeiT final-block norm1 patch tokens; separately normalized branch maps averaged.",
        "limit": "Qualitative attribution only; the external dataset does not provide segmentation masks for these classification samples.",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def select(rows):
    correct = [r for r in rows if r["correct"]]
    incorrect = [r for r in rows if not r["correct"]]
    picked = {}
    if correct:
        picked["correct"] = max(correct, key=lambda row: row["confidence"])
        picked["difficult"] = min(correct, key=lambda row: row["margin"])
    if incorrect:
        picked["failure"] = max(incorrect, key=lambda row: row["confidence"])
    return picked


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("hybrid", device=device)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"device": str(device), "examples": {}, "external_dataset_available": False}

    pv = ImageFolder(ROOT / "data/processed/test", transform=make_transform())
    pv_rows = score_dataset(model, pv, device, labels_are_local=True)
    for kind, row in select(pv_rows).items():
        category = f"plantvillage_{kind}"
        report["examples"][category] = save_example(model, row, category, device)

    external_root = ROOT / "data/external/plantdoc_prepared/test"
    if external_root.is_dir() and any(external_root.iterdir()):
        external = ImageFolder(external_root, transform=make_transform())
        external_rows = score_dataset(model, external, device, labels_are_local=True)
        report["external_dataset_available"] = True
        selected = select(external_rows)
        for kind, row in selected.items():
            # Keep one field-correct and one field-failure example when available.
            if kind in ("correct", "failure"):
                category = f"plantdoc_{kind}"
                report["examples"][category] = save_example(model, row, category, device)
        report["external_images"] = len(external)
    else:
        report["external_note"] = "PlantDoc has not yet been acquired and prepared; external examples will be generated by rerunning this script after preparation."

    (OUTPUT / "selection_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
