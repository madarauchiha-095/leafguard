"""Evaluate the four fixed PlantVillage checkpoints on mapped PlantDoc test images."""
import argparse
import csv
from io import BytesIO
import json
import re
import time
from zipfile import ZipFile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.datasets import ImageFolder

from src.inference.common import (
    CLASS_NAMES, MODEL_SPECS, ROOT, build_model, make_transform,
)


class PlantSegZipDataset(Dataset):
    """ImageFolder-compatible view over mapped image members, read in memory."""

    def __init__(self, archive_path, manifest_path, transform=None):
        self.archive_path = Path(archive_path)
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.records = sorted(payload["eligible_records"], key=lambda row: row["image_id"])
        self.classes = sorted({row["true_label"] for row in self.records})
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.sample_ids = [row["image_id"] for row in self.records]
        self.samples = [
            (row["image_member"], self.class_to_idx[row["true_label"]])
            for row in self.records
        ]
        self.root = str(self.archive_path)
        self.transform = transform
        # Keep the central directory open; reopening this 1 GB archive for
        # every image would repeatedly parse its full member table.
        self._archive = ZipFile(self.archive_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        data = self._archive.read(record["image_member"])
        with Image.open(BytesIO(data)) as image:
            image = image.convert("RGB")
            tensor = self.transform(image) if self.transform is not None else image
        return tensor, self.class_to_idx[record["true_label"]]

    def close(self):
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def __del__(self):
        self.close()


def evaluate(model_key, dataset, device, batch_size):
    spec = MODEL_SPECS[model_key]
    model = build_model(model_key, device=device)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    true_ids, pred_ids, confidences, records = [], [], [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for images, local_labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = logits.softmax(dim=1)
            predictions = probs.argmax(dim=1)
            true_ids.extend(
                CLASS_NAMES.index(dataset.classes[int(i)]) for i in local_labels
            )
            batch_predictions = predictions.cpu().tolist()
            batch_confidences = probs.max(dim=1).values.cpu().tolist()
            batch_true_ids = [CLASS_NAMES.index(dataset.classes[int(i)]) for i in local_labels]
            pred_ids.extend(batch_predictions)
            confidences.extend(batch_confidences)
            start_index = len(records)
            for offset, (true_id, pred_id, confidence) in enumerate(
                zip(batch_true_ids, batch_predictions, batch_confidences)
            ):
                if hasattr(dataset, "sample_ids"):
                    image_id = dataset.sample_ids[start_index + offset]
                else:
                    image_path = Path(dataset.samples[start_index + offset][0])
                    image_id = image_path.relative_to(Path(dataset.root)).as_posix()
                records.append({
                    "image_id": image_id,
                    "true_label": CLASS_NAMES[true_id],
                    "predicted_label": CLASS_NAMES[pred_id],
                    "confidence": float(confidence),
                    "correct": bool(true_id == pred_id),
                    "model": model_key,
                })
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    mapped_ids = [CLASS_NAMES.index(name) for name in dataset.classes]
    report = classification_report(
        true_ids, pred_ids,
        labels=mapped_ids,
        target_names=dataset.classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(true_ids, pred_ids, labels=list(range(len(CLASS_NAMES))))
    result = {
        "model": spec["name"],
        "model_key": model_key,
        "checkpoint": str(spec["checkpoint"].relative_to(ROOT)),
        "dataset": getattr(dataset, "dataset_name", "PlantDoc official test split, mapped subset"),
        "preprocessing": {
            "resize": [224, 224],
            "normalization": "ImageNet mean/std",
            "clahe": bool(spec["clahe"]),
            "clahe_clip_limit": 2.0 if spec["clahe"] else None,
            "clahe_tile_grid_size": [8, 8] if spec["clahe"] else None,
        },
        "sample_count": len(dataset),
        "mapped_classes": dataset.classes,
        "metrics": {
            "accuracy": float(accuracy_score(true_ids, pred_ids)),
            "precision_macro": float(precision_score(
                true_ids, pred_ids, labels=mapped_ids, average="macro", zero_division=0
            )),
            "recall_macro": float(recall_score(
                true_ids, pred_ids, labels=mapped_ids, average="macro", zero_division=0
            )),
            "f1_macro": float(f1_score(
                true_ids, pred_ids, labels=mapped_ids, average="macro", zero_division=0
            )),
            "f1_weighted": float(f1_score(
                true_ids, pred_ids, labels=mapped_ids, average="weighted", zero_division=0
            )),
            "correct": int(sum(y == p for y, p in zip(true_ids, pred_ids))),
            "total": len(true_ids),
            "total_inference_seconds": elapsed,
        },
        "per_class": {name: report[name] for name in dataset.classes},
        "confusion_matrix_labels": CLASS_NAMES,
        "confusion_matrix": cm.tolist(),
        "mean_max_softmax_confidence": float(np.mean(confidences)),
        "prediction_records": records,
        "interpretation_note": (
            "Predictions into any of the model's ten output classes are counted as errors "
            "when they differ from the mapped external ground truth. Macro metrics average "
            "only over mapped ground-truth classes."
        ),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_confusion(result, path):
    cm = np.asarray(result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    names = [name.replace("___", "\n") for name in result["confusion_matrix_labels"]]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Predicted LeafGuard class")
    ax.set_ylabel("True mapped external class")
    ax.set_title(f"{result['model']} — {result['dataset']}")
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            if cm[row, col]:
                ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                        color="white" if cm[row, col] > threshold else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/external/plantdoc_prepared/test")
    parser.add_argument("--manifest", type=Path, help="Mapped external inventory JSON for ZIP-hosted samples")
    parser.add_argument("--archive", type=Path, help="Source ZIP used with --manifest; images are read in memory")
    parser.add_argument("--output", type=Path, default=ROOT / "results/domain_shift")
    parser.add_argument("--dataset-name", default="PlantDoc mapped test split")
    parser.add_argument("--dataset-slug", default="plantdoc")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if bool(args.manifest) != bool(args.archive):
        raise ValueError("--manifest and --archive must be supplied together")
    if args.manifest:
        inventory = PlantSegZipDataset(args.archive, args.manifest)
    else:
        if not args.dataset.is_dir():
            raise FileNotFoundError(
                f"Prepared field dataset missing: {args.dataset}. "
                "See python -m src.data.prepare_plantdoc --help."
            )
        inventory = ImageFolder(args.dataset)
    if not inventory.classes:
        raise RuntimeError(f"No mapped classes found under {args.dataset}")
    unknown = sorted(set(inventory.classes) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Prepared dataset has class folders outside the trained labels: {unknown}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    expected_samples = inventory.samples
    summary = {"device": str(device), "external_images": len(inventory), "mapped_classes": inventory.classes,
               "dataset_name": args.dataset_name, "models": {}, "predictions": []}
    for key in MODEL_SPECS:
        spec = MODEL_SPECS[key]
        # Preserve the checkpoint's training/evaluation preprocessing. In
        # particular, the CLAHE checkpoint must receive LAB-lightness CLAHE.
        dataset = (
            PlantSegZipDataset(args.archive, args.manifest, transform=make_transform(spec["clahe"]))
            if args.manifest else
            ImageFolder(args.dataset, transform=make_transform(spec["clahe"]))
        )
        dataset.dataset_name = args.dataset_name
        if dataset.classes != inventory.classes or dataset.samples != expected_samples:
            raise RuntimeError(f"External sample inventory changed while preparing {key}")
        print(f"Evaluating {spec['name']} on {len(dataset)} images ({device}; CLAHE={spec['clahe']})...")
        result = evaluate(key, dataset, device, args.batch_size)
        slug = re.sub(r"[^a-z0-9]+", "_", args.dataset_slug.lower()).strip("_") or "external"
        name = f"{key}_{slug}_test_results"
        save_confusion(result, args.output / f"{key}_{slug}_confusion_matrix.png")
        summary["models"][key] = result["metrics"]
        summary["predictions"].extend(result["prediction_records"])
        result.pop("prediction_records", None)
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result["metrics"], indent=2))
    with (args.output / f"{slug}_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["image_id", "true_label", "predicted_label", "confidence", "correct", "model"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["predictions"])
    (args.output / f"{slug}_predictions.json").write_text(
        json.dumps(summary["predictions"], indent=2), encoding="utf-8"
    )
    summary.pop("predictions")
    (args.output / f"{slug}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if slug == "plantdoc":
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
