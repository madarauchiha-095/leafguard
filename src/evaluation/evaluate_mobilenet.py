import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import numpy as np
import matplotlib.pyplot as plt

from src.models.mobilenet import MobileNetV3Classifier


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("C:/LeafGuard")

DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "test"

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "mobilenet_v3_best.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics"
)

PLOTS_DIR = (
    PROJECT_ROOT /
    "results" /
    "plots"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_CLASSES = 10


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LeafGuard - MobileNetV3 Test Evaluation")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: {torch.version.cuda}"
        )

    # --------------------------------------------------------
    # LOAD TEST DATASET
    # --------------------------------------------------------

    test_dataset = datasets.ImageFolder(
        root=DATA_ROOT,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(
        f"\nTest images: {len(test_dataset)}"
    )

    print(
        f"Number of classes: "
        f"{len(test_dataset.classes)}"
    )

    print("\nClasses:")

    for index, class_name in enumerate(
        test_dataset.classes
    ):

        print(
            f" {index}: {class_name}"
        )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "\nLoading best MobileNetV3 checkpoint..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = MobileNetV3Classifier(
        num_classes=NUM_CLASSES,
        pretrained=False
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation F1: "
        f"{checkpoint['val_f1']:.4f}"
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    all_predictions = []
    all_labels = []

    correct = 0
    total = 0

    print(
        "\nRunning inference..."
    )

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"\nCorrect predictions: "
        f"{correct}/{total}"
    )

    print(
        f"Test Accuracy:  {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro Precision: {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Macro Recall:    {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"Macro F1:        {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("PER-CLASS RESULTS")
    print("=" * 70)

    print("\n" + report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    report_dict = classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        output_dict=True,
        zero_division=0
    )

    results = {
        "model": "MobileNetV3-Small",
        "checkpoint_epoch":
            checkpoint["epoch"],
        "validation_f1":
            checkpoint["val_f1"],
        "test_samples": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "classification_report":
            report_dict,
        "confusion_matrix":
            cm.tolist()
    }

    results_path = (
        RESULTS_DIR /
        "mobilenet_test_results.json"
    )

    with open(
        results_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    im = ax.imshow(cm)

    ax.set_title(
        "MobileNetV3 - Test Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    ax.set_xticks(
        np.arange(NUM_CLASSES)
    )

    ax.set_yticks(
        np.arange(NUM_CLASSES)
    )

    ax.set_xticklabels(
        test_dataset.classes,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        test_dataset.classes
    )

    # Write numbers inside cells
    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    fig.colorbar(im)

    plt.tight_layout()

    confusion_path = (
        PLOTS_DIR /
        "mobilenet_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print(
        "\nResults saved:"
    )

    print(
        results_path
    )

    print(
        "\nConfusion matrix saved:"
    )

    print(
        confusion_path
    )

    print(
        "\n✓ MobileNetV3 test evaluation complete."
    )


if __name__ == "__main__":
    main()