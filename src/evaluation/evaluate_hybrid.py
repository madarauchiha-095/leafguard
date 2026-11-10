import os
import json
import numpy as np

import torch
import matplotlib.pyplot as plt

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

from src.models.hybrid import MobileNetDeiTHybrid


# ============================================================
# Configuration
# ============================================================

BASE_DIR = r"C:\LeafGuard"

TEST_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "test"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hybrid_best.pth"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "metrics"
)

PLOTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "plots"
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "hybrid_test_results.json"
)

CONFUSION_MATRIX_PATH = os.path.join(
    PLOTS_DIR,
    "hybrid_confusion_matrix.png"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("LeafGuard - MobileNetV3 + DeiT-Tiny Hybrid Test Evaluation")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: {torch.version.cuda}"
        )

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Test transform
    # --------------------------------------------------------

    test_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print()
    print(
        f"Test samples: {len(test_dataset)}"
    )

    print()
    print("Classes:")

    for idx, class_name in enumerate(
        test_dataset.classes
    ):

        print(
            f"{idx}: {class_name}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Loading hybrid model...")

    model = MobileNetDeiTHybrid(
        num_classes=len(
            test_dataset.classes
        )
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(device)
    model.eval()

    print(
        "✓ Model loaded successfully."
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Starting test evaluation")
    print("=" * 70)

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # --------------------------------------------------------
    # Metrics
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

    correct = sum(
        p == y
        for p, y in zip(
            all_predictions,
            all_labels
        )
    )

    total = len(
        all_labels
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        output_dict=True,
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HYBRID TEST RESULTS")
    print("=" * 70)

    print()
    print(
        f"Correct:           "
        f"{correct}/{total}"
    )

    print(
        f"Test Accuracy:     "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro Precision:   "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Macro Recall:      "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"Macro F1:          "
        f"{f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("Per-Class Results")
    print("-" * 70)

    for class_name in test_dataset.classes:

        metrics = report[class_name]

        print()
        print(class_name)

        print(
            f"  Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"  Recall:    "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"  F1:        "
            f"{metrics['f1-score']:.4f}"
        )

        print(
            f"  Support:   "
            f"{int(metrics['support'])}"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("Confusion Matrix")
    print("-" * 70)

    print(cm)

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    results = {

        "model":
            "MobileNetV3 + DeiT-Tiny Hybrid",

        "architecture": {
            "mobilenet_features":
                576,

            "deit_features":
                192,

            "fused_features":
                768
        },

        "dataset": {

            "test_samples":
                total,

            "num_classes":
                len(test_dataset.classes),

            "classes":
                test_dataset.classes
        },

        "metrics": {

            "accuracy":
                float(accuracy),

            "precision_macro":
                float(precision),

            "recall_macro":
                float(recall),

            "f1_macro":
                float(f1),

            "correct":
                int(correct),

            "total":
                int(total)
        },

        "per_class":
            report,

        "confusion_matrix":
            cm.tolist()
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Plot confusion matrix
    # --------------------------------------------------------

    plt.figure(
        figsize=(11, 9)
    )

    plt.imshow(cm)

    plt.title(
        "MobileNetV3 + DeiT-Tiny Hybrid - Confusion Matrix"
    )

    plt.colorbar()

    tick_positions = np.arange(
        len(test_dataset.classes)
    )

    plt.xticks(
        tick_positions,
        test_dataset.classes,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_positions,
        test_dataset.classes
    )

    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                )
            )

    plt.ylabel(
        "True Label"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Evaluation Complete")
    print("=" * 70)

    print()
    print("Results:")
    print(RESULTS_PATH)

    print()
    print("Confusion matrix:")
    print(CONFUSION_MATRIX_PATH)

    print()
    print(
        "✓ Hybrid test evaluation successful."
    )


if __name__ == "__main__":
    main()