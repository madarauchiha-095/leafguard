import os
import json
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.models.hybrid import MobileNetDeiTHybrid


# ============================================================
# Configuration
# ============================================================

BASE_DIR = r"C:\LeafGuard"

TRAIN_DIR = os.path.join(
    BASE_DIR, "data", "processed", "train"
)

VAL_DIR = os.path.join(
    BASE_DIR, "data", "processed", "val"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR, "results", "metrics"
)

MODEL_PATH = os.path.join(
    MODEL_DIR, "hybrid_best.pth"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR, "hybrid_training_history.json"
)


# ============================================================
# Training parameters
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

NUM_EPOCHS = 20

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

EARLY_STOPPING_PATIENCE = 5

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("LeafGuard - MobileNetV3 + DeiT-Tiny Hybrid Training")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
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
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    train_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=15
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([

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

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print()
    print("Dataset sizes:")

    print(
        f"Train:      {len(train_dataset)}"
    )

    print(
        f"Validation: {len(val_dataset)}"
    )

    print()
    print("Classes:")

    for idx, class_name in enumerate(
        train_dataset.classes
    ):

        print(
            f"{idx}: {class_name}"
        )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    targets = np.array(
        train_dataset.targets
    )

    class_counts = np.bincount(
        targets,
        minlength=len(
            train_dataset.classes
        )
    )

    class_weights = (
        len(targets)
        /
        (
            len(class_counts)
            *
            class_counts
        )
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(device)

    print()
    print("Class weights:")
    print(class_weights)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Loading pretrained hybrid model...")

    model = MobileNetDeiTHybrid(
        num_classes=len(
            train_dataset.classes
        )
    )

    model = model.to(device)

    print()
    print(
        f"MobileNet features: "
        f"{model.mobilenet_feature_dim}"
    )

    print(
        f"DeiT features: "
        f"{model.deit_feature_dim}"
    )

    print(
        f"Fused features: "
        f"{model.mobilenet_feature_dim + model.deit_feature_dim}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = []

    best_val_f1 = 0.0

    epochs_without_improvement = 0

    total_start = time.time()

    # ========================================================
    # Training
    # ========================================================

    print()
    print("=" * 70)
    print("Starting hybrid training")
    print("=" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        epoch_start = time.time()

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_losses = []

        train_predictions = []
        train_labels = []

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            train_losses.append(
                loss.item()
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            train_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )

            train_labels.extend(
                labels.detach()
                .cpu()
                .numpy()
            )

        # ----------------------------------------------------
        # Train metrics
        # ----------------------------------------------------

        train_loss = np.mean(
            train_losses
        )

        train_accuracy = accuracy_score(
            train_labels,
            train_predictions
        )

        train_precision = precision_score(
            train_labels,
            train_predictions,
            average="macro",
            zero_division=0
        )

        train_recall = recall_score(
            train_labels,
            train_predictions,
            average="macro",
            zero_division=0
        )

        train_f1 = f1_score(
            train_labels,
            train_predictions,
            average="macro",
            zero_division=0
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_losses = []

        val_predictions = []
        val_labels = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=torch.cuda.is_available()
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )

                val_losses.append(
                    loss.item()
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_predictions.extend(
                    predictions.cpu()
                    .numpy()
                )

                val_labels.extend(
                    labels.cpu()
                    .numpy()
                )

        # ----------------------------------------------------
        # Validation metrics
        # ----------------------------------------------------

        val_loss = np.mean(
            val_losses
        )

        val_accuracy = accuracy_score(
            val_labels,
            val_predictions
        )

        val_precision = precision_score(
            val_labels,
            val_predictions,
            average="macro",
            zero_division=0
        )

        val_recall = recall_score(
            val_labels,
            val_predictions,
            average="macro",
            zero_division=0
        )

        val_f1 = f1_score(
            val_labels,
            val_predictions,
            average="macro",
            zero_division=0
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_f1
        )

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = (
            time.time() - epoch_start
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        print()
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print("-" * 70)

        print(
            f"Train Loss:      "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy:  "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Train Precision: "
            f"{train_precision:.4f}"
        )

        print(
            f"Train Recall:    "
            f"{train_recall:.4f}"
        )

        print(
            f"Train F1:        "
            f"{train_f1:.4f}"
        )

        print()

        print(
            f"Val Loss:        "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy:    "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Val Precision:   "
            f"{val_precision:.4f}"
        )

        print(
            f"Val Recall:      "
            f"{val_recall:.4f}"
        )

        print(
            f"Val F1:          "
            f"{val_f1:.4f}"
        )

        print()

        print(
            f"Learning Rate:   "
            f"{current_lr:.7f}"
        )

        print(
            f"Epoch Time:      "
            f"{epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append({

            "epoch": epoch,

            "train_loss": float(
                train_loss
            ),

            "train_accuracy": float(
                train_accuracy
            ),

            "train_precision": float(
                train_precision
            ),

            "train_recall": float(
                train_recall
            ),

            "train_f1": float(
                train_f1
            ),

            "val_loss": float(
                val_loss
            ),

            "val_accuracy": float(
                val_accuracy
            ),

            "val_precision": float(
                val_precision
            ),

            "val_recall": float(
                val_recall
            ),

            "val_f1": float(
                val_f1
            ),

            "learning_rate": float(
                current_lr
            ),

            "epoch_time": float(
                epoch_time
            )
        })

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print()
            print(
                "✓ New best hybrid model saved!"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                f"No improvement "
                f"({epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ========================================================
    # Save history
    # ========================================================

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    total_time = (
        time.time() - total_start
    )

    # ========================================================
    # Complete
    # ========================================================

    print()
    print("=" * 70)
    print("Hybrid Training Complete")
    print("=" * 70)

    print()
    print(
        f"Best Validation F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Total Training Time: "
        f"{total_time / 60:.2f} minutes"
    )

    print()
    print("Best model:")
    print(MODEL_PATH)

    print()
    print("Training history:")
    print(HISTORY_PATH)


if __name__ == "__main__":
    main()