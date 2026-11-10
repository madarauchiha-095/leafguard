import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.models.mobilenet import MobileNetV3Classifier


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("C:/LeafGuard")

DATA_ROOT = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

NUM_CLASSES = 10

EPOCHS = 20

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001

PATIENCE = 5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLAHE TRANSFORM
# ============================================================

class CLAHETransform:
    """
    Apply CLAHE to the L channel in LAB color space.

    This enhances local contrast while preserving
    the overall color information.
    """

    def __init__(
        self,
        clip_limit=2.0,
        tile_grid_size=(8, 8)
    ):

        self.clip_limit = clip_limit

        self.tile_grid_size = tile_grid_size

        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

    def __call__(self, image):

        # PIL -> NumPy
        image = np.array(image)

        # RGB -> LAB
        lab = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2LAB
        )

        # Split LAB channels
        l_channel, a_channel, b_channel = (
            cv2.split(lab)
        )

        # Apply CLAHE only to luminance
        l_channel = self.clahe.apply(
            l_channel
        )

        # Merge channels
        lab = cv2.merge([
            l_channel,
            a_channel,
            b_channel
        ])

        # LAB -> RGB
        image = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2RGB
        )

        # NumPy -> PIL
        from PIL import Image

        return Image.fromarray(
            image
        )


# ============================================================
# TRANSFORMS
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    CLAHETransform(
        clip_limit=2.0,
        tile_grid_size=(8, 8)
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
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    CLAHETransform(
        clip_limit=2.0,
        tile_grid_size=(8, 8)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# DATA
# ============================================================

def create_dataloaders():

    train_dataset = datasets.ImageFolder(
        DATA_ROOT / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        DATA_ROOT / "val",
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        DATA_ROOT / "test",
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataset):

    targets = torch.tensor(
        dataset.targets
    )

    class_counts = torch.bincount(
        targets,
        minlength=NUM_CLASSES
    ).float()

    weights = 1.0 / class_counts

    weights = (
        weights / weights.mean()
    )

    return weights


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            device_type="cuda",
            enabled=(DEVICE.type == "cuda")
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

        running_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    loss = (
        running_loss /
        len(loader.dataset)
    )

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

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            with autocast(
                device_type="cuda",
                enabled=(DEVICE.type == "cuda")
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            running_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu()
                .numpy()
            )

            all_labels.extend(
                labels.cpu()
                .numpy()
            )

    loss = (
        running_loss /
        len(loader.dataset)
    )

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

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LeafGuard - MobileNetV3 + CLAHE Training")
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

    print(
        "\nCLAHE configuration:"
    )

    print(
        "Clip limit: 2.0"
    )

    print(
        "Tile grid:  8 x 8"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders()

    print("\nDataset sizes:")

    print(
        f"Train:      {len(train_dataset)}"
    )

    print(
        f"Validation: {len(val_dataset)}"
    )

    print(
        f"Test:       {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = (
        calculate_class_weights(
            train_dataset
        )
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nLoading pretrained MobileNetV3..."
    )

    model = MobileNetV3Classifier(
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(DEVICE)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda")
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "learning_rate": []
    }

    best_val_f1 = 0.0

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Starting CLAHE training")
    print("=" * 70)

    total_start = time.time()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print("-" * 70)

        (
            train_loss,
            train_accuracy,
            train_precision,
            train_recall,
            train_f1
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler
        )

        (
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1
        ) = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val_f1
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        epoch_time = (
            time.time() -
            epoch_start
        )

        print(
            f"Train Loss:      {train_loss:.4f}"
        )

        print(
            f"Train Accuracy:  {train_accuracy:.4f}"
        )

        print(
            f"Train Precision: {train_precision:.4f}"
        )

        print(
            f"Train Recall:    {train_recall:.4f}"
        )

        print(
            f"Train F1:        {train_f1:.4f}"
        )

        print()

        print(
            f"Val Loss:        {val_loss:.4f}"
        )

        print(
            f"Val Accuracy:    {val_accuracy:.4f}"
        )

        print(
            f"Val Precision:   {val_precision:.4f}"
        )

        print(
            f"Val Recall:      {val_recall:.4f}"
        )

        print(
            f"Val F1:          {val_f1:.4f}"
        )

        print()

        print(
            f"Learning Rate:   {current_lr:.7f}"
        )

        print(
            f"Epoch Time:      {epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["train_precision"].append(
            train_precision
        )

        history["train_recall"].append(
            train_recall
        )

        history["train_f1"].append(
            train_f1
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        history["val_precision"].append(
            val_precision
        )

        history["val_recall"].append(
            val_recall
        )

        history["val_f1"].append(
            val_f1
        )

        history["learning_rate"].append(
            current_lr
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            epochs_without_improvement = 0

            checkpoint_path = (
                MODEL_DIR /
                "mobilenet_v3_clahe_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_accuracy":
                        val_accuracy,
                    "class_names":
                        train_dataset.classes,
                    "clahe_clip_limit": 2.0,
                    "clahe_tile_grid":
                        (8, 8)
                },
                checkpoint_path
            )

            print(
                "\n✓ New best CLAHE model saved!"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement "
                f"({epochs_without_improvement}/{PATIENCE})"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_path = (
        METRICS_DIR /
        "mobilenet_clahe_training_history.json"
    )

    with open(
        history_path,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    total_time = (
        time.time() -
        total_start
    )

    print("\n" + "=" * 70)
    print("CLAHE Training Complete")
    print("=" * 70)

    print(
        f"\nBest Validation F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Total Training Time: "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        "\nBest model:"
    )

    print(
        MODEL_DIR /
        "mobilenet_v3_clahe_best.pth"
    )

    print(
        "\nTraining history:"
    )

    print(
        history_path
    )


if __name__ == "__main__":
    main()