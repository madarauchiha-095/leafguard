"""Shared model loading and preprocessing used by inference/evaluation."""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.deit_tiny import DeiTTinyClassifier
from src.models.hybrid import MobileNetDeiTHybrid
from src.models.mobilenet import MobileNetV3Classifier

ROOT = Path(__file__).resolve().parents[2]
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Corn_(maize)___Common_rust_",
    "Grape___Black_rot",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

MODEL_SPECS = {
    "mobilenet": {
        "name": "MobileNetV3-Small",
        "checkpoint": ROOT / "models/mobilenet_v3_best.pth",
        "clahe": False,
    },
    "mobilenet_clahe": {
        "name": "MobileNetV3-Small + CLAHE",
        "checkpoint": ROOT / "models/mobilenet_v3_clahe_best.pth",
        "clahe": True,
    },
    "deit_tiny": {
        "name": "DeiT-Tiny",
        "checkpoint": ROOT / "models/deit_tiny_best.pth",
        "clahe": False,
    },
    "hybrid": {
        "name": "MobileNetV3-Small + DeiT-Tiny Hybrid",
        "checkpoint": ROOT / "models/hybrid_best.pth",
        "clahe": False,
    },
}


class CLAHETransform:
    """Legacy evaluation CLAHE: LAB L channel, clip=2, tiles=(8, 8)."""
    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lightness = op.apply(lightness)
        enhanced = cv2.cvtColor(
            cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2RGB
        )
        return Image.fromarray(enhanced)


def make_transform(clahe: bool = False):
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if clahe:
        steps.append(CLAHETransform())
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(steps)


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def build_model(model_key: str, device=None, load_checkpoint=True):
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model key: {model_key}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_key in ("mobilenet", "mobilenet_clahe"):
        model = MobileNetV3Classifier(num_classes=len(CLASS_NAMES), pretrained=False)
    elif model_key == "deit_tiny":
        model = DeiTTinyClassifier(num_classes=len(CLASS_NAMES), pretrained=False)
    else:
        model = MobileNetDeiTHybrid(
            num_classes=len(CLASS_NAMES), pretrained_backbones=False
        )
    checkpoint_path = MODEL_SPECS[model_key]["checkpoint"]
    if load_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(unwrap_state_dict(checkpoint), strict=True)
    model.to(device).eval()
    return model
