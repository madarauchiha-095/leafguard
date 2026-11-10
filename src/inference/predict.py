"""Reproducible inference using the frozen hybrid checkpoint.

Example: python -m src.inference.predict --image path/to/leaf.jpg
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.explainability.gradcampp import explain_pil
from src.inference.common import CLASS_NAMES, build_model, make_transform
from src.security.integrity import verify_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--explanation", type=Path, default=None,
                        help="Optional path to save the Grad-CAM++ overlay")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    integrity = verify_checkpoint()
    model = build_model("hybrid", device=device)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    x = make_transform()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(x).softmax(dim=1)[0].cpu().numpy()
    order = np.argsort(probabilities)[::-1][:3]
    payload = {
        "image": str(args.image),
        "predicted_class": CLASS_NAMES[int(order[0])],
        "confidence": float(probabilities[int(order[0])]),
        "top_3": [
            {"class": CLASS_NAMES[int(index)], "confidence": float(probabilities[int(index)])}
            for index in order
        ],
        "model_integrity": integrity,
        "model": "MobileNetV3-Small + DeiT-Tiny hybrid",
    }
    if args.explanation:
        explanation = explain_pil(model, image, model_key="hybrid")
        args.explanation.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(explanation["overlay"]).save(args.explanation)
        payload["explanation"] = str(args.explanation)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
