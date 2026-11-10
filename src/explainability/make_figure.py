"""Create a compact publication-style panel from generated explanation examples."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.inference.common import ROOT


def main():
    root = ROOT / "results/explainability"
    candidates = [
        ("plantvillage_correct", "PlantVillage · correct"),
        ("plantvillage_difficult", "PlantVillage · low-margin"),
        ("plantvillage_failure", "PlantVillage · failure"),
        ("plantdoc_correct", "PlantDoc · correct"),
        ("plantdoc_failure", "PlantDoc · failure"),
    ]
    available = []
    for folder, title in candidates:
        original = root / folder / "original.jpg"
        overlay = root / folder / "gradcampp_overlay.png"
        metadata = root / folder / "metadata.json"
        if original.exists() and overlay.exists() and metadata.exists():
            info = json.loads(metadata.read_text(encoding="utf-8"))
            available.append((folder, title, info))
    if not available:
        raise FileNotFoundError("No generated Grad-CAM++ examples found; run generate_examples first.")
    fig, axes = plt.subplots(len(available), 2, figsize=(8, 3.0 * len(available)), squeeze=False)
    for row, (folder, title, info) in enumerate(available):
        original = Image.open(root / folder / "original.jpg").convert("RGB")
        overlay = Image.open(root / folder / "gradcampp_overlay.png").convert("RGB")
        axes[row, 0].imshow(original)
        axes[row, 1].imshow(overlay)
        axes[row, 0].set_title(f"{title} · original", fontsize=10)
        axes[row, 1].set_title(
            f"True: {info['true_class'].replace('___', ' / ').replace('_', ' ')}\n"
            f"Pred: {info['predicted_class'].replace('___', ' / ').replace('_', ' ')} · "
            f"{info['confidence']:.1%}", fontsize=9
        )
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Grad-CAM++ on hybrid predictions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    destination = ROOT / "results/final_figures/explainability_examples.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=280, bbox_inches="tight")
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
