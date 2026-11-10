import torch
import torch.nn as nn
import timm


class DeiTTinyClassifier(nn.Module):

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        self.model = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    print()
    print("Loading pretrained DeiT-Tiny...")

    model = DeiTTinyClassifier(
        num_classes=10
    ).to(device)

    x = torch.randn(
        4, 3, 224, 224
    ).to(device)

    with torch.no_grad():
        output = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")

    print()
    print("✓ DeiT-Tiny model test successful.")
