import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3Classifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        if pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None

        self.model = mobilenet_v3_small(weights=weights)

        # Get the input size of the original classifier
        in_features = self.model.classifier[-1].in_features

        # Replace final classifier
        self.model.classifier[-1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    print("=" * 60)
    print("LeafGuard - MobileNetV3 Model Test")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = MobileNetV3Classifier(
        num_classes=10,
        pretrained=True
    ).to(device)

    # Test forward pass
    dummy_input = torch.randn(
        4, 3, 224, 224
    ).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    print("\n✓ MobileNetV3 model test successful.")