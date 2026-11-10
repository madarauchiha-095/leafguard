import torch
import torch.nn as nn
import torchvision.models as models
import timm


class MobileNetDeiTHybrid(nn.Module):

    def __init__(self, num_classes=10, dropout=0.3, pretrained_backbones=True):

        super().__init__()

        # ====================================================
        # MobileNetV3-Small backbone
        # ====================================================

        mobilenet = models.mobilenet_v3_small(
            weights=(
                models.MobileNet_V3_Small_Weights.DEFAULT
                if pretrained_backbones else None
            )
        )

        # Remove final classifier
        self.mobilenet_features = mobilenet.features
        self.mobilenet_avgpool = mobilenet.avgpool

        mobilenet_feature_dim = (
            mobilenet.classifier[0].in_features
        )

        # ====================================================
        # DeiT-Tiny backbone
        # ====================================================

        self.deit = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained_backbones,
            num_classes=0
        )

        deit_feature_dim = self.deit.num_features

        # ====================================================
        # Feature dimensions
        # ====================================================

        self.mobilenet_feature_dim = (
            mobilenet_feature_dim
        )

        self.deit_feature_dim = (
            deit_feature_dim
        )

        fused_dim = (
            mobilenet_feature_dim
            + deit_feature_dim
        )

        # ====================================================
        # Fusion classifier
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                fused_dim,
                512
            ),

            nn.BatchNorm1d(
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                512,
                256
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        # ====================================================
        # MobileNetV3 features
        # ====================================================

        mobile_features = self.mobilenet_features(x)

        mobile_features = self.mobilenet_avgpool(
            mobile_features
        )

        mobile_features = torch.flatten(
            mobile_features,
            1
        )

        # ====================================================
        # DeiT features
        # ====================================================

        deit_features = self.deit(
            x
        )

        # ====================================================
        # Feature fusion
        # ====================================================

        fused_features = torch.cat(
            [
                mobile_features,
                deit_features
            ],
            dim=1
        )

        # ====================================================
        # Classification
        # ====================================================

        output = self.classifier(
            fused_features
        )

        return output


if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("LeafGuard - MobileNetV3 + DeiT-Tiny Hybrid Test")
    print("=" * 70)

    print()
    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )

    print()
    print("Loading hybrid model...")

    model = MobileNetDeiTHybrid(
        num_classes=10
    ).to(device)

    print()
    print(
        f"MobileNet feature dimension: "
        f"{model.mobilenet_feature_dim}"
    )

    print(
        f"DeiT feature dimension: "
        f"{model.deit_feature_dim}"
    )

    print(
        f"Fused feature dimension: "
        f"{model.mobilenet_feature_dim + model.deit_feature_dim}"
    )

    # --------------------------------------------------------
    # Test forward pass
    # --------------------------------------------------------

    print()
    print("Testing forward pass...")

    x = torch.randn(
        4,
        3,
        224,
        224
    ).to(device)

    model.eval()

    with torch.no_grad():

        output = model(x)

    print()
    print(
        f"Input shape:  "
        f"{x.shape}"
    )

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    print()
    print("✓ Hybrid model test successful.")
