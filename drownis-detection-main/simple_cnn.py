# ============================================================
#  simple_cnn.py  —  Small CNN built from scratch
#  No pretrained weights — fully trained from zero
# ============================================================

import torch
import torch.nn as nn


class SimpleFatigueCNN(nn.Module):
    """
    A small, simple CNN for binary fatigue classification.
    Built entirely from scratch — no pretrained backbone.

    Architecture:
        Conv Block 1: 3 → 32 channels   (224→112)
        Conv Block 2: 32 → 64 channels  (112→56)
        Conv Block 3: 64 → 128 channels (56→28)
        Conv Block 4: 128 → 256 channels(28→14)
        Global Average Pooling → 256
        FC: 256 → 128 → 2
    """

    def __init__(self, num_classes=2):
        super(SimpleFatigueCNN, self).__init__()

        # ── Conv Block 1 ──
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 224 → 112
        )

        # ── Conv Block 2 ──
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 112 → 56
        )

        # ── Conv Block 3 ──
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 56 → 28
        )

        # ── Conv Block 4 ──
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28 → 14
        )

        # ── Global Average Pooling ──
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 14×14 → 1×1

        # ── Classifier Head ──
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

        # Initialize weights from scratch
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for conv layers, Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)      # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)    # (B, 256)
        x = self.classifier(x)       # (B, 2)
        return x


def build_simple_model(num_classes=2):
    """Build the simple CNN from scratch."""
    model = SimpleFatigueCNN(num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  SimpleFatigueCNN created from scratch")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable:,}")
    return model


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    return device
