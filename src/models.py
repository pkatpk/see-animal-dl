import torch.nn as nn

# ---------------------------------------------------------------------
#  CNN01  (8-16-32)  – σούπερ μίνιμαλ
# ---------------------------------------------------------------------
class CNN01(nn.Module):
    """
    Super-minimal CNN:
        Conv(3→8)  → BN → ReLU → MaxPool
        Conv(8→16) → BN → ReLU → MaxPool
        Conv(16→32)→ BN → ReLU → MaxPool
        Flatten → FC(32*H*W → 128) → ReLU → Dropout
                 → FC(128 → num_classes)
    """
    def __init__(self, num_classes: int, img_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),   nn.BatchNorm2d(8),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),  nn.BatchNorm2d(16),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
        )
        flat = 32 * (img_size // 8) * (img_size // 8)        # ↓3 MaxPool → /8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------
#  CNN02  (32-64-128)  – το προηγούμενο «MinimalCNN»
# ---------------------------------------------------------------------
class CNN02(nn.Module):
    """
    3-layer CNN με 32-64-128 filters (όπως το παλιό MinimalCNN).
    """
    def __init__(self, num_classes: int, img_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128,3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        flat = 128 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
