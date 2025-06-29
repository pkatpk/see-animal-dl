import torch.nn as nn
import torchvision.models as models

# ---------------------------------------------------------------------
#  CNN01
# ---------------------------------------------------------------------
class CNN01(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),   nn.BatchNorm2d(8),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),  nn.BatchNorm2d(16),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
        )
        flat = 32 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ---------------------------------------------------------------------
#  CNN02
# ---------------------------------------------------------------------
class CNN02(nn.Module):
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

# ---------------------------------------------------------------------
#  ResNet18 Pretrained (Transfer Learning)
# ---------------------------------------------------------------------
class ResNetTransfer(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 224, drop: float = 0.3):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze base

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
