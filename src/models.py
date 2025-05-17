# src/models.py
import torch.nn as nn

class CNN01(nn.Module):                   
    """
    Minimal 3-layer CNN (32-64-128 filters).
    """
    def __init__(self, num_classes: int, img_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
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
