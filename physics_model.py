import torch
import torch.nn as nn

class PhysicsPredictor(nn.Module):
    """Predicts physical attributes from galaxy images"""
    def __init__(self, img_size=64, num_attrs=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),  # 32x32
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),  # 16x16
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),  # 8x8
            nn.AdaptiveAvgPool2d(1)
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_attrs),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, img):
        x = self.encoder(img)
        return self.mlp(x)
