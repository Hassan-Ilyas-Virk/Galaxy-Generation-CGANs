"""
Physics Predictor Network for Galaxy CGAN
Predicts physical attributes (size, brightness, ellipticity, redshift) from galaxy images
"""

import torch
import torch.nn as nn


class PhysicsPredictor(nn.Module):
    """Lightweight CNN + MLP to predict physical attributes from images"""
    
    def __init__(self, img_size=64, num_attrs=4):
        super(PhysicsPredictor, self).__init__()
        
        # CNN Encoder (downsample image)
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        # MLP Head (predict attributes)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_attrs),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, img):
        x = self.encoder(img)
        x = self.mlp(x)
        return x
