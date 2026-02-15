import torch
import torch.nn as nn

class Generator(nn.Module):
    """DCGAN Generator"""
    def __init__(self, noise_dim=100, condition_dim=8, img_size=64):
        super().__init__()
        self.init_size = img_size // 16
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 256 * self.init_size * self.init_size),
            nn.BatchNorm1d(256 * self.init_size * self.init_size),
            nn.ReLU(True)
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh()
        )
    
    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, self.init_size, self.init_size)
        return self.conv_blocks(x)


class Discriminator(nn.Module):
    """DCGAN Discriminator"""
    def __init__(self, condition_dim=8, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.condition_dim = condition_dim
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3 + condition_dim, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 8, 1, 0), nn.Sigmoid()
        )
    
    def forward(self, img, condition):
        condition = condition.view(condition.size(0), self.condition_dim, 1, 1)
        condition = condition.expand(-1, -1, self.img_size, self.img_size)
        x = torch.cat([img, condition], dim=1)
        return self.conv_blocks(x).view(-1, 1)
