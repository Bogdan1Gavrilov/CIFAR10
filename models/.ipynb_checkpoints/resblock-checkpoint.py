import torch.nn as nn
from src.SEBlock import SEblock

class Resblock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.Mish(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.SiLU()
        self.se = SEblock(channels)

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return self.act(x + out)