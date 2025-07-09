import torch
import torch.nn as nn
from src.SEBlock import SEblock
from src.activations import SquaredReLU

class Resblock(nn.Module):
    def __init__(self, channels, layerscale_init_value=1e-6):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.se = SEblock(channels)
        self.act2 = nn.ReLU()  # активация после сложения

        # LayerScale параметр (масштаб на каждый канал)
        self.layerscale = nn.Parameter(torch.ones((channels, 1, 1)) * layerscale_init_value)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        out = out * self.layerscale  # LayerScale - масштабируем выход

        out = out + identity  # остаточная связь

        out = self.act2(out)  # финальная активация

        return out
