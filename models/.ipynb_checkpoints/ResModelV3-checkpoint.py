import torch
import torch.nn as nn
from src.Dropblock import DropBlock2d
from src.stochastic import StochasticBlock
from .resblock import Resblock
from src.Blurpool import BlurPool

class ResNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Mish()
        self.dropout = DropBlock2d(block_size=3, drop_prob=0.1)

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            self.act,
            StochasticBlock(Resblock(32), drop_prob=0.2),
            StochasticBlock(Resblock(32), drop_prob=0.2),
            StochasticBlock(Resblock(32), drop_prob=0.2),
            BlurPool(channels=32, filt_size=3, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.act,
            StochasticBlock(Resblock(64), drop_prob=0.2),
            StochasticBlock(Resblock(64), drop_prob=0.2),
            StochasticBlock(Resblock(64), drop_prob=0.2),
            BlurPool(channels=64, filt_size=3, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            self.act,
            StochasticBlock(Resblock(128), drop_prob=0.2),
            StochasticBlock(Resblock(128), drop_prob=0.2),
            StochasticBlock(Resblock(128), drop_prob=0.2),
            BlurPool(channels=128, filt_size=3, stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            self.act,
            StochasticBlock(Resblock(256), drop_prob=0.2),
            StochasticBlock(Resblock(256), drop_prob=0.2),
            StochasticBlock(Resblock(256), drop_prob=0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.Dropout(0.3),
            self.act,
            nn.Linear(512, 100),
            nn.Dropout(0.3),
            self.act,
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
