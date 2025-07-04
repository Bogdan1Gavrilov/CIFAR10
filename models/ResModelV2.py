#МОДЕЛЬ ИДЕНТИЧНАЯ ПЕРВОЙ ВЕРСИИ, ЕДИНСТВЕННОЕ ИЗМЕНЕНИЕ НА ДАННЫЙ МОМЕНТ - ЗАМЕНА DROPOUT НА КАСТОМНЫЙ DROPBLOCK
#UPD заменил функцию активации на mish
import torch
import torch.nn as nn
from src.Dropblock import DropBlock2d
from src.stochastic import StochasticBlock
from .resblock import Resblock #точку необходимо писать чтобы было понятно, что resblock лежит в той же папке, иначе не работает

class ResNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Mish()
        self.dropout = DropBlock2d(block_size=3, drop_prob=0.1)

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            self.act,
            StochasticBlock(Resblock(32), 0.2),
            StochasticBlock(Resblock(32), 0.2),
            StochasticBlock(Resblock(32), 0.2),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            self.act,
            StochasticBlock(Resblock(64), 0.2),
            StochasticBlock(Resblock(64), 0.2),
            StochasticBlock(Resblock(64), 0.2),
            nn.MaxPool2d(2, 2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            self.act,
            StochasticBlock(Resblock(128), 0.2),
            StochasticBlock(Resblock(128), 0.2),
            StochasticBlock(Resblock(128), 0.2),
            nn.MaxPool2d(2, 2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            self.act,
            StochasticBlock(Resblock(256), 0.2),
            StochasticBlock(Resblock(256), 0.2),
            StochasticBlock(Resblock(256), 0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.Dropout(0.3),
            self.act,
            nn.Linear(256, 100),
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