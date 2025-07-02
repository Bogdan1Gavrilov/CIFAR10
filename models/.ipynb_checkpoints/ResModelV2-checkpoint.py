#МОДЕЛЬ ИДЕНТИЧНАЯ ПЕРВОЙ ВЕРСИИ, ЕДИНСТВЕННОЕ ИЗМЕНЕНИЕ НА ДАННЫЙ МОМЕНТ - ЗАМЕНА DROPOUT НА КАСТОМНЫЙ DROPBLOCK
import torch
import torch.nn as nn
from src.Dropblock import DropBlock2d
from .resblock import Resblock #точку необходимо писать чтобы было понятно, что resblock лежит в той же папке, иначе не работает

class ResNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()
        self.dropout = DropBlock2d(block_size=3, drop_prob=0.1)

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            self.act,
            Resblock(32),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            self.act,
            Resblock(64),
            nn.MaxPool2d(2, 2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            self.act,
            Resblock(128),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.Dropout(0.3),
            self.act,
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x