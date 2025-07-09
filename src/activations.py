import torch
import torch.nn as nn

class SquaredReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x) ** 2

class SquaredGeLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x) ** 2