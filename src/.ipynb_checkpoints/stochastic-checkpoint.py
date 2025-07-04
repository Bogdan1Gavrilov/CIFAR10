import torch
import torch.nn as nn

class StochasticBlock(nn.Module):
    def __init__(self, block, drop_prob=0.2):
        super().__init__()
        self.block = block
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and torch.rand(1).item() < self.drop_prob:
            return x
        else:
            return self.block(x)