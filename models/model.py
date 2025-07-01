import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        
        #Первый сверточный блок
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        #Второй сверточный блок
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        #Линейный слой
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.silu(self.conv1(x)))
        x = self.pool2(F.silu(self.conv2(x)))
        x = x.view(-1, 64*8*8)
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return x