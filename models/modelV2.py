import torch
import torch.nn as nn


class CIFAR10CNNV2(nn.Module):
    def __init__(self):
        super(CIFAR10CNNV2, self).__init__()
        self.act = nn.SiLU() #Чтобы каждый раз не прописывать silu создам блок активации 
        #Первый сверточный блок
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)#Добавляем нормализацию батчей для ускорения обучения
        self.pool1 = nn.MaxPool2d(2, 2)

        #Второй сверточный блок
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        #Третий сверточный блок
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        #Линейный слой
        self.fc1 = nn.Linear(128*4*4, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.act(self.bn1(self.conv1(x))))  # → [B, 32, 16, 16]
        x = self.pool2(self.act(self.bn2(self.conv2(x))))  # → [B, 64, 8, 8]
        x = self.pool3(self.act(self.bn3(self.conv3(x))))  # → [B, 128, 4, 4]
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x