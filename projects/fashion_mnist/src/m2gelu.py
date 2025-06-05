import torch.nn as nn

class FashionCNNv2gelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), #(1, 28, 28) -> (64, 28, 28) сверточный блок принимает чб изображение(1 канал), накладывает 64 фильтра(64 выходных канала)
            nn.BatchNorm2d(64),
            nn.GELU(), 
            nn.MaxPool2d(2),# Отбираем признаки "окном" 2х2: (64, 28, 28) -> (64, 14, 14)
            nn.Dropout(0,25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),#(64, 14, 14) -> (128, 14, 14)
            nn.GELU(),
            nn.MaxPool2d(2),# (128, 14, 14) -> (128, 7, 7)
            nn.Dropout(0,25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 200, kernel_size=3, padding=1),
            nn.BatchNorm2d(200),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        self.linear = nn.Sequential(
            nn.Flatten(), #Переводим 3D "массив" в вектор, перемножая количество признаков
            nn.Linear(200*7*7, 256),# Стандартный слой линейной обработки
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)#Выходнsой слой на 10 классов
        )

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.linear(X)
        return X
