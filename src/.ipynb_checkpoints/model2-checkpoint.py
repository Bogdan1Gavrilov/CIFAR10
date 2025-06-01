import torch.nn as nn

class FashionCNNv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), #(1, 28, 28) -> (64, 28, 28) сверточный блок принимает чб изображение(1 канал), накладывает 32 фильтра(32 выходных канала)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2),# Отбираем признаки "окном" 2х2: (64, 28, 28) -> (64, 14, 14)
            nn.Dropout(0,25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),#(64, 14, 14) -> (128, 14, 14)
            nn.LeakyReLU(),
            nn.MaxPool2d(2),# (128, 14, 14) -> (128, 7, 7)
            nn.Dropout(0,25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 200, kernel_size=3, padding=1),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.linear = nn.Sequential(
            nn.Flatten(), #Переводим 3D "массив" в вектор, перемножая количество признаков
            nn.Linear(200*7*7, 256),# Стандартный слой линейной обработки
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)#Выходной слой на 10 классов
        )

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.linear(X)
        return X
