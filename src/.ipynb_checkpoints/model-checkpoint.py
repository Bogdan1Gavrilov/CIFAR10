import torch.nn as nn

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), #(1, 28, 28) -> (32, 28, 28) сверточный блок принимает чб изображение(1 канал), накладывает 32 фильтра(32 выходных канала)
            nn.LeakyReLU(), 
            nn.MaxPool2d(2),# Отбираем признаки "окном" 2х2: (32, 28, 28) -> (32, 14, 14)
    
            nn.Conv2d(32, 64, kernel_size=3, padding=1),#(32, 14, 14) -> (64, 14, 14)
            nn.LeakyReLU(),
            nn.MaxPool2d(2),# (64, 14, 14) -> (64, 7, 7)
    
            nn.Flatten(), #Переводим 3D "массив" в вектор, перемножая количество признаков
            nn.Linear(64*7*7, 128),# Стандартный слой линейной обработки
            nn.LeakyReLU(),
            nn.Linear(128, 10)#Выходной слой на 10 классов
        )

    def forward(self, X):
        return self.model(X)
