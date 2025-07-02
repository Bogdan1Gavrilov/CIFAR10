import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.prepare_data import get_data_loaders
from models.modelV2 import CIFAR10CNNV2

#1. Настраиваем устройство и гиперпараметры
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10
lr = 0.001
batch_size = 64

#2.Даталоадер
train_loader, test_loader = get_data_loaders(batch_size=batch_size)

#3.Функция потерь, модель, оптимизатор

model = CIFAR10CNNV2().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#4. Логгер
writer = SummaryWriter(log_dir="runs/cifar10_exp")

#5. Цикл обучения
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        #Обнуляем градиенты
        optimizer.zero_grad()

        #Прямой проход
        outputs = model(images)

        #Потери и градиенты
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #Статистика
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct/total

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", epoch_acc, epoch)

#6.Сохранение весов модели
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/cifar10_siluV2.pth")

writer.close()