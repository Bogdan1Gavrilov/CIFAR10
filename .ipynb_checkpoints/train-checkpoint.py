import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR


from src.prepare_data import get_data_loaders
from models.ResModelV2 import ResNetV2

#1. Настраиваем устройство и гиперпараметры
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 12
lr = 0.001
batch_size = 64

#2.Даталоадер
train_loader, test_loader = get_data_loaders(batch_size=batch_size)

#3.Функция потерь, модель, оптимизатор

model = ResNetV2().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
steps_per_epoch = len(train_loader)
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=epochs,
    steps_per_epoch = steps_per_epoch,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4)

#4. Логгер
writer = SummaryWriter(log_dir="runs/cifar10_exp_res2")

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
        scheduler.step()

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
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
        writer.add_scalar("LearningRate", current_lr, epoch)


#6.Сохранение весов модели
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/cifar10_resnetV2.pth")

writer.close()