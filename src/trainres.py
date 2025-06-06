import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from m2silu import FashionCNNv2silu
import os

def train():
    #Настройки
    batch_size = 64
    lr = 0.001
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    # Преобразования
    #Преобразуем данные в Tensor и нормируем их в [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    #Модель, функция потерь, оптимизатор

    model = FashionCNNv2silu().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #TensorBoard
    #Создаем логгер в папке runs/exp2
    writer = SummaryWriter(log_dir="runs/exp2")

    best_accuracy = 0.0

    #Цикл обучения

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Тестирование
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_loss /= len(test_loader.dataset)
        accuracy = correct / total

        #Логирование
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)

        print(f"Epoch[{epoch+1}/{num_epochs}], "
              f"Train_loss: {train_loss:.4f}, Test_loss: {test_loss:.4f}, Test_Acc: {accuracy:.4f}")
        
        #Сохранение лучшей модели

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/fashion_cnn_best.pth")

    writer.close()

if __name__ == "__main__":
    train()