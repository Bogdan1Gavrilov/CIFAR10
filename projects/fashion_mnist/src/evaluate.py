import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from m2gelu import FashionCNNv2gelu


#Добавляем нормализацию данных и приводим их к виду тензора

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
#Разбиваем данные на батчи по 64 изображения, не перемешиваем
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#Настраиваем код для работы на видеокарте
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Загружаем модель и лучшие веса
model = FashionCNNv2gelu().to(device)
model.load_state_dict(torch.load("models/fashion_cnn_best.pth", map_location=device))
#Настраиваем 

#Переводим модель в режим тестирования
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {acc:.4f}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
           xticklabels=test_dataset.classes, 
           yticklabels=test_dataset.classes)
plt.xlabel("Предсказание")
plt.ylabel("Реальность")
plt.title("Матрица ошибок")
plt.show()