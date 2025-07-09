import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def get_data_loaders(batch_size=64):
    # Трансформации для датасета
    transform = transforms.ToTensor()
    transformAugmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    # Загрузка CIFAR10
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transformAugmentation)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Создание DataLoader-ов
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader