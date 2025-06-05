import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=64, val_split=0.2):
    #конвертируем данные в тензоры и нормализуем их
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    #скачиваем тренировочный датасет
    train_dataset_full = datasets.FashionMNIST(
        root="datasets",
        train=True,
        download=True,
        transform=transform
    )
    #Разделяем данные на тренировочные и валидационные
    val_size = int(len(train_dataset_full) * val_split)
    train_size = len(train_dataset_full) - val_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    #Загружаем тестовый датасет
    test_dataset = datasets.FashionMNIST(
        root="datasets",
        train=False, #То же самое как с train_dataset только в данном случае train = false
        download=True,
        transform=transform
    )

    #Оборачиваем всё в DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader