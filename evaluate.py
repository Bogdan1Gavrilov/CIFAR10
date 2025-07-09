import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

from models.ResModelV3 import ResNetV3
from src.prepare_data import get_data_loaders


def evaluate(model, device, test_loader, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test accuracy: {acc*100:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Предсказание")
    plt.ylabel("Реальность")
    plt.title("Матрица ошибок модель3")
    plt.tight_layout()
    plt.savefig('runs/confusion_resnetV3.png')
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders()
    classes = train_loader.dataset.classes
    model = ResNetV3()
    model.load_state_dict(torch.load('weights/cifar10_res50netV31.pth', map_location=device))
    model.to(device)

    evaluate(model, device, test_loader, classes)

if __name__ == '__main__':
    main()