import torch

from m2gelu import FashionCNNv2gelu
import os

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Создаём модель
model = FashionCNNv2gelu().to(device)
model.load_state_dict(torch.load("models/fashion_cnn_best.pth", map_location=device))
model.eval()

# Пример входных данных (1 изображение, 1 канал, 28x28)
dummy_input = torch.randn(1, 1, 28, 28, device=device)

# Папка для модели
os.makedirs("export", exist_ok=True)

# Экспорт в ONNX
torch.onnx.export(
    model,                         # модель
    dummy_input,                   # вход
    "export/fashion_cnn.onnx",    # путь
    input_names=["input"],        # имена входов
    output_names=["output"],      # имена выходов
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11              # стандартная версия ONNX
)

print("Модель успешно экспортирована в export/fashion_cnn.onnx")