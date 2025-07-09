import copy

class EMA:
    def __init__(self, model, decay=0.999):
        #decay: коэффициент сглаживания EMA, обычно около 0.999
        self.decay = decay
        self.model = model
        self.shadow = copy.deepcopy(model.state_dict())  # Копия весов модели для EMA
        self.backup = None  # Для хранения текущих весов при подмене EMA весов

    def update(self):
        #Обновить EMA веса после одного шага оптимизации.
        current_state = self.model.state_dict()
        for key in current_state:
            self.shadow[key] = self.decay * self.shadow[key] + (1 - self.decay) * current_state[key]

    def apply_shadow(self):
        """
        Подменить текущие веса модели на EMA веса (например, для валидации).
        Сохраняет текущие веса в backup.
        """
        self.backup = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow)
        self.model.eval()

    def restore(self):
        #Вернуть обратно обычные веса модели после применения EMA.
        if self.backup is not None:
            self.model.load_state_dict(self.backup)
            self.backup = None
