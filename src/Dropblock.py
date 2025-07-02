import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock2d(nn.Module):
    def __init__(self, block_size=3, drop_prob=0.1):
        super(DropBlock2d, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        gamma = self._compute_gamma(x)
        mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) < gamma).float()#Создаем карту из 0 и 1 где 1 - возможные ценры для блока
        # mask = (torch.rand(batch_size, 1, h, w) < gamma).float() интерпретация строчки выше для лучшего понимания что происходит
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)#С помощью пулинга "растягиваем единицы" до размеров блока дропа
        mask = 1 - mask #Инвертируем маску (1 - центр блока дропа, а нам надо умножать, поэтому заменяем их на 0 и наоборот)
        x = x * mask * (mask.numel() / mask.sum())
        return x

    def _compute_gamma(self, x):
        #ФОРМУЛА УМНЫХ ДЯДЕК ИЗ СТАТЬИ 
        h, w = x.shape[2], x.shape[3]
        return self.drop_prob * (h * w) / ((self.block_size ** 2) * ((h - self.block_size + 1) * (w - self.block_size + 1)))