from torch.nn import functional as F
import torch
import torch.nn as nn

class BlurPool(nn.Module):
    def __init__(self, channels, stride=2, filt_size=3):
        super().__init__()
        self.stride = stride
        if filt_size == 3:
            kernel = torch.tensor([1., 2., 1.])
        elif filt_size == 5:
            kernel = torch.tensor([1., 4., 6., 4., 1.])
        else:
            raise ValueError ("Unsupported filter size")

        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        kernel = kernel[None, None, :, :].repeat((channels, 1, 1, 1))
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=self.stride, padding=1, groups=x.shape[1])