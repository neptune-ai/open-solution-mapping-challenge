import math

import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


def get_downsample_pad(stride, kernel, dilation=1):
    return int(math.ceil((1 - stride + dilation * kernel - 1) / 2))


def get_upsample_pad(stride, kernel, dilation=1):
    if kernel - stride >= 0 and (kernel - stride) % 2 == 0:
        return (int((kernel - stride) / 2), 0)
    elif kernel - stride < 0:
        return (0, stride - kernel)
    else:
        return (int(math.ceil((kernel - stride) / 2)), 1)
