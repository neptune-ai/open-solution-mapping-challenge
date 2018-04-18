import torch.nn as nn
import math

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


def get_pad(stride, kernel, dilation=1, mode="downsample"):
    if mode=="downsample":
        return int(math.ceil((1-stride+dilation*kernel-1)/2))
    elif mode=="upsample":
        if kernel-stride>=0 and (kernel-stride)%2==0:
            return (int((kernel-stride)/2), 0)
        elif kernel-stride<0:
            return (0, stride - kernel)
        else:
            return (int(math.ceil((kernel-stride)/2)), 1)
    else:
        return 0

