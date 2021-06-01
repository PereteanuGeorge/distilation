import torch
from torch import nn

from utils import device
import tenseal as ts


class ConvFWD(nn.Module):
    def __init__(self):
        super(ConvFWD, self).__init__()
        self.base = nn.Conv2d(1, 16, kernel_size=3, stride=1, bias=True)
    def forward(self, x):
        out = self.base(x)
        return out


conv = ConvFWD().to(device)


class EncServer2(torch.nn.Module):
    def __init__(self, torch_nn):
        super(EncServer2, self).__init__()

        self.conv1_weight = torch_nn.base.weight.data.view(
            torch_nn.base.out_channels, torch_nn.base.kernel_size[0],
            torch_nn.base.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.base.bias.data.tolist()

    def forward(self, enc_x, windows_nb):
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
