import torch
import torch.nn as nn
import torch.nn.functional as F
import tenseal as ts

from utils import device, load_weights


class Server2(nn.Module):
    def __init__(self):
        super(Server2, self).__init__()
        self.base = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        # self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        out = self.base(x)
        # out = self.dropout(out)
        return out


model3 = Server2().to(device)

load_weights(model3)


class EncServer2(torch.nn.Module):
    def __init__(self, torch_nn):
        super(EncServer2, self).__init__()

        self.conv1_weight = torch_nn.base.weight.data.view(
            torch_nn.base.out_channels, torch_nn.base.kernel_size[0],
            torch_nn.base.kernel_size[1]
        ).tolist()

    def forward(self, enc_x, windows_nb):
        enc_channels = []
        for kernel in self.conv1_weight:
              y = enc_x.conv2d_im2col(kernel, windows_nb)
              enc_channels.append(y)
        enc_x = ts.CKKSVector.pack_vectors(enc_x)
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# first_part = EncServer2(model3).to(device)
