import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import device, load_weights


class Attack2(nn.Module):
    def __init__(self):
        super(Attack2, self).__init__()
        #self.base = nn.Sequential(
        #    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #)
        self.conv1 = nn.Conv2d(3,16, kernel_size = 1, stride = 1)
        self.conv2 = nn.Conv2d(16,32, kernel_size = 1, stride = 1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out
