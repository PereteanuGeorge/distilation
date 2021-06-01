import torch
from torch import nn

from utils import device, load_weights
import tenseal as ts

class ConvNet2(torch.nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        #self.conv12 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        #self.conv12 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        #self.linear7 = nn.Linear(1024, 256)
        #self.linear8 = nn.Linear(256, 64)
        #self.linear9 = nn.Linear(64, 32)
        #self.linear10 = nn.Linear(32, 10)
        #self.linear11 = nn.Linear(128, 10)



        #self.linear7 = nn.Linear(1024, 512)
        self.linear8 = nn.Linear(512, 256)
        self.linear9 = nn.Linear(256, 64)
        self.linear10 = nn.Linear(64, 10)

        #self.linear7 = nn.Linear(1024, 256)
        #self.linear8 = nn.Linear(256, 128)
        #self.linear9 = nn.Linear(128, 64)
        #self.linear10 = nn.Linear(64, 10)

    def forward(self, x):
        #out = self.conv12(x)
        #out = out.view(out.size(0), -1)
        #out = self.conv12(x)
        #out = out.view(out.size(0), -1)
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! VEZI AICI SIZE URUILEEEEEEEEEEE {out.shape}')

        #out = self.linear7(out)
        out = self.linear8(out)
        out = self.linear9(out)
        out = self.linear10(out)
        #out = self.linear11(out)
        #out = self.linear7(out)
        #out = self.linear8(out)
        #out = self.linear9(out)
        #out = self.linear10(out)
        return out

model2 = ConvNet2().to(device)

load_weights(model2)

#print(model2.conv12.weight.shape) #[256, 512, 3, 3]


class EncConvNet(torch.nn.Module):
    def __init__(self, torch_nn):
        super(EncConvNet, self).__init__()

        #self.conv12_weight = torch_nn.conv12.weight.data.view(torch_nn.conv12.out_channels, -1).tolist()
        #self.conv12_bias = torch_nn.conv12.bias.data.tolist()

        #self.fc7_weight = torch_nn.linear7.weight.T.data.tolist()
        #self.fc7_bias = torch_nn.linear7.bias.data.tolist()

        self.fc8_weight = torch_nn.linear8.weight.T.data.tolist()
        self.fc8_bias = torch_nn.linear8.bias.data.tolist()

        self.fc9_weight = torch_nn.linear9.weight.T.data.tolist()
        self.fc9_bias = torch_nn.linear9.bias.data.tolist()

        self.fc10_weight = torch_nn.linear10.weight.T.data.tolist()
        self.fc10_bias = torch_nn.linear10.bias.data.tolist()

    def forward(self, enc_x):
        #enc_channles = []
        #for kernel, bias in zip(self.conv12_weight, self.conv12_bias):
              #y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
              #enc_channels.append(y)
        enc_x = ts.CKKSVector.pack_vectors(enc_x)
        #enc_x = enc_x.mm(self.fc7_weight) + self.fc7_bias
        enc_x = enc_x.mm(self.fc8_weight) + self.fc8_bias
        enc_x = enc_x.mm(self.fc9_weight) + self.fc9_bias
        enc_x = enc_x.mm(self.fc10_weight) + self.fc10_bias
        #enc_x = enc_x.mm(self.fc11_weight) + self.fc11_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

enc_model = EncConvNet(model2).to(device)
