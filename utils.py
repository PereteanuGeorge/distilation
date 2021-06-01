import os
import sys
import time
from torch.utils.data import DataLoader, random_split
import torch
import torchvision

import torchvision.transforms as transforms
import tenseal as ts
from random import randint

torch.manual_seed(73)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_simple = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train, valid = random_split(trainset, [45000, 5000])
trainloader = torch.utils.data.DataLoader(
    train, batch_size=128, shuffle=True, num_workers=2)

validloader = DataLoader(valid, batch_size=128)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

cuda_dev = '0' #GPU device 0 (can be changed if multiple GPUs are available)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")

print('Device: ' + str(device))
if use_cuda:
    print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))

def context():
    bits_scale = 40

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=32768,
        coeff_mod_bit_sizes=[60, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 60]
    )
    # set the scale
    context.global_scale = pow(2, bits_scale)

    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()
    return context

context = context()

# Sample an image
def load_input():
    idx = randint(1, len(testloader))
    i = 0
    for data, target in testloader:
        if i == idx:
            data_to_return = data
            target_to_return = target
        i += 1
    return data_to_return, target_to_return


def load_weights(self):
    pretrained_dict = torch.load('model.pt')
    model_dict = self.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    self.load_state_dict(model_dict)
