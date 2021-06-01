import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from attack_model import Attack
from attack_model2 import Attack2
from client import model1
from utils import device, trainset
from utils import testset
from server2 import model3
import numpy as np

criterion = nn.CrossEntropyLoss()
my_criterin = nn.MSELoss()

def reproducibilitySeed():
    """
    Ensure reproducibility of results; Seeds to 0
    """
    torch_init_seed = 73
    torch.manual_seed(torch_init_seed)
    numpy_init_seed = 73
    np.random.seed(numpy_init_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


reproducibilitySeed()


def train_attack2(epoch, optimizer2, attack2, validloader):
    print('\nEpoch: %d' % epoch)
    attack2.train()
    for batch_idx, (inputs, _) in enumerate(validloader):
        inputs = inputs.to(device)
        preds1 = attack2(inputs)
        targets = model3(inputs)
        loss = my_criterin(preds1, targets)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()


def train_attack(epoch, optimizer, attack, attack2, validloader):
    print('\nEpoch: %d' % epoch)
    attack.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(validloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            preds1 = attack2(inputs)
            preds = model1(preds1)
        outputs = attack(preds)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


def test(attack, attack2, testloader):
    attack.eval()
    attack2.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            preds1 = attack2(inputs)
            with torch.no_grad():
                preds = model1(preds1)
            outputs = attack(preds)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # print(f' TEEST accuracy e ceva {100.*correct/total} si loss e {test_loss/(batch_idx+1)}')
    return 100. * correct / total


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

f = open("results4-but-1-layer-remaining.txt", "a+")

lr = [0.07]
batch_size = [128]
num_samples = [5000, 2500, 500]
for samples in num_samples:
    for learning_rate in lr:
        for batch in batch_size:
            best_acc = 0
            best_batch = 0
            best_lr = 0
            attack2 = Attack2().to(device)
            attack = Attack(64).to(device)
            attack.apply(weight_reset)
            attack2.apply(weight_reset)
            optimizer = optim.SGD(attack.parameters(), lr=learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            optimizer2 = optim.SGD(attack2.parameters(), lr=learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            _, valid = random_split(trainset, [50000 - samples, samples])
            validloader = DataLoader(valid, batch_size=batch, shuffle=True)

            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch, shuffle=False, num_workers=2)

            for epoch in range(100):
                train_attack2(epoch, optimizer2, attack2, validloader)
                train_attack(epoch, optimizer, attack, attack2, validloader)
                acc = test(attack, attack2, testloader)
                if acc > best_acc:
                    best_batch = batch
                    best_lr = learning_rate
                    best_acc = acc
                scheduler.step()

            with open("results4-but-1-layer-remaining.txt", "a+") as f:
                text = "with # samples " + str(samples) + " we got best acc of " + str(best_acc) + ", batch of " + str(best_batch) + ", lr = " + str(best_lr)
                f.write(text)
                f.write('\n')
