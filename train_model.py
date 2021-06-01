import torch
from torch import nn, optim

from utils import device, trainloader, testloader
from dla_simple import DLA

net = DLA()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.06,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

labels_train = []
labels_test = []

#best_acc = 0

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f' accuracy in TRAIN is {100.*correct/total} and loss is {train_loss/(total+1)}')


def test(epoch):
    best_acc = 0
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f' accuracy in TEST is {100.*correct/total} and loss is {test_loss/(batch_idx+1)}')


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        print(f'Acc is {acc} and best acc so far was {best_acc}')



# we had 100 epochs
# for epoch in range(60):
#     train(epoch)
#     test(epoch)
#     scheduler.step()

net.load_state_dict(torch.load("model.pt"))
test(1)


# first_conv = PartConv().to(device)
# load_weights(first_conv)
#
#
#
# def test():
#     best_acc = 0
#     first_conv.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(test_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#
#             if batch_idx % 100 == 0:
#               print(f'we are at {batch_idx}')
#             R = inputs[:, 0, :, :]
#             G = inputs[:, 1, :, :]
#             B = inputs[:, 2, :, :]
#             R = R.reshape(1,1,32,32).to(device)
#             G = G.reshape(1,1,32,32).to(device)
#             B = B.reshape(1,1,32,32).to(device)
#             with torch.no_grad():
#               conv.base.weight.data = weight_R
#
#             x_enc, windows_nb = ts.im2col_encoding(
#               context, R.view(32, 32).tolist(), 1,
#               1, 1
#             )
#             first_part = EncServer2(conv).to(device)
#             output_R = first_part(x_enc, windows_nb)
#             output_R = output_R.decrypt()
#             output_R = np.reshape(output_R, (1,16,32,32))
#             output_R = torch.from_numpy(output_R).float()
#
#
#             with torch.no_grad():
#               conv.base.weight.data = weight_G
#
#             x_enc, windows_nb = ts.im2col_encoding(
#               context, G.view(32, 32).tolist(), 1,
#               1, 1
#             )
#             first_part = EncServer2(conv).to(device)
#             output_G = first_part(x_enc, windows_nb)
#             output_G = output_G.decrypt()
#             output_G = np.reshape(output_G, (1,16,32,32))
#             output_G = torch.from_numpy(output_G).float()
#
#
#             with torch.no_grad():
#               conv.base.weight.data = weight_B
#
#             x_enc, windows_nb = ts.im2col_encoding(
#               context, B.view(32, 32).tolist(), 1,
#               1, 1
#             )
#             first_part = EncServer2(conv).to(device)
#             output_B = first_part(x_enc, windows_nb)
#             output_B = output_B.decrypt()
#             output_B = np.reshape(output_B, (1,16,32,32))
#             output_B = torch.from_numpy(output_B).float()
#
#
#             output_final = output_R + output_G + output_B
#             output_final = output_final.to(device)
#
#
#             outputs = first_conv(output_final)
#             loss = criterion(outputs, targets)
#
#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#
#         print(f' accuracy in TEST is {100.*correct/total} and loss is {test_loss/(batch_idx+1)}')