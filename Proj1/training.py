

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def train_epoch(data, network_, optimizer_, criterion=F.binary_cross_entropy):
    loss_tot = []
    acc_tot = []
    network_.train()
    for batch_idx, (data, target, classes) in enumerate(data):
        optimizer_.zero_grad()
        # Why flatten?
        output = network_(data).flatten()
        # TODO : unefficient to cast to float in the loop
        loss = criterion(output, target.to(torch.float32))
        loss.backward()
        optimizer_.step()

        loss_tot.append(loss.item())
        acc_tot.append((target == torch.round(output)).sum().item())

    return torch.FloatTensor(loss_tot).mean().item(), torch.FloatTensor(acc_tot).mean().item() / 100.0


def test(data, network_, criterion_=F.binary_cross_entropy):
    network_.eval()
    test_loss = 0
    acc = 0

    with torch.no_grad():
        for data, target, classes in data:
            output = network_(data)
            test_loss += criterion_(output.flatten(), target.to(torch.float32)).item()
            acc += (target == torch.round(output.flatten())).sum().item()

    test_loss /= len(data)
    acc /= len(data)
    return test_loss, acc / 100.0


def train(network_, optimizer_, criterion_=F.binary_cross_entropy, epoch_nb=30, debug_=True):
    tot_train_loss = []
    tot_train_acc = []
    tot_test_loss = []
    tot_test_acc = []

    for epoch in range(epoch_nb):
        train_loss, train_acc = train_epoch(network_, optimizer_, criterion_)
        test_loss, test_acc = test(network_, criterion_)

        tot_train_loss.append(train_loss)
        tot_train_acc.append(train_acc)
        tot_test_loss.append(test_loss)
        tot_test_acc.append(test_acc)

        if debug_:
            print(epoch, train_loss, train_acc, test_loss, test_acc)

    return tot_train_loss, tot_train_acc, tot_test_loss, tot_test_acc
