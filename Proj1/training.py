import torch
import torch.nn.functional as F

from loss import handle_loss
from setup import train_dataloader, test_dataloader


def train_epoch(network_, optimizer_, criterion=F.binary_cross_entropy):
    internal_criterion, compute_acc = handle_loss(criterion)

    loss_tot = []
    acc_tot = []
    network_.train()
    for batch_idx, (data, target, classes) in enumerate(train_dataloader):
        optimizer_.zero_grad()
        output = network_(data)
        loss = internal_criterion(output, target.to(torch.float32), classes)
        loss.backward()
        optimizer_.step()

        loss_tot.append(loss.item())
        acc_tot.append(compute_acc(output, target))

    return torch.FloatTensor(loss_tot).mean().item(), torch.FloatTensor(acc_tot).mean().item() / 100.0


def test(network_, criterion_=F.binary_cross_entropy):
    internal_criterion, compute_acc = handle_loss(criterion_)

    network_.eval()
    test_loss = 0
    acc = 0

    with torch.no_grad():
        for data, target, classes in test_dataloader:
            output = network_(data)
            test_loss += internal_criterion(output, target.to(torch.float32), classes)
            acc += compute_acc(output, target)

    test_loss /= len(test_dataloader)
    acc /= len(test_dataloader)
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
            print(epoch, f'{train_loss:.4}\t{train_acc:.4}\t{test_loss:.4}\t{test_acc:.4}')

    return tot_train_loss, tot_train_acc, tot_test_loss, tot_test_acc
