"""
Utility functions used to train & test models.
"""

# --------------------------------------------------------- #
#                          IMPORTS                          #
# --------------------------------------------------------- #

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import generate_pair_sets

# --------------------------------------------------------- #
#                           DATA                            #
# --------------------------------------------------------- #

train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

BATCH_SIZE = 20
train_dataloader = None
test_dataloader = None


# -------------------------------------------------------- #
#                      TRAINING UTILS                      #
# -------------------------------------------------------- #

def initialize_dataset():
    # TRAIN SET

    train_dataset = TensorDataset(train_input, train_target, train_classes)
    # TEST SET
    test_dataset = TensorDataset(test_input, test_target, test_classes)

    global train_dataloader, test_dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print('>> Dataset correctly loaded')
    print(f'batch size : ', BATCH_SIZE)
    print('\n\n')


def handle_loss(criterion_):
    """
    Handle the fact that the network with auxiliary loss has three-item tuple output,
    which needs to be treated separately to compute the loss and the accuracy.
    """
    if criterion_ is F.binary_cross_entropy:
        internal_criterion = lambda output, target, _: criterion_(output.flatten(), target)
        compute_acc = lambda output, target: (target == torch.round(output.flatten())).float()
    else:
        internal_criterion = criterion_
        compute_acc = lambda output, target: (target == torch.round(output[0].flatten())).float()

    return internal_criterion, compute_acc


def train_epoch(network_, optimizer_, criterion_=F.binary_cross_entropy):
    """
    Trains the model for one epoch and returns the loss and accuracy using the specified criterion
    """
    internal_criterion, compute_acc = handle_loss(criterion_)

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

    return torch.FloatTensor(loss_tot).mean().item(), torch.cat(acc_tot).mean().item()


def test(network_, criterion_=F.binary_cross_entropy):
    internal_criterion, compute_acc = handle_loss(criterion_)

    network_.eval()
    test_loss = []
    acc = []

    with torch.no_grad():
        for data, target, classes in test_dataloader:
            output = network_(data)
            loss = internal_criterion(output, target.to(torch.float32), classes)
            test_loss.append(loss.item())
            acc.append(compute_acc(output, target))

    return torch.FloatTensor(test_loss).mean().item(), torch.cat(acc).mean().item()


def train(network_, optimizer_, criterion_=F.binary_cross_entropy, epoch_nb=30, debug_=True):
    tot_train_loss = []
    tot_train_acc = []
    tot_test_loss = []
    tot_test_acc = []

    if debug_:
        print("%7s | %11s | %11s | %11s | %11s" % ("Epoch", "Train Loss", "Train Acc", "Test Loss", "Test Acc"))

    for epoch in range(epoch_nb):

        train_loss, train_acc = train_epoch(network_, optimizer_, criterion_)
        test_loss, test_acc = test(network_, criterion_)

        tot_train_loss.append(train_loss)
        tot_train_acc.append(train_acc)
        tot_test_loss.append(test_loss)
        tot_test_acc.append(test_acc)

        if debug_:
            print("%7.f | %11.2f | %11.4f | %11.4f | %11.4f" % (epoch, train_loss, train_acc, test_loss, test_acc))

    return tot_train_loss, tot_train_acc, tot_test_loss, tot_test_acc


# --------------------------------------------------------- #
#                        CUSTOM LOSS                        #
# --------------------------------------------------------- #

def custom_loss_BCELoss_CELoss(output, target, classes):
    """
    Binary cross encropy and cross entroy loss.
    Custom loss for network with auxiliary losses. The total loss is a combination
    of the loss of the main task (binary cross entropy) and the negative log likelihood
    for the two auxiliary tasks. Importance of auxiliary losses is controlled by
    the `lambda_` hyperparameter.
    """
    main, im1, im2 = output

    criterion_im = nn.CrossEntropyLoss()
    criterion = F.binary_cross_entropy

    aux_loss_1 = criterion_im(im1, classes[:, 0].long())
    aux_loss_2 = criterion_im(im2, classes[:, 1].long())
    main_loss = criterion(main.flatten(), target)

    return main_loss + aux_loss_1 + aux_loss_2
