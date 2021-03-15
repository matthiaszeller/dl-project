

import torch.nn.functional as F
from torch import optim

from loss import custom_loss
from nets import FullyDenseNet, FullyDenseNetAux
from stats import train_multiple_runs, plot_std_loss_acc
from training import train

import matplotlib.pyplot as plt


def main():
    net = FullyDenseNet()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    tot_train_loss, tot_train_acc, tot_test_loss, tot_test_acc = train(net, optimizer, F.binary_cross_entropy)

    net = FullyDenseNetAux()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    tot_train_loss, tot_train_acc, tot_test_loss, tot_test_acc = train(net, optimizer, criterion_=custom_loss)

    all_train_loss, all_train_acc, all_test_loss, all_test_acc = train_multiple_runs(FullyDenseNetAux,
                                                                                     criterion=custom_loss,
                                                                                     epoch=25,
                                                                                     runs=15)
    plot_std_loss_acc(all_train_loss, all_train_acc, all_test_loss, all_test_acc)


if __name__ == '__main__':
    main()


