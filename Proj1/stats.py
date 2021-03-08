import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import optim

from training import train


def train_multiple_runs(network_class, runs=10, epoch=30):
    all_train_loss, all_train_acc, all_test_loss, all_test_acc = [], [], [], []

    for i in range(runs):
        n = network_class()
        optimizer = optim.SGD(n.parameters(), lr=0.01, momentum=0.5)
        criterion = F.binary_cross_entropy

        tot_train_loss, tot_train_acc, tot_test_loss, tot_test_acc = train(n, optimizer, criterion, epoch, debug_=False)
        all_train_loss.append(tot_train_loss)
        all_train_acc.append(tot_train_acc)
        all_test_loss.append(tot_test_loss)
        all_test_acc.append(tot_test_acc)

    return all_train_loss, all_train_acc, all_test_loss, all_test_acc


def plot_loss_acc(tot_train_loss, tot_train_acc, tot_test_loss, tot_test_acc):
    epochs = range(1, len(tot_train_loss) + 1)
    plt.plot(epochs, tot_train_loss, 'g', label='Training loss')
    plt.plot(epochs, tot_test_loss, 'b', label='Test loss')
    plt.plot(epochs, tot_train_acc, 'r', label='Training acc')
    plt.plot(epochs, tot_test_acc, 'y', label='Test acc')
    plt.title('Training and Test loss/acc')
    plt.xlabel('Epochs')
    plt.ylabel('loss/acc')
    plt.ylim((0, 1))
    plt.legend()
    plt.show()


def plot_std_loss_acc(all_train_loss, all_train_acc, all_test_loss, all_test_acc):
    trl_mean = np.array(all_train_loss).mean(axis=0)
    tel_mean = np.array(all_train_acc).mean(axis=0)
    tra_mean = np.array(all_test_loss).mean(axis=0)
    tea_mean = np.array(all_test_acc).mean(axis=0)

    trl_std = np.array(all_train_loss).std(axis=0)
    tel_std = np.array(all_train_acc).std(axis=0)
    tra_std = np.array(all_test_loss).std(axis=0)
    tea_std = np.array(all_test_acc).std(axis=0)

    epochs = range(1, len(tea_std) + 1)

    temp = [[trl_mean, trl_std], [tel_mean, tel_std], [tra_mean, tra_std], [tea_mean, tea_std]]

    for g in temp:
        plt.plot(epochs, g[0])
        plt.fill_between(epochs, g[0] - g[1], g[0] + g[1], alpha=0.3)

    plt.ylim((-0.1, 1.1))

