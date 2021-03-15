import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from training import train


def train_multiple_runs(network_class, criterion=F.binary_cross_entropy, runs=10, epoch=30):
    all_train_loss, all_train_acc, all_test_loss, all_test_acc = [], [], [], []

    for i in range(runs):
        n = network_class()
        optimizer = optim.SGD(n.parameters(), lr=0.01, momentum=0.5)

        tot_train_loss, tot_train_acc, tot_test_loss, tot_test_acc = train(n, optimizer, criterion, epoch, debug_=False)
        all_train_loss.append(tot_train_loss)
        all_train_acc.append(tot_train_acc)
        all_test_loss.append(tot_test_loss)
        all_test_acc.append(tot_test_acc)

        if i % 2 == 0:
            print(i, end=' ')

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
    def subplot(ax, mean, std, label=None):
        x = range(1, len(mean) + 1)
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.3)

    # Convert lists of lists to tensors
    data = {
        'loss_train': all_train_loss,
        'loss_test': all_test_loss,
        'acc_train': all_train_acc,
        'acc_test': all_test_acc
    }

    data = {k: torch.tensor(v) for k, v in data.items()}

    # Compute mean and std. The multiple runs are in axis 0, epochs in axis 1
    plot_data = {}
    for label, tensor in data.items():
        plot_data[f'{label}_mean'] = tensor.mean(dim=0)
        plot_data[f'{label}_std'] = tensor.std(dim=0)

    # Plot!
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
    subplot(ax[0], plot_data['loss_train_mean'], plot_data['loss_train_std'], label='train')
    subplot(ax[0], plot_data['loss_test_mean'], plot_data['loss_test_std'], label='test')
    ax[0].set_ylabel('Loss')
    subplot(ax[1], plot_data['acc_train_mean'], plot_data['acc_train_std'])
    subplot(ax[1], plot_data['acc_test_mean'], plot_data['acc_test_std'])
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')

    ax[0].legend()
    plt.show()

