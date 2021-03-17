

import argparse

import torch.nn.functional as F
from torch import optim

from loss import custom_loss
from nets import FullyDenseNet, FullyDenseNetAux
from stats import train_multiple_runs, plot_std_loss_acc
from training import train


def train_network(net_class, loss=F.binary_cross_entropy, lr=0.01):
    net = net_class()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.5)
    print_section(f'Training {net}')
    tot_train_loss, tot_train_acc, tot_test_loss, tot_test_acc = train(net, optimizer, loss)


def main():
    # --- User input
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stats', action='store_true', help='run statistics (multiple runs)')
    parser.add_argument('-D', '--debug', action='store_true', help='activate debugging')
    args = parser.parse_args()

    # --- Train networks
    train_network(FullyDenseNet)

    train_network(FullyDenseNetAux, custom_loss)

    # --- Run statistics
    if args.stats:
        print_section('Running statistics')
        stats = train_multiple_runs(FullyDenseNetAux,
                                    criterion=custom_loss,
                                    epoch=25,
                                    runs=15)
        all_train_loss, all_train_acc, all_test_loss, all_test_acc = stats
        plot_std_loss_acc(all_train_loss, all_train_acc, all_test_loss, all_test_acc)


def print_section(section_name):
    """Pretty-print a header/section/title on terminal"""
    print(f'\n------ {section_name}\n')


if __name__ == '__main__':
    main()
