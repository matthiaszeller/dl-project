

import torch.nn.functional as F
from torch import optim

from nets import FullyDenseNet
from training import train


def main():
    net = FullyDenseNet()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    tot_train_loss, tot_train_acc, tot_test_loss, tot_test_acc = train(net, optimizer, F.binary_cross_entropy)


if __name__ == '__main__':
    main()


