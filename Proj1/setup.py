"""Handle data (downloading & loading) and make it available to other modules."""

import tarfile
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision.datasets.utils import download_url

from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# ------------------- SETUP

BATCH_SIZE = 100

# ------------------- UTILITY


def download():
    if Path('MNIST').exists():
        return
    url = 'https://www.di.ens.fr/~lelarge/MNIST.tar.gz'
    fname = 'MNIST.tar.gz'
    md5 = 'a370fe4b9e5907b19816e70b08af1bf1'
    download_url(url, '.', fname, md5)

    # Extract
    print(f'Extracting {fname} ...')
    with tarfile.open(fname, 'r:gz') as f:
        f.extractall()


def mnist_to_pairs(nb, input, target):
    input = torch.functional.F.avg_pool2d(input, kernel_size=2)
    a = torch.randperm(input.size(0))
    a = a[:2 * nb].view(nb, 2)
    input = torch.cat((input[a[:, 0]], input[a[:, 1]]), 1)
    classes = target[a]
    target = (classes[:, 0] <= classes[:, 1]).long()
    return input, target, classes


def generate_pair_sets(nb):
    train_set = datasets.MNIST('', train=True, download=False)
    train_input = train_set.data.view(-1, 1, 28, 28).float()
    train_target = train_set.targets

    test_set = datasets.MNIST('', train=False, download=False)
    test_input = test_set.data.view(-1, 1, 28, 28).float()
    test_target = test_set.targets

    return mnist_to_pairs(nb, train_input, train_target) + \
           mnist_to_pairs(nb, test_input, test_target)


# ------------------- DOWNLOAD & LOAD DATA

download()

train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

train_dataset = TensorDataset(train_input, train_target, train_classes)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

test_dataset = TensorDataset(test_input, test_target, test_classes)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
