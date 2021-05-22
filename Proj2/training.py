"""
Utility functions/classes used to train the network with stochastic gradient descent (SGD).
"""

from time import time

import torch
from torch import tensor, randperm

from function import Function
from module import Module
from tensor import Tensor


class Dataset:
    """
    Convenience class used to iterate over samples of a dataset during SGD.
    """
    def __init__(self, data, target=None, shuffle=False):
        """Data must be an n times d matrix, n = number of samples, d = number of features."""
        self.data = data
        self.target = target
        self.shuffle = shuffle

    def __iter__(self):
        """Generate samples of (data, target)."""
        n = self.data.shape[0]
        iterator = randperm(n) if self.shuffle else range(n)
        # Generate data only
        if self.target is None:
            for i in iterator:
                # Take i-th row and instantiate Tensor
                x = Tensor(self.data[i])
                yield x
        else:
            for i in iterator:
                x = Tensor(self.data[i])
                y = Tensor(self.target[i])

                yield x, y


def train_epoch_sgd(dataset: Dataset, model: Module, loss_fun: Function, lr: float):
    """Train the `model` over one epoch of the `dataset`."""
    losses = []
    t = time()

    for i, (x, y) in enumerate(dataset):
        # Forward pass
        model.zero_grad()
        output = model(x)
        loss = loss_fun(output, y)
        # Backward pass
        loss.backward()
        losses.append(loss.item())
        # Gradient step
        model.step(lr)

    weights_norm = {
        p._name: p.data.pow(2).sum() for p in model._params()
    }

    t = time() - t
    data = {
        'loss': losses,
        'time': t,
        'weight': weights_norm
    }
    return data


def train_SGD(dataset: Dataset, model: Module, loss_fun: Function, lr: float, epochs: int):
    """
    Train the `model` with the given `dataset`.
    Returns a dictionnary containing:
        - mean loss per epoch
        - computation time of each epoch
        - squared frobenius norm of model parameters at each epoch
    """
    losses = []
    times = []
    weights = []
    print('epochs: ', end='')
    for epoch in range(1, epochs+1):
        data = train_epoch_sgd(dataset, model, loss_fun, lr)
        losses.append(tensor(data['loss']).mean().item())
        times.append(data['time'])
        weights.append(data['weight'])
        print(epoch, end=' ')
    print()

    data = {
        'loss': torch.tensor(losses),
        'time': torch.tensor(times),
        'weight': weights
    }
    return data

