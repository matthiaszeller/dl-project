

from time import time

import numpy as np
import torch

from autograd import Tensor, Module, Function


class Dataset:
    """Used to iterate over samples of a dataset during SGD."""
    def __init__(self, data, target, shuffle=False):
        """data must be an n times d matrix, n = number of samples, d = number of features."""
        self.data = data
        self.target = target
        self.shuffle = shuffle

    def __iter__(self):
        n = self.data.shape[0]
        iterator = np.random.permutation(n) if self.shuffle else range(n)
        for i in iterator:
            # Take i-th row and instantiate Tensor
            x = Tensor(self.data[i])
            y = Tensor(self.target[i])

            yield x, y


def train_epoch_sgd(dataset: Dataset, model: Module, loss_fun: Function, lr: float, lambda_: float):
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


def train_sgd(dataset: Dataset, model: Module, loss_fun: Function, lr: float, epochs: int, lambda_: float = 0):
    losses = []
    times = []
    weights = []
    print('epochs: ', end='')
    for epoch in range(1, epochs+1):
        data = train_epoch_sgd(dataset, model, loss_fun, lr, lambda_)
        losses.append(torch.tensor(data['loss']).mean().item())
        times.append(data['time'])
        weights.append(data['weight'])
        print(epoch, end=' ')

    data = {
        'loss': losses,
        'time': times,
        'weight': weights
    }
    return data

