

import numpy as np
import matplotlib.pyplot as plt
import torch
from variable import Sequential, LinearLayer, ReLU, Tensor, Module, Function, MSELoss

# Disable globally autograd
torch.set_grad_enabled(False)


def load_data(sub_sample=True):
    """Load data and convert it to the metric system."""
    path_dataset = "height_weight_genders.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454

    # sub-sample
    if sub_sample:
        height = height[::50]
        weight = weight[::50]
        gender = gender[::50]

    return height, weight, gender


# Predict weight in function of height and gender
h, w, g = load_data(sub_sample=True)
male = g == 1
plt.plot(h[male], w[male], 'o')
plt.plot(h[~male], w[~male], 'o')

# Data matrix, d x n
offset = np.ones_like(h)
X = np.vstack((offset, h, g)).T

# Closed form solution for the MSE loss
wstar = np.linalg.solve(X.T @ X, X.T @ w)
loss = np.power(X @ wstar - w, 2).sum()
print(f'MSE of TRAINING loss for closed form solution: {loss:.4}')

# Visualize solution
hgrid = np.linspace(h.min(), h.max(), 100)

# For gender = 1
Xtest = np.ones((100, 3))
Xtest[:, 1] = hgrid
y = Xtest @ wstar
plt.plot(hgrid, y, label='regline gender=1')

# For gender = 0
Xtest[:, 2] = 0.0
y = Xtest @ wstar
plt.plot(hgrid, y, label='regline gender=0')



class Dataset:

    def __init__(self, data, target):
        """data must be an n times d matrix, n = number of samples, d = number of features."""
        self.data = data
        self.target = target

    def __iter__(self):
        for i in range(self.data.shape[0]):
            # Take i-th row and instantiate Tensor
            x = Tensor(self.data[i])
            y = Tensor(self.target[i])

            yield x, y


def train_epoch_sgd(dataset: Dataset, model: Module, loss_fun: Function, lr: float):
    losses = []
    for i, (x, y) in enumerate(dataset):
        model.zero_grad()
        # Forward pass
        output = model(x)
        loss = loss_fun(output, y)
        # Backward pass
        loss.backward()
        # Gradient step
        model.step(lr)

        losses.append(loss.item())

    return losses


def train_sgd(dataset: Dataset, model: Module, loss_fun: Function, lr: float, epochs: int):
    losses = []
    for epoch in range(1, epochs+1):
        loss_epoch = train_epoch_sgd(dataset, model, loss_fun, lr)
        losses.append(torch.tensor(loss_epoch).mean().item())

    return losses


X = np.vstack((h, g)).T
dataset = Dataset(X, w)

s = Sequential(
    LinearLayer(2, 2),
    #ReLU(),
    LinearLayer(2, 1)
)

# Smoothness constant
L = np.linalg.eigvals(2 * X.T @ X).max()
losses = train_sgd(dataset, s, MSELoss(), lr=0.01, epochs=10)

# Manual sanity check
Xtest = Xtest[:, 1:]
testset = Dataset(Xtest, w)

y = np.array([
    s(x).item() for x, _ in testset
])

plt.plot(hgrid, y, label='network, gender=0')
plt.legend()
plt.show()

