

import matplotlib.pyplot as plt
import numpy as np
import torch

from function import MSELoss
from module import Sequential, LinearLayer
from training import Dataset, train_SGD

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

X = np.vstack((h, g)).T

# Data normalization
mu, std = X.mean(0), X.std(0)
X = (X - mu) / std

dataset = Dataset(X, w)

s = Sequential(
    LinearLayer(2, 1)
)

# Smoothness constant and strongly-convex constant
eigs = np.linalg.eigvals(2 * X.T @ X)
L = max(eigs)
mu_sc = min(eigs)

data = train_SGD(dataset, s, MSELoss(), lr=1/L, epochs=10)
losses = data['loss']

# Manual sanity check
# For gender = 1
Xtest = np.ones((100, 2))
Xtest[:, 0] = hgrid
Xtest = (Xtest - mu) / std
testset = Dataset(Xtest, w)

y = np.array([
    s(x).item() for x, _ in testset
])

plt.plot(hgrid, y, label='1 layer, gender=1')


# --- More complicated network
s = Sequential(
    LinearLayer(2, 2),
    LinearLayer(2, 1)
)

dataset = Dataset(X, w)
data = train_SGD(dataset, s, MSELoss(), lr=1/L, epochs=10)
y = np.array([
    s(x).item() for x, _ in testset
])

plt.plot(hgrid, y, label='2 linear layers, no non-linearity')

plt.legend()
plt.show()

