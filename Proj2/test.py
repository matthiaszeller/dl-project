
# --------------------------------------------------------- #
#                          IMPORTS                          #
# --------------------------------------------------------- #

from math import pi

import torch
from matplotlib import pyplot as plt

import function as F
from module import LinearLayer, Sequential
from training import Dataset, train_SGD

# Disable globally autograd
torch.set_grad_enabled(False)


# -------------------------------------------------------- #
#                    DATASET GENERATION                    #
# -------------------------------------------------------- #

# Generate data: 1000 2D points uniformly distributed in [0,1]^2
n = 1000
d = 2
X = torch.rand((n, d))

# Generate target
# Compute distance of each point from (0.5, 0.5)
center = torch.tensor([0.5, 0.5])
squared_dist = ((X - center)**2).sum(axis=1)
radius_squared = 1 / 2 / pi
target = (squared_dist < radius_squared) * 1

# Plot dataset
plt.plot(X[target == 1, 0], X[target == 1, 1], 'o', label='target 1')
plt.plot(X[target == 0, 0], X[target == 0, 1], 'o', label='target 0')
plt.legend()
plt.show()

# Data splitting and data normalization
train_ratio = 0.8
split_point = int(train_ratio * n)
xtrain, xtest = X[:split_point], X[split_point:]
ytrain, ytest = target[:split_point], target[split_point:]

mu, std = xtrain.mean(0), xtrain.std(0)
xtrain = (xtrain - mu) / std
xtest  = (xtest  - mu) / std

# -------------------------------------------------------- #
#                         TRAINING                         #
# -------------------------------------------------------- #

# 3 hidden layers of
model = Sequential(
    LinearLayer(2, 25),
    F.ReLU(),
    LinearLayer(25, 25),
    F.ReLU(),
    LinearLayer(25, 25),
    F.ReLU(),
    LinearLayer(25, 1)
)
print(f'Model: {model}')

mse = F.MSELoss()
train_dataset = Dataset(xtrain, ytrain)
print('Launching training...')
train_log = train_SGD(
    train_dataset,
    model,
    mse,
    lr=0.1,
    epochs=30
)
print('Training done')
losses, times, weights_norm = train_log['loss'], train_log['time'], train_log['weight']

plt.plot(losses, '-o')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()


# --------------------------------------------------------- #
#                          TESTING                          #
# --------------------------------------------------------- #

print('Computing accuracy as: i) round output to nearest integer ii) clip to (0, 1)')


def acc(y, yhat):
    return (y.round().clip(0, 1) == yhat).to(float).mean().item()


# Training accuracy
y = torch.tensor([
    model(x).item() for x, _ in train_dataset
])

train_acc = acc(y, ytrain)


# Test accuracy
test_dataset = Dataset(xtest, ytest)
y = torch.tensor([
    model(x).item() for x, _ in test_dataset
])

test_acc = acc(y, ytest)

print(f'Train accuracy:\t{train_acc}\nTest accuracy:\t{test_acc}')

