"""
Test file to generate a dataset and train a simple network with three hidden layers.

torch.autograd is disabled with
    >>> torch.set_grad_enabled(False)
"""

# --------------------------------------------------------- #
#                          IMPORTS                          #
# --------------------------------------------------------- #

# --- Standard modules
from math import pi
import torch
from matplotlib import pyplot as plt

# --- Our custom modules
import function as F
from module import LinearLayer, Sequential
from training import Dataset, train_SGD

# --- Disable globally autograd
torch.set_grad_enabled(False)


# --------------------------------------------------------- #
#                           UTILS                           #
# --------------------------------------------------------- #

def pretty_print_section(section: str) -> None:
    """Pretty print a header in order to make the script output look nice"""
    print(f'\n# ------ {section.upper()}\n')


# -------------------------------------------------------- #
#                    DATASET GENERATION                    #
# -------------------------------------------------------- #

# --- Generate data
# 1000 2D points uniformly distributed in [0,1]^2
n = 1000
d = 2
X = torch.rand((n, d))

# --- Generate target
# Compute distance of each point from (0.5, 0.5)
center = torch.tensor([0.5, 0.5])
squared_dist = ((X - center)**2).sum(axis=1)
radius_squared = 1 / 2 / pi
target = (squared_dist < radius_squared) * 1

# --- Plot dataset
plt.plot(X[target == 1, 0], X[target == 1, 1], 'o', label='target 1')
plt.plot(X[target == 0, 0], X[target == 0, 1], 'o', label='target 0')
plt.legend()
plt.show()

# --- Data splitting - Training & test sets
train_ratio = 0.8
split_point = int(train_ratio * n)
xtrain, xtest = X[:split_point], X[split_point:]
ytrain, ytest = target[:split_point], target[split_point:]

# --- Data normalization
mu, std = xtrain.mean(0), xtrain.std(0)
xtrain = (xtrain - mu) / std
xtest  = (xtest  - mu) / std


# -------------------------------------------------------- #
#                         TRAINING                         #
# -------------------------------------------------------- #

# --- Create models
# 3 hidden layers of 25 neurons each
model_relu = Sequential(
    LinearLayer(2, 25),
    F.ReLU(),
    LinearLayer(25, 25),
    F.ReLU(),
    LinearLayer(25, 25),
    F.ReLU(),
    LinearLayer(25, 1)
)

model_tanh = Sequential(
    LinearLayer(2, 25),
    F.ReLU(),
    LinearLayer(25, 25),
    F.ReLU(),
    LinearLayer(25, 25),
    F.ReLU(),
    LinearLayer(25, 1)
)

pretty_print_section('Models')
print(f'Model 1:\t{model_relu}')
print(f'Model 2:\t{model_tanh}')

# --- Train with MSE loss & SGD
# Initialize training
mse = F.MSELoss()
train_dataset = Dataset(xtrain, ytrain)
lr, epochs = 0.13, 10

# ReLU model
print('Training model 1 ...')
relu_log = train_SGD(
    train_dataset,
    model_relu,
    mse,
    lr=lr,
    epochs=epochs
)
print(f'Training time: {relu_log["time"].sum().item():.2} s')

# Tanh model
print('Training model 2 ...')
tanh_log = train_SGD(
    train_dataset,
    model_tanh,
    mse,
    lr=lr,
    epochs=epochs
)
print(f'Training time: {tanh_log["time"].sum().item():.2} s')

# --- Plot loss
losses, times, weights_norm = relu_log['loss'], relu_log['time'], relu_log['weight']
plt.plot(relu_log['loss'], '-o', label='relu activation funs')
plt.plot(tanh_log['loss'], '-o', label='tanh activation funs')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()


# --------------------------------------------------------- #
#                          TESTING                          #
# --------------------------------------------------------- #

pretty_print_section('performance evaluation')
print('Computing accuracy as: i) round output to nearest integer ii) clip to (0, 1)')


def acc(y, yhat):
    return (y.round().clip(0, 1) == yhat).to(float).mean().item()


# --- Training accuracy
# Predict
yhat_relu = torch.tensor([model_relu(x).item() for x, _ in train_dataset])
yhat_tanh = torch.tensor([model_tanh(x).item() for x, _ in train_dataset])

# Compute accuracy
train_acc_relu = acc(yhat_relu, ytrain)
train_acc_tanh = acc(yhat_tanh, ytrain)

# --- Test accuracy
test_dataset = Dataset(xtest, ytest)
# Predict
yhat_relu = torch.tensor([model_relu(x).item() for x, _ in test_dataset])
yhat_tanh = torch.tensor([model_tanh(x).item() for x, _ in test_dataset])
# Compute accuracy
test_acc_relu = acc(yhat_relu, ytest)
test_acc_tanh = acc(yhat_tanh, ytest)

# --- Print accuracy
print('\t\t\tTrain accuracy\t\tTest accuracy')
print(f'Relu model\t{train_acc_relu:.3}\t\t\t\t{test_acc_relu:.3}')
print(f'Tanh model\t{train_acc_tanh:.3}\t\t\t\t{test_acc_tanh:.3}')
