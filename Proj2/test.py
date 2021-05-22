"""
Test file to generate a dataset and train a simple network with three hidden layers.

torch.autograd is disabled with `torch.set_grad_enabled(False)`
"""

# --------------------------------------------------------- #
#                          IMPORTS                          #
# --------------------------------------------------------- #

# --- Standard modules
from math import pi, sqrt
from typing import Union, List

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


def acc(yhat, y):
    """Computes accuracy"""
    return (yhat.round().clip(0, 1) == y).to(float).mean().item()


def build_model(activation_fun: F.Function, xavier_init: bool = True):
    """Build a deep neural network with a fixed architecture:
    2 input neurons, 3 hidden layers of 25 neurons each, 1 output unit"""
    return Sequential(
        LinearLayer(2, 25, xavier_init=xavier_init),
        activation_fun(),
        LinearLayer(25, 25, xavier_init=xavier_init),
        activation_fun(),
        LinearLayer(25, 25, xavier_init=xavier_init),
        activation_fun(),
        LinearLayer(25, 1, xavier_init=xavier_init)
    )


def train_and_test(model: Sequential, train_set: Dataset, test_set: Dataset,
                   loss_fun: F.Function, lr: Union[float, List[float]], epochs: int):
    # Train
    log = train_SGD(
        train_set,
        model,
        loss_fun,
        lr=lr,
        epochs=epochs
    )
    # Predict
    yhat_train, ytrain = torch.tensor([[model(x).item(), y.item()] for x, y in train_set]).T
    yhat_test, ytest = torch.tensor([[model(x).item(), y.item()] for x, y in test_set]).T
    # Performance evaluation
    acc_train = acc(yhat_train, ytrain)
    acc_test = acc(yhat_test, ytest)

    return log, acc_train, acc_test


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
# Decision boundary (for plotting)
angles = torch.linspace(0., 2. * pi, 200)
decision_bdry_true = torch.vstack([angles.cos(), angles.sin()]).T * sqrt(radius_squared) + center

# --- Data splitting - Training & test sets
train_ratio = 0.8
split_point = int(train_ratio * n)
xtrain, xtest = X[:split_point], X[split_point:]
ytrain, ytest = target[:split_point], target[split_point:]


plt.figure(figsize=(5, 5))
# --- Plot dataset
plt.scatter(xtrain[ytrain == 1, 0], xtrain[ytrain == 1, 1], marker='o', c='m', label='target 1, training')
plt.scatter(xtrain[ytrain == 0, 0], xtrain[ytrain == 0, 1], marker='o', c='b', label='target 0, training')
plt.scatter(xtest[ytest == 1, 0], xtest[ytest == 1, 1], marker='x', c='m', label='target 1, test')
plt.scatter(xtest[ytest == 0, 0], xtest[ytest == 0, 1], marker='x', c='b', label='target 0, test')

# --- Plot decision boundaries
plt.plot(decision_bdry_true[:, 0], decision_bdry_true[:, 1], 'k--', label='true decision boundary')

# --- Labeling
plt.xlabel('x1')
plt.ylabel('x2')
legend = plt.legend(bbox_to_anchor=(1.05, 1))

plt.savefig('fig_dataset.pdf', bbox_inches='tight')
plt.show()

# --- Data normalization
mu, std = xtrain.mean(0), xtrain.std(0)
xtrain = (xtrain - mu) / std
xtest  = (xtest  - mu) / std


# -------------------------------------------------------- #
#                         TRAINING                         #
# -------------------------------------------------------- #

# --- Build models
models = [
    {'activation_fun': F.ReLU, 'xavier_init': True},
    # TODO DOESNT WORK {'activation_fun': F.Tanh, 'xavier_init': False, 'lr': 0.1},
    {'activation_fun': F.Sigmoid, 'xavier_init': False}
]

models = [build_model(**kwargs) for kwargs in models]

pretty_print_section('models')
for m in models:
    print(m)


# --- Train models
train_dataset = Dataset(xtrain, ytrain)
test_dataset = Dataset(xtest, ytest)

pretty_print_section('training')
for model in models:
    log, train_acc, test_acc = train_and_test(model, train_dataset, test_dataset, F.MSELoss(), lr=0.15, epochs=10)
    plt.plot(log['loss'])

plt.show()

# --- Performance evaluation
pretty_print_section('performance evaluation')
print('\t\t\tTrain accuracy\t\tTest accuracy')
for r in results:
    _, acc_train, acc_test = r
    print(f'  model   \t{acc_train:.3}\t\t\t\t{acc_test:.3}')

exit()

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
    LinearLayer(2, 25, xavier_init=False),
    F.Sigmoid(),
    LinearLayer(25, 25, xavier_init=False),
    F.Sigmoid(),
    LinearLayer(25, 25, xavier_init=False),
    F.Sigmoid(),
    LinearLayer(25, 1, xavier_init=False)
)

pretty_print_section('Models')
print(f'Model 1:\t{model_relu}')
print(f'Model 2:\t{model_tanh}')

# --- Train with MSE loss & SGD
# Initialize training
mse = F.MSELoss()
train_dataset = Dataset(xtrain, ytrain)
lr, epochs = 0.15, 10

# ReLU model
print('Training model 1 ...')
relu_log = train_SGD(
    train_dataset,
    model_relu,
    mse,
    lr=0.15,
    epochs=8
)
print(f'Training time: {relu_log["time"].sum().item():.3} s')

# Tanh model
print('Training model 2 ...')
tanh_log = train_SGD(
    train_dataset,
    model_tanh,
    mse,
    lr=0.15,
    epochs=35
)
print(f'Training time: {tanh_log["time"].sum().item():.3} s')

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



# --- Training accuracy
# Predict
yhat_relu_train = torch.tensor([model_relu(x).item() for x, _ in train_dataset])
yhat_tanh_train = torch.tensor([model_tanh(x).item() for x, _ in train_dataset])

# Compute accuracy
train_acc_relu = acc(yhat_relu_train, ytrain)
train_acc_tanh = acc(yhat_tanh_train, ytrain)

# --- Test accuracy
test_dataset = Dataset(xtest, ytest)
# Predict
yhat_relu_test = torch.tensor([model_relu(x).item() for x, _ in test_dataset])
yhat_tanh_test = torch.tensor([model_tanh(x).item() for x, _ in test_dataset])
# Compute accuracy
test_acc_relu = acc(yhat_relu_test, ytest)
test_acc_tanh = acc(yhat_tanh_test, ytest)

# --- Print accuracy
print('\t\t\tTrain accuracy\t\tTest accuracy')
print(f'Relu model\t{train_acc_relu:.3}\t\t\t\t{test_acc_relu:.3}')
print(f'Tanh model\t{train_acc_tanh:.3}\t\t\t\t{test_acc_tanh:.3}')


# TODO make this work
# # Generate grid
# x = torch.linspace(0., 1., 10)
# xx, yy = torch.meshgrid(x, x)
# grid = torch.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
# yhat_grid2 = torch.tensor([model_relu(x).item() for x in Dataset(grid)])
# yhat_grid = yhat_grid2.round().clip(0, 1).reshape(xx.shape)
#
# plt.contourf(xx, yy, yhat_grid, cmap='Paired')
# plt.show()
