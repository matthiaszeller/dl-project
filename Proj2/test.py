"""
Test file to generate a dataset and train a simple network with three hidden layers.

torch.autograd is disabled with `torch.set_grad_enabled(False)`
"""

# --------------------------------------------------------- #
#                          IMPORTS                          #
# --------------------------------------------------------- #

# --- Standard modules
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from math import pi, sqrt
from typing import Union, List

import torch
from matplotlib import pyplot as plt

# --- Our custom modules
import function as F
from module import LinearLayer, Sequential
from training import Dataset, train_SGD, kfold_cv

# --- Disable globally autograd
torch.set_grad_enabled(False)


parser = argparse.ArgumentParser()
parser.add_argument('--cv', help='cross validate learning rate', action='store_true')
args = parser.parse_args()


# --------------------------------------------------------- #
#                           UTILS                           #
# --------------------------------------------------------- #

def pretty_print_section(section: str) -> None:
    """Pretty print a header in order to make the script output look nice"""
    print(f'\n# ------ {section.upper()}\n')


def acc(yhat, y):
    """Computes accuracy"""
    return (yhat.round().clip(0, 1) == y).to(float).mean().item()


def build_model(activation_fun: F.Function, xavier_init: bool = True, model_name: str = None):
    """Build a deep neural network with a fixed architecture:
    2 input neurons, 3 hidden layers of 25 neurons each, 1 output unit"""
    model = Sequential(
        LinearLayer(2, 25, xavier_init=xavier_init),
        activation_fun(),
        LinearLayer(25, 25, xavier_init=xavier_init),
        activation_fun(),
        LinearLayer(25, 25, xavier_init=xavier_init),
        activation_fun(),
        LinearLayer(25, 1, xavier_init=xavier_init)
    )
    model._name = model_name
    return model


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
    {'activation_fun': F.ReLU, 'xavier_init': True, 'model_name': 'relu'},
    {'activation_fun': F.Tanh, 'xavier_init': False, 'model_name': 'tanh'},
    {'activation_fun': F.Sigmoid, 'xavier_init': False, 'model_name': 'sigmoid'}
]

# Learning rates of each model: selected by KFoldCV (run this script with --cv argument)
lrs = [
    0.01,
    0.005,
    0.1
]

if args.cv:
    results = []
    lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
    for m in models:
        print(f'model {m["model_name"]}')
        for lr in lrs:
            print(f'lr {lr}')
            res = kfold_cv(lambda: build_model(**m), X, target, F.MSELoss(), acc, lr=lr, epochs=20, k=5)
            train_acc = [e['acc_train'] for e in res]
            test_acc = [e['acc_test'] for e in res]
            results.append({
                'model': m["model_name"],
                'lr': lr,
                'acc_train': train_acc,
                'acc_test': test_acc
            })

    with open('cv_results.json', 'w') as f:
        json.dump(results, f)

else:
    models = [build_model(**kwargs) for kwargs in models]

    pretty_print_section('models')
    for m in models:
        print(m)

    # --- Train models
    train_dataset = Dataset(xtrain, ytrain)
    test_dataset = Dataset(xtest, ytest)

    pretty_print_section('training')
    results = []
    for model, lr in zip(models, lrs):
        print(f'model {model._name}\t', end='')
        log, train_acc, test_acc = train_and_test(model, train_dataset, test_dataset, F.MSELoss(), lr=lr, epochs=25)
        plt.semilogy(log['loss'], '-o', label=model._name)
        results.append((log, train_acc, test_acc))

    plt.xlabel('epochs')
    plt.ylabel('average MSE per epoch')
    plt.legend()
    plt.savefig('fig_loss.pdf', bbox_inches='tight')
    plt.show()

    # --- Performance evaluation
    pretty_print_section('performance evaluation')
    print('\t\t\tTrain accuracy\t\tTest accuracy')
    for r, mod in zip(results, models):
        _, acc_train, acc_test = r
        print(f'{mod._name}\t\t{acc_train:.3}\t\t\t\t{acc_test:.3}')


