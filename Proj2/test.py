"""
Test file to generate a dataset and train simple networks with three hidden layers.
torch.autograd is disabled with `torch.set_grad_enabled(False)`

Example of usage:
    - `python test.py`: run each model once on the generated dataset, generate figure for learning curves
    - `python test.py --cv`: perform 5-fold cross validation on the learning rate
    - `python test.py --test`: evaluate model performance with 5-fold CV, given learning rates
"""

# --------------------------------------------------------- #
#                          IMPORTS                          #
# --------------------------------------------------------- #

# --- Standard modules
import argparse
import json
from math import pi
from typing import Union, List

from torch import empty, tensor, set_grad_enabled

# --- Visualization-related imports, not used for the delivered code
# from math import sqrt
# from matplotlib import pyplot as plt
# from torch import linspace, vstack

# --- Our custom modules
import function as F
from module import LinearLayer, Sequential
from training import Dataset, train_SGD, kfold_cv


# --- Disable globally autograd
set_grad_enabled(False)


parser = argparse.ArgumentParser()
parser.add_argument('--cv', help='cross validate learning rate', action='store_true')
parser.add_argument('--stats', help='evaluate models with KFoldCV', action='store_true')
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


def build_model(activation_fun: F.Function, xavier_init: bool = True, model_name: str = None, clip=False):
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
    if clip:
        model.add_layer(F.Sigmoid())
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
    yhat_train, ytrain = tensor([[model(x).item(), y.item()] for x, y in train_set]).T
    yhat_test, ytest = tensor([[model(x).item(), y.item()] for x, y in test_set]).T
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
X = empty(n, d)
X.uniform_()

# --- Generate target
# Compute distance of each point from (0.5, 0.5)
center = tensor([0.5, 0.5])
squared_dist = ((X - center)**2).sum(axis=1)
radius_squared = 1 / 2 / pi
target = (squared_dist < radius_squared) * 1
# Decision boundary (for plotting)
# angles = linspace(0., 2. * pi, 200)
# decision_bdry_true = vstack([angles.cos(), angles.sin()]).T * sqrt(radius_squared) + center

# --- Data splitting - Training & test sets
# Important note: we don't need to shuffle the data, as this each sample is already random
train_ratio = 0.8
split_point = int(train_ratio * n)
xtrain, xtest = X[:split_point], X[split_point:]
ytrain, ytest = target[:split_point], target[split_point:]


# --- Plotting data
# plt.figure(figsize=(5, 5))
# plt.scatter(xtrain[ytrain == 1, 0], xtrain[ytrain == 1, 1], marker='o', c='m', label='target 1, training')
# plt.scatter(xtrain[ytrain == 0, 0], xtrain[ytrain == 0, 1], marker='o', c='b', label='target 0, training')
# plt.scatter(xtest[ytest == 1, 0], xtest[ytest == 1, 1], marker='x', c='m', label='target 1, test')
# plt.scatter(xtest[ytest == 0, 0], xtest[ytest == 0, 1], marker='x', c='b', label='target 0, test')
#
# # Plot decision boundaries
# plt.plot(decision_bdry_true[:, 0], decision_bdry_true[:, 1], 'k--', label='true decision boundary')
#
# plt.xlabel('x1')
# plt.ylabel('x2')
# legend = plt.legend(bbox_to_anchor=(1.05, 1))
# plt.savefig('fig_dataset.pdf', bbox_inches='tight')
# plt.show()

# --- Data normalization
mu, std = xtrain.mean(0), xtrain.std(0)
xtrain = (xtrain - mu) / std
xtest  = (xtest  - mu) / std


# -------------------------------------------------------- #
#                          MODELS                          #
# -------------------------------------------------------- #

# Define models
models = [
    {'activation_fun': F.ReLU, 'xavier_init': True, 'model_name': 'relu', 'clip': False},
    {'activation_fun': F.ReLU, 'xavier_init': True, 'model_name': 'relu clipped', 'clip': True},
    {'activation_fun': F.Tanh, 'xavier_init': False, 'model_name': 'tanh', 'clip': False},
    {'activation_fun': F.Tanh, 'xavier_init': False, 'model_name': 'tanh clipped', 'clip': True},
    {'activation_fun': F.Sigmoid, 'xavier_init': False, 'model_name': 'sigmoid', 'clip': False},
    {'activation_fun': F.Sigmoid, 'xavier_init': False, 'model_name': 'sigmoid clipped', 'clip': True}
]

# -------------------------------------------------------- #
#                     CROSS VALIDATION                     #
# -------------------------------------------------------- #

# Learning rates selected by KFoldCV (run this script with --cv argument)
# the values in the list correspond (by position in the list) to the models defined right above
lrs = [
    0.01,
    0.1,
    0.005,
    0.01,
    0.1,
    0.06
]

# --- Cross validating learning rates
if args.cv:
    results = []
    # Grid for learning rates, specific for each model
    lrs = [
        [0.008, 0.01, 0.02, 0.03, 0.05],
        [0.03, 0.05, 0.1, 0.15, 0.2],
        [0.003, 0.005, 0.008, 0.01],
        [0.003, 0.005, 0.01, 0.02],
        [0.05, 0.08, 0.1, 0.2, 0.3],
        [0.04, 0.05, 0.06, 0.07],
    ]
    for i, m in enumerate(models):
        print(f'model {m["model_name"]}')
        for lr in lrs[i]:
            print(f'lr {lr}')
            res = kfold_cv(lambda: build_model(**m), X, target, F.MSELoss(), acc, lr=lr, epochs=30, k=5)
            # Log results
            train_acc = [e['acc_train'] for e in res]
            test_acc = [e['acc_test'] for e in res]
            results.append({
                'model': m["model_name"],
                'lr': lr,
                'acc_train': train_acc,
                'acc_test': test_acc
            })
    # Output results to file
    with open('cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)

# -------------------------------------------------------- #
#                  PERFORMANCE EVALUATION                  #
# -------------------------------------------------------- #

# Evaluate performance with 5-fold CV, 2 times, for each model
elif args.stats:
    results = []
    n_runs = 2
    for m, lr in zip(models, lrs):
        print('statistics for model', m["model_name"])
        for i in range(n_runs):
            print(f'run {i}')
            res = kfold_cv(lambda: build_model(**m), X, target, F.MSELoss(), acc, lr=lr, epochs=30, k=5)
            train_acc = [e['acc_train'] for e in res]
            test_acc = [e['acc_test'] for e in res]
            results.append({
                'model': m["model_name"],
                'lr': lr,
                'acc_train': train_acc,
                'acc_test': test_acc
            })

    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

# --------------------------------------------------------- #
#                      STANDARD SCRIPT                      #
# --------------------------------------------------------- #

# This part is run if no argument is provided
else:
    models = [build_model(**kwargs) for kwargs in models]

    pretty_print_section('models')
    for m in models:
        print(m)

    # --- Train models
    train_dataset = Dataset(xtrain, ytrain, shuffle=True)
    test_dataset = Dataset(xtest, ytest)

    pretty_print_section('training')
    results = []
    for model, lr in zip(models, lrs):
        print(f'model {model._name}\t', end='')
        log, train_acc, test_acc = train_and_test(model, train_dataset, test_dataset, F.MSELoss(), lr=lr, epochs=30)
        # plt.semilogy(log['loss'], '-o', label=model._name)
        results.append((log, train_acc, test_acc))

    # --- Plotting
    # plt.xlabel('epochs')
    # plt.ylabel('average MSE per epoch')
    # plt.legend()
    # plt.savefig('fig_loss.pdf', bbox_inches='tight')
    # plt.show()

    # --- Performance evaluation
    pretty_print_section('performance evaluation')
    print(f'{"Train accuracy":>35}{"Test accuracy":>18}')
    for r, mod in zip(results, models):
        _, acc_train, acc_test = r
        print(f'{mod._name:<21}{acc_train:.3}\t\t\t\t{acc_test:.3}')


