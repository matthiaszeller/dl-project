"""
Perform cross validation and performance evaluation by running models multiple times, and generate plots.

WARNING: this takes a long time to terminate.
"""

# --------------------------------------------------------- #
#                          IMPORTS                          #
# --------------------------------------------------------- #


import torch.nn.functional as F

# --- Custom Imports
from matplotlib import pyplot as plt

from models import FCNN3, \
    WS2, WS1, WS3, AL1, \
    AL3, AL2, AL4, FCNN_1LAYER, FCNN_2LAYER, FCNN_3LAYER, FCNN_4LAYER
from stats import train_multiple_runs, plot_std_loss_acc
from train import custom_loss_BCELoss_CELoss, initialize_dataset

initialize_dataset()

# --------------------------------------------------------- #
#                STATISTICS ON MULTIPLE RUNS                #
# --------------------------------------------------------- #

# Learning rates of this section were determined by cross validation (done at the end of this script).

print('\n\n\n',
      '################### \n',
      '#    Stat graph   # \n',
      '################### \n\n')

# ------------------------------------ #
#        COMPARE AUXILIARY LOSS        #
# ------------------------------------ #

print('>> Generate figure 7, assess benefit of auxiliary loss')

stats = [
    [1000, 0.0005, 'r', WS2, F.binary_cross_entropy],
    [1000, 0.0005, 'r', AL3, custom_loss_BCELoss_CELoss],

    [1000, 0.0005, 'black', FCNN3, F.binary_cross_entropy],
    [1000, 0.0005, 'black', AL1, custom_loss_BCELoss_CELoss],

    [-1, 0.0008, 'y', WS1, F.binary_cross_entropy],
    [-1, 0.0008, 'y', AL2, custom_loss_BCELoss_CELoss],

    [-1, 0.002, 'b', WS3, F.binary_cross_entropy],
    [-1, 0.002, 'b', AL4, custom_loss_BCELoss_CELoss],
]

for el in stats:
    print(f'--- configuration : {el}')
    all_train_loss, all_train_acc, all_test_loss, all_test_acc = train_multiple_runs(el[3],
                                                                                     runs=5,
                                                                                     epoch=30,
                                                                                     lr_=el[1],
                                                                                     criterion_=el[4],
                                                                                     debug_v=False,
                                                                                     nodes_nb=el[0])
    plot_std_loss_acc(all_train_loss, all_train_acc, all_test_loss, all_test_acc, color=el[2])

plt.show()

# ------------------------------------ #
#        COMPARE WEIGHT SHARING        #
# ------------------------------------ #

print('>> Generate figure 6. Compare weight sharing with FCNN.')

stats = [
    [1000, 0.0005, 'r', WS2],
    [1000, 0.0005, 'black', FCNN3],
    [-1, 0.0008, 'y', WS1],
    [-1, 0.002, 'b', WS3],
]

for el in stats:
    print(f'--- configuration : {el}')
    all_train_loss, all_train_acc, all_test_loss, all_test_acc = train_multiple_runs(el[3],
                                                                                     runs=5,
                                                                                     epoch=30,
                                                                                     lr_=el[1],
                                                                                     criterion_=F.binary_cross_entropy,
                                                                                     debug_v=False,
                                                                                     nodes_nb=el[0])
    plot_std_loss_acc(all_train_loss, all_train_acc, all_test_loss, all_test_acc, color=el[2])

plt.show()

# ------------------------------------ #
#    COMPARE FULLY CONNECTED MODELS    #
# ------------------------------------ #

print('>> Generate figure 5: FCNNs with different number of layers and neurons.')

stats = [
    [200, 0.001, 'r', FCNN3],
    [500, 0.001, 'r', FCNN3],
    [700, 0.0007, 'r', FCNN3],
    [1000, 0.0005, 'r', FCNN3],
    [1500, 0.0002, 'r', FCNN3],

    [200, 0.0004, 'b', FCNN_1LAYER],
    [500, 0.00025, 'b', FCNN_1LAYER],
    [700, 0.0004, 'b', FCNN_1LAYER],
    [1000, 0.00025, 'b', FCNN_1LAYER],

    [-1, 0.0001, 'g', FCNN_2LAYER],
    [-1, 0.0005, 'g', FCNN_3LAYER],
    [-1, 0.0005, 'g', FCNN_4LAYER]
]

for el in stats:
    print(f'--- configuration : {el}')
    all_train_loss, all_train_acc, all_test_loss, all_test_acc = train_multiple_runs(el[3],
                                                                                     runs=5,
                                                                                     epoch=30,
                                                                                     lr_=el[1],
                                                                                     criterion_=F.binary_cross_entropy,
                                                                                     debug_v=True,
                                                                                     nodes_nb=el[0])
    plot_std_loss_acc(all_train_loss, all_train_acc, all_test_loss, all_test_acc, color=el[2])

plt.show()

# --------------------------------------------------------- #
#                   HYPERPARAMETER TUNING                   #
# --------------------------------------------------------- #

models = [
    WS2, WS1, WS3, AL1, AL3, AL2, AL4, FCNN_1LAYER, FCNN_2LAYER, FCNN_3LAYER, FCNN_4LAYER
]

for model in models:
    print(f'model {model}')
    for lr in [0.00005, 0.00007, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.0004]:
        print(f'at LR = {lr}')
        all_train_loss, all_train_acc, all_test_loss, all_test_acc = train_multiple_runs(model,
                                                                                         runs=1,
                                                                                         epoch=25,
                                                                                         lr_=lr,
                                                                                         criterion_=F.binary_cross_entropy,
                                                                                         debug_v=True,
                                                                                         nodes_nb=-1)
        plot_std_loss_acc(all_train_loss, all_train_acc, all_test_loss, all_test_acc)

    plt.show()
