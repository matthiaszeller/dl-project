"""
Main script which runs models once and displays training & test accuracies.
"""

# --------------------------------------------------------- #
#                          IMPORTS                          #
# --------------------------------------------------------- #

import torch.nn.functional as F

# --- Custom Imports
from models import FCNN3, \
    WS2, WS1, WS3, AL1, \
    AL3, AL2, AL4, FCNN_1LAYER, FCNN_2LAYER, FCNN_3LAYER, FCNN_4LAYER
from stats import train_multiple_runs
from train import custom_loss_BCELoss_CELoss, initialize_dataset

initialize_dataset()

print('######################\n',
      '#      PROJECT 1     #\n',
      '######################\n')

print('Authors :  \n',
      '-- Matthias \n',
      '-- Fatih   \n',
      '-- Etienne \n\n')

print('>> One run model : FCNN3 : \n')

res = train_multiple_runs(FCNN3,
                          runs=1,
                          epoch=25,
                          lr_=0.00015,
                          criterion_=F.binary_cross_entropy,
                          debug_v=True,
                          nodes_nb=1000)

print('>> One run model : WS2 : \n')

# WS_FCNN_ image_specific1
res = train_multiple_runs(WS2,
                          runs=1,
                          epoch=25,
                          lr_=0.0005,
                          criterion_=F.binary_cross_entropy,
                          debug_v=True,
                          nodes_nb=1000)

print('>> One run model : WS1 : \n')

res = train_multiple_runs(WS1,
                          runs=1,
                          epoch=25,
                          lr_=0.0008,
                          criterion_=F.binary_cross_entropy,
                          debug_v=True,
                          nodes_nb=-1)

print('>> One run model : WS3 : \n')

res = train_multiple_runs(WS3,
                          runs=1,
                          epoch=25,
                          lr_=0.0008,
                          criterion_=F.binary_cross_entropy,
                          debug_v=True,
                          nodes_nb=-1)

print('>> One run model : AL1 : \n')

res = train_multiple_runs(AL1,
                          runs=1,
                          epoch=25,
                          lr_=0.0005,
                          criterion_=custom_loss_BCELoss_CELoss,
                          debug_v=True,
                          nodes_nb=1000)

print('>> One run model : AL3 : \n')

res = train_multiple_runs(AL3,
                          runs=1,
                          epoch=25,
                          lr_=0.0005,
                          criterion_=custom_loss_BCELoss_CELoss,
                          debug_v=True,
                          nodes_nb=1000)

print('>> One run model : AL2 : \n')

res = train_multiple_runs(AL2,
                          runs=1,
                          epoch=25,
                          lr_=0.0008,
                          criterion_=custom_loss_BCELoss_CELoss,
                          debug_v=True,
                          nodes_nb=1000)

print('>> One run model : AL4 : \n')

res = train_multiple_runs(AL4,
                          runs=1,
                          epoch=25,
                          lr_=0.002,
                          criterion_=custom_loss_BCELoss_CELoss,
                          debug_v=True,
                          nodes_nb=-1)


