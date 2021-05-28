# --------------------------------------------------------- #
#                          IMPORTS                          #
# --------------------------------------------------------- #
import torch.nn as nn
import torch.nn.functional as F


import os

# --- Custom Imports
from models import FCNN_1LAYER, FCNN_2LAYER, FCNN_3LAYER, FCNN_4LAYER, FCNN_image_specific, \
  WS_FCNN_image_specific1, CNN_image_specific1, WS_CNN_image_specific1, AL_FCNN_image_specific,\
     AL_WS_FCNN_image_specific1, AL_CNN_image_specific1, AL_WS_CNN_image_specific
from train import custom_loss_BCELoss_CELoss, initialize_dataset
initialize_dataset()
from stats import train_multiple_runs

print(' ######################\n',
       '#      PROJECT 1     #\n',
       '######################\n')

print('Authors :  \n',
      '-- Mathias \n',
      '-- Fatih   \n',
      '-- Etienne \n\n')


print('>> One run model : FCNN_image_specific : \n')

all_train_loss , all_train_acc , all_test_loss , all_test_acc = train_multiple_runs( FCNN_image_specific ,
                                                                                     runs = 1 ,
                                                                                     epoch = 25 ,
                                                                                     lr_= 0.0007  ,
                                                                                     criterion_ = F.binary_cross_entropy,
                                                                                      debug_v = True,
                                                                                     nodes_nb = 700 )




print('>> One run model : WS_FCNN_image_specific1 : \n')

all_train_loss , all_train_acc , all_test_loss , all_test_acc = train_multiple_runs( WS_FCNN_image_specific1 ,
                                                                                     runs = 1 ,
                                                                                     epoch = 25 ,
                                                                                     lr_= 0.0005 ,
                                                                                     criterion_ = F.binary_cross_entropy,
                                                                                      debug_v = True,
                                                                                     nodes_nb = 1000 )


print('>> One run model : WS_CNN_image_specific1 : \n')

all_train_loss , all_train_acc , all_test_loss , all_test_acc = train_multiple_runs( WS_CNN_image_specific1 ,
                                                                                     runs = 1 ,
                                                                                     epoch = 25 ,
                                                                                     lr_= 0.002 ,
                                                                                     criterion_ = F.binary_cross_entropy,
                                                                                      debug_v = True,
                                                                                     nodes_nb = -1 )
                                                                                    
                                                                                     
print('>> One run model : AL_WS_FCNN_image_specific1 : \n')

all_train_loss , all_train_acc , all_test_loss , all_test_acc = train_multiple_runs( AL_WS_FCNN_image_specific1 ,
                                                                                     runs = 1 ,
                                                                                     epoch = 25 ,
                                                                                     lr_= 0.0005 ,
                                                                                     criterion_ = custom_loss_BCELoss_CELoss,
                                                                                      debug_v = True,
                                                                                     nodes_nb = 1000 )

'''
print('\n\n\n',
      '################### \n',
      '#    Stat graph   # \n',
      '################### \n\n')


print('>> Auxiliary Loss')
###########################################################################
######################### AL GRAPH ########################################
###########################################################################

stats = [
         [1000  ,0.0005  ,'r' , WS_FCNN_image_specific1       , F.binary_cross_entropy ] ,
         [1000  ,0.0005  ,'r' , AL_WS_FCNN_image_specific1    , custom_loss_BCELoss_CELoss ] ,


         [1000  ,0.0005  ,'black' , FCNN_image_specific     , F.binary_cross_entropy ] ,
         [1000  ,0.0005  ,'black' , AL_FCNN_image_specific  , custom_loss_BCELoss_CELoss ] ,


         [ -1    ,0.0008 , 'y' ,CNN_image_specific1          , F.binary_cross_entropy ] ,
         [ -1    ,0.0008 , 'y' ,AL_CNN_image_specific1       , custom_loss_BCELoss_CELoss ] ,

         [ -1     ,0.002  , 'b' ,WS_CNN_image_specific1       , F.binary_cross_entropy ] ,
         [ -1    ,0.002  , 'b' ,AL_WS_CNN_image_specific     , custom_loss_BCELoss_CELoss ] ,
         ]

for el in stats:
  print(f'--- configuration : {el}')
  all_train_loss , all_train_acc , all_test_loss , all_test_acc = train_multiple_runs( el[3] ,
                                                                                     runs = 5 ,
                                                                                     epoch = 30 ,
                                                                                     lr_= el[1] ,
                                                                                     criterion_ = el[4],
                                                                                      debug_v = False,
                                                                                     nodes_nb = el[0] )
  # plot_std_loss_acc(all_train_loss , all_train_acc , all_test_loss , all_test_acc , color = el[2])


print('>> Weight Sharing')
###########################################################################
######################### WS GRAPH ########################################
###########################################################################

stats = [
         [1000  ,0.0005  ,'r' , WS_FCNN_image_specific1] ,
         [1000  ,0.0005  ,'black' , FCNN_image_specific   ] ,
         [ -1     ,0.0008 , 'y' ,CNN_image_specific1 ] ,
         [ -1    ,0.002 , 'b' ,WS_CNN_image_specific1 ] ,
         ]

for el in stats:
  print(f'--- configuration : {el}')
  all_train_loss , all_train_acc , all_test_loss , all_test_acc = train_multiple_runs( el[3] ,
                                                                                     runs = 5 ,
                                                                                     epoch = 30 ,
                                                                                     lr_= el[1] ,
                                                                                     criterion_ = F.binary_cross_entropy,
                                                                                      debug_v = False,
                                                                                     nodes_nb = el[0] )
  # plot_std_loss_acc(all_train_loss , all_train_acc , all_test_loss , all_test_acc , color = el[2])


print('>> Fully Connected')


###########################################################################
####################### FCNN GRAPH ########################################
###########################################################################

stats = [[200  ,0.001  ,'r' , FCNN_image_specific] ,
         [500  ,0.001  ,'r' , FCNN_image_specific] ,
         [700  ,0.0007 ,'r' , FCNN_image_specific] ,
         [1000 ,0.0005 ,'r' , FCNN_image_specific] ,
         [1500 ,0.0002 ,'r' , FCNN_image_specific] ,
         
         [200 ,0.0004  ,'b', FCNN_1LAYER],
         [500 ,0.00025 ,'b', FCNN_1LAYER],
         [700 ,0.0004  ,'b', FCNN_1LAYER],
         [1000,0.00025 ,'b', FCNN_1LAYER],

         [ -1 ,0.0001 ,'g', FCNN_2LAYER],
         [ -1 ,0.0005 ,'g', FCNN_3LAYER],
         [ -1 ,0.0005 ,'g', FCNN_4LAYER]
         
         ]

for el in stats:
  print(f'--- configuration : {el}')
  all_train_loss , all_train_acc , all_test_loss , all_test_acc = train_multiple_runs( el[3] ,
                                                                                     runs = 5 ,
                                                                                     epoch = 30 ,
                                                                                     lr_= el[1] ,
                                                                                     criterion_ = F.binary_cross_entropy,
                                                                                      debug_v = True,
                                                                                     nodes_nb = el[0] )
  # plot_std_loss_acc(all_train_loss , all_train_acc , all_test_loss , all_test_acc , color = el[2])

## Brute force lr investigation

for lr in [0.00005 , 0.00007 , 0.0001 , 0.00015 , 0.0002 , 0.00025 , 0.0003 , 0.0004]:
  print(f'at LR = {lr}')
  all_train_loss , all_train_acc , all_test_loss , all_test_acc = train_multiple_runs( WS_CNN_image_specific ,
                                                                                     runs = 1 ,
                                                                                     epoch = 25 ,
                                                                                     lr_= lr ,
                                                                                     criterion_ = F.binary_cross_entropy,
                                                                                      debug_v = True,
                                                                                     nodes_nb = -1 )
  # plot_std_loss_acc(all_train_loss , all_train_acc , all_test_loss , all_test_acc)
'''