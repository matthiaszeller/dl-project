

# --------------------------------------------------------- #
#                          IMPORTS                          #
# --------------------------------------------------------- #

import torch.nn.functional as F

# --- Custom Imports
from models import FCNN3, \
    WS2, WS1, WS3, AL1, \
    AL3, AL2, AL4
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
                          lr_=0.0007,
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
         [1000  ,0.0005  ,'r' , WS2       , F.binary_cross_entropy ] ,
         [1000  ,0.0005  ,'r' , AL3    , custom_loss_BCELoss_CELoss ] ,


         [1000  ,0.0005  ,'black' , FCNN3     , F.binary_cross_entropy ] ,
         [1000  ,0.0005  ,'black' , AL1  , custom_loss_BCELoss_CELoss ] ,


         [ -1    ,0.0008 , 'y' ,WS1          , F.binary_cross_entropy ] ,
         [ -1    ,0.0008 , 'y' ,AL2       , custom_loss_BCELoss_CELoss ] ,

         [ -1     ,0.002  , 'b' ,WS3       , F.binary_cross_entropy ] ,
         [ -1    ,0.002  , 'b' ,AL4     , custom_loss_BCELoss_CELoss ] ,
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
         [1000  ,0.0005  ,'r' , WS2] ,
         [1000  ,0.0005  ,'black' , FCNN3   ] ,
         [ -1     ,0.0008 , 'y' ,WS1 ] ,
         [ -1    ,0.002 , 'b' ,WS3 ] ,
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

stats = [[200  ,0.001  ,'r' , FCNN3] ,
         [500  ,0.001  ,'r' , FCNN3] ,
         [700  ,0.0007 ,'r' , FCNN3] ,
         [1000 ,0.0005 ,'r' , FCNN3] ,
         [1500 ,0.0002 ,'r' , FCNN3] ,
         
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
