# --------------------------------------------------------- #
#                          IMPORTS                          #
# --------------------------------------------------------- #
import matplotlib.pyplot as plt
import torch.optim as optim
from train import train
#############################
#     Stat train utils      #
#############################


def train_multiple_runs( network_class , runs  , epoch  , lr_  , criterion_ , debug_v ,  **kwargs):
  all_train_loss , all_train_acc , all_test_loss , all_test_acc = [],[],[],[]
  
  for i in range(runs):
    n = network_class(**kwargs)
    optimizer = optim.Adam(n.parameters(), lr=lr_)
    criterion = criterion_

    tot_train_loss , tot_train_acc , tot_test_loss , tot_test_acc = train(n , optimizer , criterion , epoch , debug_ = debug_v)
    all_train_loss.append(tot_train_loss)
    all_train_acc.append(tot_train_acc)
    all_test_loss.append(tot_test_loss)
    all_test_acc.append(tot_test_acc)

  return all_train_loss , all_train_acc , all_test_loss , all_test_acc

import numpy as np


def plot_std_loss_acc(all_train_loss , all_train_acc , all_test_loss , all_test_acc , color = ''):
  trl_mean = np.array(all_train_loss).mean(axis = 0)
  tra_mean = np.array(all_train_acc).mean(axis = 0)
  tel_mean = np.array(all_test_loss).mean(axis = 0)
  tea_mean = np.array(all_test_acc).mean(axis = 0)

  trl_std = np.array(all_train_loss).std(axis = 0)
  tra_std = np.array(all_train_acc).std(axis = 0)
  tel_std = np.array(all_test_loss).std(axis = 0)
  tea_std = np.array(all_test_acc).std(axis = 0)

  epochs = range(1, len(tea_std)+1 )

  early_stopping = np.argmin( tel_mean )
  print(f"at epoch {early_stopping} mean test acc : { tea_mean[early_stopping] }")
  print(f"mean last test acc : { tea_mean [-1] }")

  temp = [ [trl_mean , trl_std , 'g' , 'trl'] ,
          [tel_mean , tel_std  , 'b' , 'tel'] ,
          [tra_mean , tra_std  , 'r' , 'tra'] ,
          [tea_mean , tea_std  , 'y' , 'tea'] ]

  if color != '':
    plt.plot( epochs , tea_mean , c=color )
    plt.fill_between(epochs, tea_mean-tea_std, tea_mean+tea_std ,alpha=0.1 , color=color )
    #plt.ylim((-0.1,1.1))
    return

  for g in temp : 
    plt.plot( epochs , g[0] , c=g[2] , label = g[3])
    plt.fill_between(epochs, g[0]-g[1], g[0]+g[1] ,alpha=0.3 , color=g[2] )

  plt.legend()
  plt.ylim((-0.1,1.1))
  plt.show()