#!/usr/bin/env python
# coding: utf-8

import argparse
import os


parser = argparse.ArgumentParser()
# data param
parser.add_argument('--window_size',type=int, default = 10, help = 'time window size (s)')
parser.add_argument('--sample_time',type=int, default = 1, help = 'sample frequency, e.g. 2 means 1 sample every 2 seconds')
parser.add_argument('--padding',type=bool, default = False, help= 'input vehicle features padded')

    
# model param
parser.add_argument('--hidden_dim',type=int, default = 10)
parser.add_argument('--drop_out',type=float, default = 0.2)
parser.add_argument('--rnn_layers',type=int, default = 3)


# training param
parser.add_argument('--learning_rate',type=float, default = 1e-3)
parser.add_argument('--GPU',type=str, default = "1")
parser.add_argument('--epochs',type=int, default = 200)
parser.add_argument('--train_file_name',type=str, default = 'nan', help='parent dir for training data file')
parser.add_argument('--experiment_name',type=str, default = '', help='unique experiment name for saving to dir')
parser.add_argument('--verbal',type=bool, default = False, help='print training results')
parser.add_argument('--batch_size',type=int, default = 512)
parser.add_argument('--more_training',action='store_true', default = False, help = 'more free flow training data')
# parser.add_argument('--no_acc',action='store_true', default = False, help = 'no acceleration data')
parser.add_argument('--recon_speed',action='store_true', default = False, help = 'reconstruct speed and trajectory')
parser.add_argument('--recon_traj',action='store_true', default = False, help = 'reconstruct trajectory')

args = parser.parse_args()

print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

import torch
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import random
import json
import torch.nn as nn
from torch.utils import data
from torch.nn import functional as F
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   
from random import shuffle
import pickle
import time
import math
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from util import plot_ROC_curve, search_thresholds_F1
from util import cal_loss_each, cal_graph_loss, cal_loss
from util import normalize, scale_back

from myDataset_Seq2Seq import trafficDataset_Seq2Seq
from model_seq2seq import Seq2Seq

np.random.seed(seed=1)
random.seed(1)
torch.manual_seed(1)


result_sub_dir = "../expriments/{}_{}/".format(args.train_file_name, args.experiment_name)
if not os.path.exists(result_sub_dir):
    os.makedirs(result_sub_dir)


# data dir
file_name = args.train_file_name #'test0_processed_categ'
train_dir = '../Data/{}/training/'.format(file_name)

# if args.experiment_name in ['Seq2Seq', 'Seq2Seq_traj']:
save_name = 'hiddim-{}-lr-{}-windowsize-{}-rnnlayer-{}-recon_speed-{}'.format(args.hidden_dim, args.learning_rate, args.window_size, args.rnn_layers, args.recon_speed)



# save arg
arg_path = os.path.join(result_sub_dir, "arg-{}.txt".format(save_name))
with open(arg_path, 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# parser = ArgumentParser()
# args = parser.parse_args()
# with open('commandline_args.txt', 'r') as f:
#     args.__dict__ = json.load(f)
    

# Load data
if args.more_training:
    train_list = list(np.load(train_dir+'train_list_more.npy')) 
else:
    train_list = list(np.load(train_dir+'train_list.npy'))

# In[8]:

all_maxs, all_mins = pickle.load( open( train_dir+"min_max.p", "rb" ) )
max_dist, min_dist  = all_maxs.Mileage, all_mins.Mileage 

# In[9]:


partition_train, partition_val= train_test_split(train_list, test_size=0.1, random_state=42)
# len(partition_train), len(partition_val), len(test_list )


# In[10]:
window_size = args.window_size
sample_time = args.sample_time

params_train = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 0}

params_val = {'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 0}

train_dataset = trafficDataset_Seq2Seq(partition_train,train_dir,window_size, sample_time )
val_dataset = trafficDataset_Seq2Seq(partition_val,train_dir,window_size, sample_time )

# loaders
train_loader = DataLoader(train_dataset, **params_train)
val_loader = DataLoader(val_dataset,**params_val )


# ## Model

input_feat_dim = 4
hidden_dim = args.hidden_dim
dropout = args.drop_out
n_layers = args.rnn_layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Seq2Seq( input_feat_dim = input_feat_dim,
                hidden_dim = hidden_dim, 
                dropout= dropout, 
                n_layers = n_layers,
               decode_steps = window_size,
               recon_speed = args.recon_speed,
               recon_traj = args.recon_traj
               )


model.float()


# In[14]:


lr = args.learning_rate
# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

def traj_loss(obs_y,y_pred):
    traj = obs_y[:,:,[2,-1]]
    return criterion(traj, traj_pred)

def recon_speed_loss(obs_y,y_pred):
    traj = obs_y[:,:,[2,3,-1]]
    return criterion(traj, traj_pred)


def train(epoch, train_loader ,test_loader, best_val, best_epoch, verbal = True):
    model.train()
    train_loss = 0
    loss_train = []
    loss_val = []
    for x_traj  in train_loader:
        x_traj = x_traj.to(device) # x_traj- [batch, seq_len, input_dim]
        optimizer.zero_grad()
        x_traj = torch.permute(x_traj,(1,0,2))
        # x_traj- [seq_len, batch, input_dim]
        # input_dim - "Lane", "Class","x","Speed", "Acceleration","Mileage"
        y_pred = model(x_traj)
        # loss
        if args.recon_speed:
            obs_y = x_traj[:,:,[2,3,-1]]
        elif args.recon_traj:
            obs_y = x_traj[:,:,[2,-1]]
        else:
            obs_y = x_traj[:, :, 2:]
        loss = criterion(obs_y,y_pred)
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
    for x_traj in val_loader:
        # evaluation
        model.eval()
        x_traj = x_traj.to(device)
        x_traj = torch.permute(x_traj,(1,0,2))
        # x_traj- [seq_len, batch, input_dim]
        if args.recon_speed:
            obs_y = x_traj[:,:,[2,3,-1]]
        elif args.recon_traj:
            obs_y = x_traj[:,:,[2,-1]]
        else:
            obs_y = x_traj[:, :, 2:]
            
        with torch.no_grad():
            y_pred = model(x_traj)

        mse_val = criterion(obs_y,y_pred)
        loss_val.append(mse_val.item())
    
    loss_train = sum(loss_train) / len(loss_train)
    loss_val = sum(loss_val) / len(loss_val)
 
    # print results
    if verbal and epoch % 3 == 0:
        print('Train Epoch: {}  loss: {:e}  val_loss: {:e}'.format(
            epoch, loss_train, loss_val ))
        rmse =  math.sqrt(loss_val)
        print('validation speed rmse mean: {:e}'.format(rmse))
    
    #  early-stopping
    if loss_val < best_val:
        torch.save({
            'epoch' : epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': lr
        }, model_path)
        best_val = loss_val
        best_epoch = epoch

    return  loss_train, loss_val, best_val, best_epoch


# # Train

# In[17]:


loss_track = []
val_track = []
model_path = os.path.join(result_sub_dir, "model-{}.pt".format(save_name))



model = model.to(device)
n_epochs = args.epochs
start = time.time()
best_epoch = 0
best_val = float('inf') # for early stopping
lr_decay_step_size = 50

for epoch in tqdm(range(0, n_epochs+1)):
    train_loss, val_loss, best_val,best_epoch = train(epoch, train_loader, val_loader, best_val, best_epoch, verbal = args.verbal)
    loss_track.append(train_loss)
    val_track.append(val_loss)
    if epoch % lr_decay_step_size == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']
    
print("time for {} epochs: {:.3f} min".format(n_epochs, (time.time() - start)/60))


# save model at final epoch
final_model_path = os.path.join(result_sub_dir, "model-final_epoch-{}.pt".format(save_name))
torch.save({
            'epoch' : epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': lr
        }, final_model_path)




# In[19]:


torch.cuda.empty_cache()


# In[20]:


# plot learning curve
plt.plot(np.array(loss_track), label = 'training')
plt.plot(np.array(val_track), label = 'validaiton')
plt.title("loss")
plt.xlabel("# epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.title("{}\n{}".format(result_sub_dir, save_name))
# plt.ylim(0.4, 1)
fig_path = os.path.join(result_sub_dir, "curve-{}.pdf".format(save_name))
plt.savefig(fig_path)

#save val loss
val_path = os.path.join(result_sub_dir, "val_loss-{}.npy".format(save_name))

np.save(val_path, [val_track[best_epoch],val_track[-1]])


