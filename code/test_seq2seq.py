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
parser.add_argument('--no_acc',action='store_true', default = False, help = 'no acceleration data')


# testing parameter
parser.add_argument('--test_file_name',type=str, default = 'nan', help = 'parent dir for training data file')
parser.add_argument('--test_scenario_file',type=str, default = 'nan', help = 'test scenario, same as file name')
parser.add_argument('--dist_stretch',type=float, default = 0.1, help= 'stretch distance for anomaly detecting')
parser.add_argument('--decode_steps',type=int, default =10 )
parser.add_argument('--loss_type',type=str, default = 'nll_cel', help = 'loss type as anomaly score. nll_cel or l2')

parser.add_argument('--recon_speed',action='store_true', default = False, help = 'reconstruct speed and trajectory')
parser.add_argument('--recon_traj',action='store_true', default = False, help = 'reconstruct trajectory')

# parser.add_argument('--recon_va',action='store_true', default = False, help='reonstruct speed and acc as part of loss')

args = parser.parse_args()

print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

import torch
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import random
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

# data dir
file_name = args.train_file_name #'test0_processed_categ'
train_dir = '../Data/{}/training/'.format(file_name)

# data dir
file_name = args.train_file_name #'test0_processed_categ'
train_dir = '../Data/{}/training/'.format(file_name)

# if args.experiment_name in ['Seq2Seq', 'Seq2Seq_traj']:
save_name = save_name = 'hiddim-{}-lr-{}-windowsize-{}-rnnlayer-{}-recon_speed-{}'.format(args.hidden_dim, args.learning_rate, args.window_size, args.rnn_layers, args.recon_speed)
# else:
#     pass


# # read arg from training
# arg_path = os.path.join(result_sub_dir, "arg-{}.txt".format(save_name))
# parser = argparse.ArgumentParser()
# args = parser.parse_args()
# with open(arg_path, 'r') as f:
#     args.__dict__ = json.load(f)
    


# In[8]:

all_maxs, all_mins = pickle.load( open( train_dir+"min_max.p", "rb" ) )
max_dist, min_dist  = all_maxs.Mileage, all_mins.Mileage 

# In[9]:

# In[10]:
window_size = args.window_size
sample_time = args.sample_time



# ## Model

# In[12]:


# ## Model

input_feat_dim = 4
hidden_dim = args.hidden_dim
output_dim = input_feat_dim
dropout = args.drop_out
n_layers = args.rnn_layers #rnn depth

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

model_path = os.path.join(result_sub_dir, "model-final_epoch-{}.pt".format(save_name))


# ## examine results

# In[21]:



checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

epoch = checkpoint['epoch']
# model_setting = checkpoint['model_setting']

print("best model is at epoch", epoch)


## test set

file_name_test = file_name if args.test_file_name == 'nan' else args.test_file_name

test_dir =  '../Data/{}/{}/'.format(file_name_test,args.test_scenario_file)

test_list = list(np.load(test_dir+'test_list.npy'))

params_test = {'batch_size': 1024,
          'shuffle': False,
          'num_workers': 0}

test_dataset = trafficDataset_Seq2Seq(test_list,test_dir,window_size, label = True )

# loaders
test_loader = DataLoader(test_dataset, **params_test)

model = model.to(device)

### car detection ############

loss_each =  nn.MSELoss(reduction='none')
loss_seq = []
for x_traj in tqdm(test_loader):
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
        loss = loss_each(obs_y,y_pred) #[seq_len, batch, input_dim]
        loss = loss.mean(-1).permute(1,0) # [batch, seq_len]
        loss_seq.append(loss.cpu().numpy())
loss_seq = np.concatenate(loss_seq,0)        
loss_cars = loss_seq.mean(-1)

label_all = test_dataset.label.numpy()

print(args.test_scenario_file, "anormaly rate: ", sum(label_all)/len(label_all))

# Compute ROC curve and ROC area 
roc_auc =plot_ROC_curve(label_all,loss_cars)

print(args.test_scenario_file, "ROC car detection:", roc_auc)

# Array for finding the optimal threshold
thresholds = np.linspace(min(loss_cars), max(loss_cars), 500)
fscoreOpt = search_thresholds_F1(thresholds, label_all,loss_cars)

score_path = os.path.join(result_sub_dir,  "{}-roc-car-{}-loss-{}-recon_speed-{}-recon_traj-{}".format(args.test_scenario_file,save_name,args.loss_type, args.recon_speed, args.recon_traj))
np.save(score_path, [roc_auc, fscoreOpt ])



# ### stretch detection ############

thresh_stretch= 0
dist_stretch = 0.05 # args.dist_stretch
stretch_label_all = []
stretch_loss_all = []

group_idx = test_dataset.group_idx
label_all = test_dataset.label.numpy()
x_all = test_dataset.x.numpy()

for i in tqdm(range(1,len(group_idx))):
    for stretch_min_ in np.arange(0.5, 4.5,dist_stretch):
        xs = x_all[group_idx[i-1]: group_idx[i]]
        dist = xs[:,:, -1]
        label_xs = label_all[group_idx[i-1]: group_idx[i]]
        loss_xs = loss_seq[group_idx[i-1]: group_idx[i]]
        
        # select
        stretch_min = normalize(stretch_min_, min_dist, max_dist )
        stretch_max = normalize(stretch_min_+dist_stretch, min_dist, max_dist )
        stretch_mask = (dist >= stretch_min) & (dist < stretch_max)
        if stretch_mask.sum() == 0:
            continue
            
        # loss, lable of selected
        stretch_label = label_xs[stretch_mask.any(1)].sum() > thresh_stretch
#         stretch_loss = np.nanmean(loss_xs[stretch_mask])
        stretch_loss = np.nanmax(loss_xs[stretch_mask])
        # append
        stretch_label_all.append(stretch_label)
        stretch_loss_all.append(stretch_loss)
        

# Compute ROC curve and ROC area 
roc_auc =plot_ROC_curve(stretch_label_all,stretch_loss_all)

print(args.test_scenario_file, "ROC stretch detection:", roc_auc)


# Array for finding the optimal threshold
thresholds = np.linspace(min(stretch_loss_all), max(stretch_loss_all), 500)
fscoreOpt = search_thresholds_F1(thresholds, stretch_label_all,stretch_loss_all)


score_path = os.path.join(result_sub_dir, "{}-roc-stretch-{}-loss-{}-recon_speed-{}-recon_traj-{}".format(args.test_scenario_file,save_name,args.loss_type, args.recon_speed, args.recon_traj))
np.save(score_path, [roc_auc, fscoreOpt ])

