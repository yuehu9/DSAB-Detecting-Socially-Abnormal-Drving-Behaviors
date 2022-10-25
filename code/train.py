#!/usr/bin/env python
# coding: utf-8

import argparse
import os

parser = argparse.ArgumentParser()
# data param
parser.add_argument('--window_size',type=int, default = 15, help = 'time window size (s)')
parser.add_argument('--sample_time',type=int, default = 1, help = 'sample frequency, e.g. 2 means 1 sample every 2 seconds')
parser.add_argument('--padding',action='store_true', default = False, help= 'whether input vehicle features padded. Padded parts will be masked for loss calculation')
parser.add_argument('--data_edge_attr',type=str, default = None, help = 'edge attribute type to be loaded. Not used for final model')
parser.add_argument('--train_file_name',type=str, default = 'simulation_processed', help='parent dir for training data file')
    
# model param
parser.add_argument('--edge_dropout',type=float, default = 0., help = 'edge_dropout for graph layers')
parser.add_argument('--hidden_dim',type=int, default = 5, help = 'dimension of recurrent hidden states')
parser.add_argument('--encode_dim',type=int, default = -1, help = 'dimension of final encode embedding vector. If -1, set to the same as hidden_dim')
parser.add_argument('--graph_layer',type=str, default =  'GATv2Conv', help = 'GATConv, GATv2Conv,TransformerConv, SAGEConv')
parser.add_argument('--graph_aggr',type=str, default =  'mean', help = 'graph aggregation scheme: mean, add, max')
parser.add_argument('--model_self_loop', action='store_true', default = False, help= 'add self loop for GAT and GATv2 graph layers')
parser.add_argument('--model_edge_attr', action='store_true', default = False, help = 'use pre-defined edge weights in the model')
parser.add_argument('--head_concat', action='store_true', default = False, help = 'concatenate attention heads (and pass through a fc layer) instead of averaging')
parser.add_argument('--n_head', type=int, default = 3, help= '# attention head')
parser.add_argument('--no_decode_graph', action='store_true', default = False, help = 'whether to use graph information at decoding')


# training param
parser.add_argument('--learning_rate',type=float, default = 5e-2)
parser.add_argument('--lr_decay_step_size',type=int, default = 50, help = 'decrease lerarning rate by half every lr_decay_step_size')
parser.add_argument('--clip_grad',type=float, default =1)
parser.add_argument('--loss_rate_lane',type=float, default = 2, help = 'weighting paramter in loss function for the lane loss term')
parser.add_argument('--loss_rate_acc',type=float, default = 2, help = 'weighting paramter in loss function for the acceleration loss term')
parser.add_argument('--loss_rate_speed',type=float, default = 1, help = 'weighting paramter in loss function for the speed loss term')
parser.add_argument('--GPU',type=str, default = "1")
parser.add_argument('--epochs',type=int, default = 500)
parser.add_argument('--experiment_name',type=str, default = '', help='unique experiment name, used to save results to dir')
parser.add_argument('--verbal',action='store_true', default = False, help='print training results')
parser.add_argument('--batch_size',type=int, default = 64)
parser.add_argument('--highD_data',action='store_true', default = False, help = 'training using highD data. Otherwise use simulation data')
parser.add_argument('--seed',type=int, default = 1)

args = parser.parse_args()

print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

import torch
import numpy as np
from sklearn.model_selection import train_test_split
import json
import sys
import random
import torch.nn as nn
import matplotlib.pyplot as plt 
from random import shuffle
import pickle
import time
import math
from tqdm import tqdm
from util import normalize, scale_back
from util import cal_loss

if args.highD_data:
    from myDataset_padding import trafficGraphDataset as trafficDataset
else:
    from myDataset import trafficGraphDataset as trafficDataset
from dataLoader import myDataLoader
from model import DSAB
from util import pred_to_distribution

# set seeds
np.random.seed(seed=args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.cuda.manual_seed(args.seed)
# # diterministic
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True

result_sub_dir = "../expriments/{}_{}/".format(args.train_file_name, args.experiment_name)
if not os.path.exists(result_sub_dir):
    os.makedirs(result_sub_dir)


# data dir
file_name = args.train_file_name 
train_dir = '../Data/{}/training/'.format(file_name)

save_name =  "DSAB"


# save arg
arg_path = os.path.join(result_sub_dir, "arg-{}.txt".format(save_name))
with open(arg_path, 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# Load data
train_list = list(np.load(train_dir+'train_list.npy'))
partition_train, partition_val= train_test_split(train_list, test_size=0.1, random_state=42)


window_size = args.window_size
sample_time = args.sample_time

params_train = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 0}

params_val = {'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 0}

root = train_dir

# data loaders:
train_dataset = trafficDataset(root, partition_train,
                           source_dir = train_dir,
                          window_size = window_size,
                          sample_time = sample_time,
                              mask = args.padding,
                              edge_attr = args.data_edge_attr)
val_dataset = trafficDataset(root, partition_val,
                         source_dir = train_dir, 
                          window_size = window_size,
                          sample_time = sample_time,
                            mask = args.padding,
                          edge_attr = args.data_edge_attr)


train_loader = myDataLoader(train_dataset, **params_train)
val_loader = myDataLoader(val_dataset,**params_val )



######## Model  ########

hidden_dim =args.hidden_dim
encode_dim = hidden_dim if args.encode_dim == -1 else args.encode_dim
dropout = 0.2
graph_layer = args.graph_layer
edge_dropout = args.edge_dropout
decode_steps = len(range(0,window_size, sample_time))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.highD_data:
    n_lane = 3
else:
    n_lane =4

model = DSAB(input_feat_dim = 4,
                hidden_dim = hidden_dim,
                encode_dim = encode_dim,
                n_head = args.n_head,
                n_lane = n_lane,
                graph_layer = graph_layer,
                dropout= dropout, 
                decode_steps = decode_steps,
               edge_dropout = edge_dropout,
               self_loop = args.model_self_loop,
               use_edge_attr = args.model_edge_attr,
               head_concat = args.head_concat,
               graph_aggr = args.graph_aggr,
               decode_graph = ~args.no_decode_graph)


model.float()
# print(model)


lr = args.learning_rate
# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

rate_lane = args.loss_rate_lane
rate_acc =  args.loss_rate_acc
rate_speed = args.loss_rate_speed


def train(epoch, train_loader ,val_loader, best_val, best_epoch, verbal = False, padding = False):
    model.train()
    train_loss = 0
    loss_train = []
    loss_val = []
    for i, graph_data in enumerate(train_loader):
        graph_data = [graph_i.to(device) for graph_i in graph_data]
        optimizer.zero_grad()
        output = model(graph_data) #[node, time, feature_dim]
        graph_x = [graph_i.x for graph_i in graph_data]
        graph_x = torch.stack(graph_x, dim = 1) # [node, time, feature_dim]

        if padding:
            # padding mask
            graph_mask = [graph_i.mask for graph_i in graph_data]
            graph_mask = torch.stack(graph_mask, dim = 1) # [node, time]
        else:
            graph_mask = []
        loss = cal_loss(output, graph_x, rate_lane, rate_speed, rate_acc, graph_mask, n_lane)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        optimizer.step()
        loss_train.append(loss.item())
    for graph_val in val_loader:
        # evaluation
        model.eval()
        graph_val = [graph_i.to(device) for graph_i in graph_val]
        with torch.no_grad():
            output = model(graph_val) #[node, time, feature_dim]
        graph_x = [graph_i.x for graph_i in graph_val]
        graph_x = torch.stack(graph_x, dim = 1) # [node, time, feature_dim]

        if padding:
            # padding mask
            graph_mask = [graph_i.mask for graph_i in graph_val]
            graph_mask = torch.stack(graph_mask, dim = 1) # [node, time]
        else:
            graph_mask = []

        loss = cal_loss(output, graph_x, rate_lane, rate_speed, rate_acc, graph_mask, n_lane)
       
        loss_val.append(loss.item())
    
    loss_train = sum(loss_train) / len(loss_train)
    loss_val = sum(loss_val) / len(loss_val)
 
    # print results
    if verbal and epoch % 3 == 0:
        print('Train Epoch: {}  loss: {:e}  val_loss: {:e}'.format(
            epoch, loss_train, loss_val ))
    
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


################ Train ################## 



loss_track = []
val_track = []
model_path = os.path.join(result_sub_dir, "model-{}.pt".format(save_name))

model = model.to(device)
n_epochs = args.epochs
start = time.time()
best_epoch = 0
best_val = float('inf') # for early stopping
lr_decay_step_size = args.lr_decay_step_size

for epoch in tqdm(range(0, n_epochs+1)):
    train_loss, val_loss, best_val,best_epoch = train(epoch, train_loader, val_loader, best_val, best_epoch, verbal = args.verbal, padding= args.padding)
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
torch.cuda.empty_cache()


# plot learning curve
plt.plot(np.array(loss_track), label = 'training')
plt.plot(np.array(val_track), label = 'validaiton')
plt.title("loss")
plt.xlabel("# epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.title("{}\n{}".format(result_sub_dir, save_name))
fig_path = os.path.join(result_sub_dir, "curve-{}.pdf".format(save_name))
plt.savefig(fig_path)

#save val loss
val_path = os.path.join(result_sub_dir, "val_loss-{}.npy".format(save_name))

np.save(val_path, [val_track[best_epoch],val_track[-1]])
print("validation loss: ", val_track[best_epoch], val_track[-1])

