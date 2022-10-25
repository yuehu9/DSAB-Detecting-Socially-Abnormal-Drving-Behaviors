#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import torch
import numpy as np
import json
import argparse
import os
import sys
import random
import torch.nn as nn
import pickle
import time
import math
from tqdm import tqdm
from util_metrics import eval_recall_at_k, eval_precision_at_k, eval_average_precision,eval_roc_auc
from util import cal_loss_car, cal_loss, normalize, scale_back

from myDataset import trafficGraphDataset as trafficDataset
from dataLoader import myDataLoader
from model import DSAB


parser = argparse.ArgumentParser()
# testing parameter
parser.add_argument('--train_file_name',type=str, default = 'simulation_processed', help='parent dir for training data file')
parser.add_argument('--experiment_name',type=str, default = 'DSAB', help='unique experiment name, used to read results from dir')
parser.add_argument('--test_scenario_file',type=str, default = 'testing_comprehensive', help = 'test scenario: testing_comprehensive, testing_stalled_car, testing_slow, testing_speeding, testing_tailgating')
parser.add_argument('--dist_stretch',type=float, default = 0.15, help= 'highway stretch distance in miles for anomaly detecting')
test_args = parser.parse_args()
print(test_args)

# read training args
result_sub_dir = "../expriments/{}_{}/".format(test_args.train_file_name, test_args.experiment_name)
save_name =  "DSAB"
arg_path = os.path.join(result_sub_dir, "arg-{}.txt".format(save_name))
args = parser.parse_args()
with open(arg_path, 'r') as f:
    args.__dict__ = json.load(f)
print(args)

# set seeds
np.random.seed(seed=args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# read from data dir
file_name = args.train_file_name 
train_dir = '../Data/{}/training/'.format(file_name)
all_maxs, all_mins = pickle.load( open( train_dir+"min_max.p", "rb" ) )
max_dist, min_dist  = all_maxs.Mileage, all_mins.Mileage 

window_size = args.window_size
sample_time = args.sample_time

## test set
file_name_test = args.train_file_name
test_dir =  '../Data/{}/{}/'.format(file_name_test, test_args.test_scenario_file)
test_list = list(np.load(test_dir+'test_list.npy'))
root = test_dir
test_dataset = trafficDataset(root, test_list, 
                              label = True,
                              source_dir = test_dir,
                              window_size = window_size,
                          sample_time = sample_time,
                              mask = args.padding,
                             edge_attr = args.data_edge_attr)


############ Model ############ 

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

rate_lane = args.loss_rate_lane
rate_acc =  args.loss_rate_acc
rate_speed = args.loss_rate_speed

############ Evaluate results ############ 

model_path = os.path.join(result_sub_dir, "model-{}.pt".format(save_name))
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

epoch = checkpoint['epoch']
print("best model is at epoch", epoch)
model.eval()
model = model.to(device)

### abnormal car detection

start = time.time()
loss_all = []
label_all = []
all_xs = []
all_index = []

for i, graph_data in enumerate(test_dataset):

    graph_data = [graph_i.to(device) for graph_i in graph_data]
    with torch.no_grad():  
        output = model(graph_data) #[n_node, time, feature]

    graph_x = [graph_i.x for graph_i in graph_data]
    graph_x = torch.stack(graph_x, dim = 1) #[n_node, time, feature]
    # to cpu
    graph_x = graph_x.to('cpu')
    output = output.cpu()

    loss = cal_loss_car(output, graph_x, rate_lane, rate_speed, rate_acc,time_agg_loss=True)
    index = np.arange(len(graph_x))
    #label
    car_label = [g.label.cpu() for g in graph_data]
    car_label = np.stack(car_label,1)
    car_label = car_label.any(1)

    loss_all.append(loss)
    label_all.append(car_label)
    all_xs.append(graph_x)
    all_index.append(index)
    
label_all = np.concatenate(label_all)
loss_all = np.concatenate(loss_all)
all_xs = np.concatenate(all_xs)
all_index = [(i,val) for i, l_val in enumerate(all_index) for val in l_val ]
all_index = np.array(all_index)

## eval metrics
dict_eval = {}
dict_eval['len_smaple'] = len(loss_all)
dict_eval['num_anom'] = sum(label_all)
dict_eval['anom_rate'] = sum(label_all)/len(loss_all)

dict_eval['recall_at_k'] = eval_recall_at_k(label_all, loss_all, sum(label_all))
k_list = [50, 100,200,500,700,1000]
precision_at_k = []
for k in k_list:
    prec = eval_precision_at_k(label_all, loss_all,k)
    precision_at_k.append(prec)
dict_eval['precision_at_k'] = precision_at_k
dict_eval['k_list'] = k_list
dict_eval['average_precision'] = eval_average_precision(label_all, loss_all)
dict_eval['roc_auc'] = eval_roc_auc(label_all,loss_all)


score_path = os.path.join(result_sub_dir, "eval-{}-car-{}.pkl".format(test_args.test_scenario_file,save_name))  
with open(score_path, 'wb') as f:
    pickle.dump(dict_eval, f)

print("abnormal car detection for {}: ".format(test_args.test_scenario_file), dict_eval)


#### abnormal scene detection
start = time.time()
loss_all = []
label_all = []
all_xs_l = []
all_index = []

time_agg_loss = False

for i, graph_data in enumerate(tqdm(test_dataset)):
    graph_data = [graph_i.to(device) for graph_i in graph_data]
    with torch.no_grad():  
        output = model(graph_data) #[node, time, feat]

    graph_x = [graph_i.x for graph_i in graph_data]
    graph_x = torch.stack(graph_x, dim = 1) # [node, time, feat]
    # to cpu
    graph_x = graph_x.to('cpu')
    output = output.cpu()

    loss = cal_loss_car(output, graph_x, rate_lane, rate_speed, rate_acc,time_agg_loss)
    index = np.arange(len(graph_x))
    #label
    car_label = [g.label.cpu() for g in graph_data]
    car_label = np.stack(car_label,1)

    loss_all.append(loss)
    label_all.append(car_label)
    all_xs_l.append(graph_x)
    all_index.append(index)

dist_stretch = test_args.dist_stretch

stretch_label_all = []
stretch_loss_all = []
for i in range(len(loss_all)):
    for stretch_min_ in np.arange(0.5, 4.5,dist_stretch):
        # omit the first and last half mile because of the lack of complete context at the beginning and end of the stretch
        xs = all_xs_l[i]
        dist = xs[:,:, -1]
        label_xs = label_all[i]
        loss_xs = loss_all[i]
        # select
        stretch_min = normalize(stretch_min_, min_dist, max_dist )
        stretch_max = normalize(stretch_min_+dist_stretch, min_dist, max_dist )
        stretch_mask = (dist >= stretch_min) & (dist < stretch_max)
        if stretch_mask.sum() == 0:
            continue
        # loss, lable of selected. The scene is abnormal if existst at one abnormal behaving hcars
        stretch_label= label_xs.any(1)
        stretch_label = stretch_label[stretch_mask.any(1)].sum() > 0 
        stretch_loss = np.nanmax(loss_xs[stretch_mask])
        # append
        stretch_label_all.append(stretch_label)
        stretch_loss_all.append(stretch_loss)

print("anormaly rate: ", sum(stretch_label_all)/len(stretch_label_all))

## eval metrics
dict_eval = {}
dict_eval['len_smaple'] = len(stretch_loss_all)
dict_eval['num_anom'] = sum(stretch_label_all)
dict_eval['anom_rate'] = sum(stretch_label_all)/len(stretch_loss_all)

dict_eval['recall_at_k'] = eval_recall_at_k(stretch_label_all, stretch_loss_all, sum(stretch_label_all))
k_list = [50, 100,200,500,700,1000]
precision_at_k = []
for k in k_list:
    prec = eval_precision_at_k(stretch_label_all, stretch_loss_all,k)
    precision_at_k.append(prec)
dict_eval['precision_at_k'] = precision_at_k
dict_eval['k_list'] = k_list
dict_eval['average_precision'] = eval_average_precision(stretch_label_all, stretch_loss_all)
dict_eval['roc_auc'] = eval_roc_auc(stretch_label_all,stretch_loss_all)

score_path = os.path.join(result_sub_dir, "eval-{}-scene-{}.pkl".format(test_args.test_scenario_file,save_name))
with open(score_path, 'wb') as f:
    pickle.dump(dict_eval, f)

print("abnormal scene detection for {}: ".format(test_args.test_scenario_file), dict_eval)    

print('done testing')
       
    