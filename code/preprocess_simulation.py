#!/usr/bin/env python
# coding: utf-8

# preprocess the simulation dataframes into .npy files for training and testing
# segment the trajectories into short time windows, and compute the graph node feature and edge sets

import os
import argparse
import numpy as np
import os
import sys
import random
from datetime import datetime, timedelta
import matplotlib
import pandas as pd
from random import shuffle
import pickle
import time
import math
import matplotlib as mpl
from tqdm import tqdm

from util import node_feature_to_adj_list

# In[19]:
parser = argparse.ArgumentParser()
parser.add_argument('--attend_dist',type=float, default = 0.1, help = 'max Mileage to attend to as neighbors')
parser.add_argument('--attend_lane',type=float, default = 1, help = 'max lane distance to attend to neighbors')
parser.add_argument('--total_lane',type=int, default = 4, help = 'total number of lanes')
parser.add_argument('--time_window',type=int, default = 30, help = 'time window (s)')
parser.add_argument('--save_dir_name',type=str, help = 'directory to save processed data')
parser.add_argument('--edge_weight',action='store_true', default = False, help = 'calculate edge weights (based on position)')
parser.add_argument('--edge_weight_headway',action='store_true', default = False, help = 'calculate following car headway as edge weights (based on position)')
parser.add_argument('--save_meta',action='store_true', default = False, help = 'save metadata')

# specfy scenaro to store
parser.add_argument('--process_scenario', type=str, default = 'comprehensive', help = 'comprehensive, training, stalled_car, speeding, tailgating, slow' )


args = parser.parse_args()
print(args)

read_dir = '../Data/simulation_df/'
all_maxs, all_mins = pickle.load( open( read_dir+"min_max.pkl", "rb" ) )

max_gap = args.attend_dist / (all_maxs.Mileage - all_mins.Mileage) 

# add 0.1 buffer; normalize by lane
attend_lane_dist = (args.attend_lane + 0.1)/(args.total_lane-1) 
d_time = args.time_window #s

save_dir_name = args.save_dir_name


save_path = '../Data/{}/training/'.format(save_dir_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

pickle.dump( [all_maxs, all_mins], open( save_path+"min_max.p", "wb" ) )


xy_scale = (all_maxs.Mileage - all_mins.Mileage)*5280/(all_maxs.x - all_mins.x)


def process_df(df_orig): 
     # normalize to [0,1]
    df_orig.loc[:, ["x","Speed","Acceleration","Mileage" ]] =  (df_orig.loc[:, ["x","Speed","Acceleration","Mileage" ]] - all_mins) / (all_maxs - all_mins)
    
    return df_orig


# edge_attr_sim: 1/distance
# edge_attr_exp: gaussian kernel, exp(- distance ** 2 / std ** 2)


def save_sample(sample, count, times,df_label = False, save_meta = False, df_n = None):
    '''
    save a sample into graph features. 
    sample - data frame containing all trajectories in a time window
    times - time steps in the sample
    df_label - if the data frame contains label to store. True for testing sets
    save_meta - whether to save meta data
    df_n - the dataframe number to store as meta data
    '''

    # select cars present in all d_time
    count_ID = sample.groupby('ID').Time.count()
    IDs = count_ID[count_ID==d_time].index

    sample = sample[sample.ID.isin(IDs)]
    if IDs.shape[0] == 0:
        return False

    # save metadata
    if save_meta:
        np.save(save_path+"time_stamp_{}".format(count),times)
        np.save(save_path+"IDs_{}".format(count),IDs)
        np.save(save_path+"df_{}".format(count),df_n)
        return True
   
    # save graph for each time step
    for t in range(d_time):
        sample_t = sample[sample.Time == times[t]]
        sample_t = sample_t.set_index('ID')
        sample_t = sample_t.reindex(IDs)
    
        # X feature
        node_features = sample_t.loc[:, ["Lane", "Class","x","Speed",\
                             "Acceleration","Mileage" ]].values
        n_nodes = node_features.shape[0]
        # edges: cal adj martix, convert to list
        adj_list,edge_dict = node_feature_to_adj_list(node_features,
                                        max_gap,
                                        attend_lane_dist, 
                                       self_loop = True,
                                       return_weight= args.edge_weight,
                                        xy_scale = xy_scale)

        # save
        np.save(save_path+"x_{}_{}".format(count, t), node_features)
        np.save(save_path+"adj_list_{}_{}".format(count, t),adj_list)
        if args.edge_weight:
            np.save(save_path+"edge_attr_exp_{}_{}".format(count, t),edge_dict['weight'][0])
            np.save(save_path+"edge_attr_sim_{}_{}.npy".format(count, t),edge_dict['weight'][1])
        if df_label:
            # label for each car each time
            label_t = sample_t['label'].values
            np.save(save_path+"label_{}_{}".format(count, t),label_t)

    
    return True


###############  save data ###############

# In[29]:
if args.process_scenario == 'training':

    count = 0
    
    train_dfs = os.listdir(os.path.join(read_dir, 'training'))

    for df_name in train_dfs:
        df = pd.read_csv( os.path.join(read_dir, 'training', df_name))
        df = process_df(df)

        for t0 in tqdm(range(0, 299, 1)):
            # 300s (5min) in total
            times = list(range(t0, t0+d_time))
            time_mask = (df.Time >= t0) & (df.Time < t0 + d_time)
            sample = df[time_mask]
            success = save_sample(sample, count, times) 
            if success:
                count += 1


    train_list = list(np.arange(count))
    np.save(save_path+'train_list', train_list)
    
    
else:
    ### save testing
    scenario = args.process_scenario

    save_path = '../Data/{}/testing_{}/'.format(save_dir_name, scenario)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    count = 0
    test_labels = []

    df_label_all = pd.DataFrame()
    for sample_n in [1,2]:
        # two dataframe each 5min, for a total of 10min  for each scenario
        path = os.path.join(read_dir, 'testing', '{}_{}.csv'.format(scenario, sample_n))
        df = pd.read_csv(path)
        
        df = process_df(df)

        for t0 in tqdm(range(0, 299, 1)):
            times = list(range(t0, t0+d_time))
            time_mask = (df.Time >= t0) & (df.Time < t0 + d_time)
            sample = df[time_mask]
            success = save_sample(sample, count,times, df_label=True, 
                                  save_meta = args.save_meta, df_n = sample_n)
            if success:
                count += 1

        if args.save_meta:
            df_label = df.loc[:, ["ID", "Time", "label"]]
            df_label["df_n"] = sample_n
            df_label_all = pd.concat([df_label_all,df_label ])
    
    df_label_all.to_csv(save_path+'lable.csv')
    test_list = list(np.arange(count))
    np.save(save_path+'test_list', test_list)
    print('scenario',scenario,' , count: ', count)

