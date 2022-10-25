from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import math
import sys
import torch
import numpy as np
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch.nn as nn
from torch_geometric.utils import coalesce 


def merge_edges_over_time(graph_data):
    '''
    merge the input edge sets
    '''
    edge_l = [dt.edge_index for dt in graph_data]
    edge_all = torch.concat(edge_l, -1)
    return coalesce(edge_all)


def normalize(val, v_min, v_max):
    '''
    normalize to 0-1
    '''
    return (val-v_min)/(v_max - v_min)


def scale_back(val, v_min, v_max):
    '''
    0-1 back to orignal scale
    '''
    return val*(v_max - v_min) + v_min
    
    
def node_feature_to_adj_list(node_features,max_gap,attend_lane_dist, self_loop = True,  return_weight=False, xy_scale = 1):
    '''
    calculate adjacency list from node features (mileage and lane)
    based on input. Mileage and lane might be normalized
    e.g, for normalized 4 lane, neighboring lane delta ~= 0.33
    node_features -  ["Lane", "Class","x","Speed", "Acceleration","Mileage" ].
    max_gap - max longitudinal distance in miles to attend to.
    attend_lane_dist - max lane distance to attend to.
    return_weight - return edge weight as function of distance
    xy_scale - scaling factor between x, y positions e.g. because of units and normalization.
                             
    '''
    n_nodes = node_features.shape[0]
    lane_all = node_features[:, 2]
    dist_all = node_features[:, -1]
    speed_all = node_features[:, 3]
    lane_adj = np.expand_dims(lane_all,1) - np.expand_dims(lane_all,0)
    dist_adj = np.expand_dims(dist_all,1) - np.expand_dims(dist_all,0)
    # dist_adj[i,j] = dist_i - dist_j, positive means i in front of j
    dist_adj_bool = abs(dist_adj) <= max_gap
    lane_adj_bool = abs(lane_adj) <= attend_lane_dist 
    adj_matrix = lane_adj_bool & dist_adj_bool
    if not self_loop:
        adj_matrix &= ~np.eye(n_nodes, dtype=bool)
    adj_list = [[i,j] for i in range(n_nodes) for j in range(n_nodes) if adj_matrix[i][j] ]
    adj_list = np.array(adj_list)
    edge_dict = {}
    
    if return_weight:
        d_norm = np.sqrt((dist_adj*xy_scale) **2 + lane_adj**2) # l2 norm
        adj_exp = np.exp(-d_norm)
        adj_sim = 1/(d_norm + np.eye(d_norm.shape[0])) # eye to avoid devide by 0

        attr_exp = adj_exp[adj_matrix]
        attr_sim = adj_sim[adj_matrix]
        edge_dict['weight'] = (attr_exp, attr_sim)
    
    return adj_list.T, edge_dict
    

###### loss #####


criterion_CEL = nn.CrossEntropyLoss()
criterion_NLL = nn.GaussianNLLLoss()
criterion_MSE = nn.MSELoss()


def cal_loss(model_output, graph_x, rate_lane, rate_speed=0, rate_acc = 0, mask = [],  n_lane = 4):
    '''
    calculate loss for each sample graph, used in training
    model_output, graph_x: [n_node, time, feature], where model_output feature: [mu_dist, sig_dist, p_lane1,..,p_lanen, mu_v, sig_v, mu_a, sig_a]
    rate_lane - loss rate for lane 
    output - a single loss value 
    '''
    # pred
    dist_pred = model_output[:,:,:2]
    lane_pred = model_output[:,:, 2:2+n_lane]
    # true
    lane = graph_x[:,:,[0]].long()
    dist = graph_x[:,:,[-1]]
    # reshape lane
    lane = lane.view(-1) # [node* time]
    lane_pred = lane_pred.view(-1, lane_pred.shape[-1])
    # dist gaussian loss   
    mean_dist = dist_pred[:,:, [0]]
    var_dist = torch.exp(dist_pred[:,:,[1]])

    if len(mask) > 0: 
        # use mask
        loss_nll = criterion_NLL(mean_dist[mask] , dist[mask] , var_dist[mask])
        loss_cel = criterion_CEL(lane_pred[mask.flatten()], lane[mask.flatten()])
    else:
        loss_nll = criterion_NLL(mean_dist, dist, var_dist)
        loss_cel = criterion_CEL(lane_pred, lane)
    loss = loss_nll + rate_lane*loss_cel
    if rate_speed > 0:
        speed_pred = model_output[:,:, 2+n_lane:4+n_lane]
        speed = graph_x[:,:,[3]]
        mean_sp = speed_pred[:,:, [0]]
        var_sp = torch.exp(speed_pred[:,:,[1]])
        if len(mask) > 0:
            loss_speed = criterion_NLL(mean_sp[mask], speed[mask], var_sp[mask])
        else: 
            loss_speed = criterion_NLL(mean_sp, speed, var_sp)
        loss += rate_speed*loss_speed
    if rate_acc > 0:
        acc = graph_x[:,:,[4]]
        acc_pred = model_output[:,:, 4+n_lane:6+n_lane]
        mean_acc = acc_pred[:,:, [0]]
        var_acc = torch.exp(acc_pred[:,:,[1]])
        if len(mask) > 0:
            loss_acc = criterion_NLL(mean_acc[mask], acc[mask], var_acc[mask])
        else:
            loss_acc = criterion_NLL(mean_acc, acc, var_acc)
        loss += rate_acc*loss_acc
    return loss


NLL_loss = nn.GaussianNLLLoss(reduction = 'none')
CE_loss = nn.CrossEntropyLoss(reduction = 'none')

def cal_loss_car(model_output, graph_x, rate_lane, rate_speed=0, rate_acc = 0,time_agg_loss = True, mask = [], n_lane = 4):
    '''
    calculate loss for each sample graph, used in evaluation
    model_output, graph_x: [n_node, time, feature], where model_output feature: [mu_dist, sig_dist, p_lane1,..,p_lanen, mu_v, sig_v, mu_a, sig_a]
    output - [n_car, time] if time_agg_loss = False; else [n_car,]
    '''
    # pred
    dist_pred = model_output[:,:,:2]
    lane_pred = model_output[:,:, 2:2+n_lane]
    # true
    lane = graph_x[:,:,[0]].long()
    dist = graph_x[:,:,[-1]]
    # reshape lane
    lane = lane.view(-1) # [node* time]
    lane_pred = lane_pred.view(-1, lane_pred.shape[-1])
    # dist gaussian loss   
    mean_dist = dist_pred[:,:, [0]]
    var_dist = torch.exp(dist_pred[:,:,[1]])

    n_car = len(graph_x)
    loss_nll = NLL_loss(mean_dist, dist, var_dist).squeeze(-1) # [n_car, time]
    loss_cel = CE_loss(lane_pred, lane).view(n_car,-1) # [n_car, time]  
    loss = loss_nll + rate_lane * loss_cel

    # speed
    if rate_speed > 0:
        speed = graph_x[:,:,[3]]
        speed_pred = model_output[:,:, 2+n_lane:4+n_lane]
        mean_s = speed_pred[:,:, [0]]
        var_s = torch.exp(speed_pred[:,:,[1]])
        loss_speed = NLL_loss(mean_s, speed, var_s).squeeze(-1) # [n_car, time]
        loss += rate_speed*loss_speed

    # acc
    if rate_acc > 0:
        acc = graph_x[:,:,[4]]
        acc_pred = model_output[:,:, 4+n_lane:6+n_lane]
        mean_acc = acc_pred[:,:, [0]]
        var_acc = torch.exp(acc_pred[:,:,[1]])
        loss_acc = NLL_loss(mean_acc, acc, var_acc).squeeze(-1) # [n_car, time]
        loss += rate_acc*loss_acc
        
    if len(mask) > 0:
        loss[~mask] = np.nan

    if time_agg_loss:
        # loss per car across all time steps
        loss = loss.nanmean(1)
    return loss

    
    
###### functions for bivariate normal trajectory modeling ######
    
def pred_to_distribution(traj_pred):
    '''
    from model_output distribution to torch multivariate Normal distribution
    traj_pred - [n_node, time_step, dim_feat2], where dim_feat2 = 5 -> [mu_x, mu_y, std_x, std_y, corr]
    '''
#     traj_pred = traj_pred.permute(0,2,1) #[node, feat,time] to [node, time, feat]
    sx = torch.exp(traj_pred[:,:,2]) #sx
    sy = torch.exp(traj_pred[:,:,3]) #sy
    corr = torch.tanh(traj_pred[:,:,4]) #corr

    cov = torch.zeros(traj_pred.shape[0],traj_pred.shape[1],2,2).to(traj_pred.device)
    cov[:,:,0,0]= sx*sx
    cov[:,:,0,1]= corr*sx*sy
    cov[:,:,1,0]= corr*sx*sy
    cov[:,:,1,1]= sy*sy
    mean = traj_pred[:,:,0:2]
    
    return MultivariateNormal(mean,cov)


def bivariate_loss(traj, traj_pred):
    '''
    traj - [n_node, time_step, dim_feat1], where dim_feat = 2 -> [x,y]
    traj_pred - [n_node, time_step, dim_feat2], where dim_feat2 = 5 -> [mu_x, mu_y, std_x, std_y, corr]
    '''
    dist = pred_to_distribution(traj_pred)
    loss = -dist.log_prob(traj).mean().mean()
    return loss

    