import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from GRNN_Cell import GRNN_Cell
from util import merge_edges_over_time
        
class DSAB(torch.nn.Module): 
    def __init__(self, input_feat_dim,hidden_dim, encode_dim, n_head = 3, embed_dim = 5, graph_layer = 'GATv2Conv', decode_steps = 10, use_edge_attr = False, use_lane = True, n_lane = 4, dropout= 0, edge_dropout= 0, self_loop = False, head_concat = False, graph_aggr = 'mean', decode_graph = True):
        ''' 
        input_feat_dim - dimension of input numerical feature. (categorical features counted separately)
        hidden_dim - dimension of recurrent hidden states
        encode_dim - dimension of final encode embedding vector
        n_head - number of attention heads
        embed_dim - embedding dimension for lane ids
        graph_layer - GNN used recurrently. Choice: GATv2Conv, GATConv, TransformerConv, SAGEConv
        decode_steps - number of time steps to decode, should be same as input sequence length
        use_edge_attr: use pre-defined edge weights
        use_lane - if use vehicle lane id as attribute
        n_lane - number of lanes
        edge_dropout - edge_dropout for graph layers
        self_loop - add self loop in GNN
        head_concat - concatenate attention heads (and pass through a fc layer) instead of averaging
        graph_aggr - neighbor aggregation function in graph layers
        decode_graph - whether to use graph information at decoding
        '''
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_feat_dim = input_feat_dim
        self.encode_dim = encode_dim
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.use_edge_attr = use_edge_attr
        self.use_lane = use_lane
        self.n_lane = n_lane
        self.edge_dropout = edge_dropout
        self.decode_graph = decode_graph
        
        ### encoder 
        enncode_input_dim = input_feat_dim
        if use_lane:
            self.lane_embedding = nn.Embedding(n_lane, embed_dim)
            enncode_input_dim += embed_dim
        self.GRNN_Cell_encode = GRNN_Cell(in_channels = enncode_input_dim,out_channels = hidden_dim,K = n_head, graph_layer=graph_layer,dropout = self.edge_dropout, self_loop = self_loop, head_concat = head_concat, aggr = graph_aggr)
        self.fc_encode = nn.Linear(hidden_dim, encode_dim)
        
        ### decoder
        self.decode_steps = decode_steps
        self.output_dim_dist = 6 # (mean, logvar) of distance, speed, acceleration
        self.output_dim_dist += n_lane # predict lane
        self.fc_decode = torch.nn.Linear(encode_dim, hidden_dim)
        self.fc_decode_output = torch.nn.Linear(hidden_dim, self.output_dim_dist)
        self.GRNN_Cell_decode = GRNN_Cell(in_channels = self.output_dim_dist,out_channels = hidden_dim,K = n_head, graph_layer=graph_layer, dropout = self.edge_dropout, self_loop = self_loop, head_concat = head_concat, aggr = graph_aggr)


    def encode(self, graphs):
        '''
        graphs - list of [graph_0, ..., graph_(time_steps)]
        graph_i.x - dimension: [n_nodes, input_feat_dim], where input_feat_dim = 6 corresponding to [Lane, class, x, speed, acceleration, mileage (distance)] in our implementation
        grahph_i.edge - [2, n_dedges]
        output - encode_state, [n_nodes, encode_dim]
        '''
        hidden_state = None
        input_seq_len = self.decode_steps
        for graph_i in graphs[:input_seq_len]:
            x_nodes = graph_i.x[:,2:]  # get numerical features
            # lane embeddings
            x_lane = self.lane_embedding(graph_i.x[:,0].int())
            x_nodes = torch.cat((x_nodes,x_lane ), -1)
            # update
            edge_attr = graph_i.edge_attr if self.use_edge_attr else None
            hidden_state = self.GRNN_Cell_encode(x_nodes, graph_i.edge_index, hidden_state, edge_attr)
           
        output = self.fc_encode(hidden_state)
        return output

    def graph_x_to_decode_input(self, x):
        '''
        convert the observation of last time step to the initial input for decoder, which is a set of distribution parameters
        '''
        # Gaussian for dist,speed, acceleration.  std is 0, mean is the observation
        n_nodes = x.shape[0]
        decoder_input = torch.zeros(n_nodes, self.output_dim_dist).to(x.device)
        decoder_input[:,0] = x[:,-1] # dist 
        decoder_input[:, (2+self.n_lane)] = x[:,3] # speed 
        decoder_input[:, (4+self.n_lane)] = x[:,4] # acceleration

        # probabilistic distribution for each lane. 1 if car in the lane, 0 otherwise
        lane_rows = list(range(n_nodes))
        lane_cols = [c+2 for c in x[:,0].long()]
        decoder_input[lane_rows, lane_cols] = 1 
        return decoder_input

    def decode(self, encode_state, decoder_input, edge_index):
        '''
        encode_state - [n_nodes, encode_dim]
        decoder_input - [n_nodes, output_dim_dist]
        grahph_i.edge - [2, n_dedges]
        '''
        n_nodes = encode_state.shape[0]
        hidden_state = self.fc_decode(encode_state)
        hidden_state = self.dropout_layer(hidden_state)
        hidden_state = F.relu(hidden_state)

        # propagate 
        outputs = []
        for i in range(self.decode_steps):
            if self.use_edge_attr:
                hidden_state = self.GRNN_Cell_decode(decoder_input, edge_index,hidden_state,graph_i.edge_attr)
            else:
                hidden_state = self.GRNN_Cell_decode(decoder_input, edge_index,hidden_state)
            output = self.fc_decode_output(hidden_state)
            decoder_input = output
            outputs.append(output)
        outputs = torch.stack(outputs[::-1], dim = 1) # decode backwards in time
        return outputs


    def forward(self, graphs ):
        init_x = graphs[0].x
        encode_state = self.encode(graphs)
        
        # use last state as decoder input
        last_graph = graphs[-1]
        decoder_input = self.graph_x_to_decode_input(last_graph.x)
        if self.decode_graph:
            # consider graph connections in decoding
            edge_index = merge_edges_over_time(graphs)
        else:
            edge_index = torch.zeros(2,0).type(torch.LongTensor).to(encode_state.device)

        outputs = self.decode(encode_state, decoder_input,edge_index)
        
        return outputs
