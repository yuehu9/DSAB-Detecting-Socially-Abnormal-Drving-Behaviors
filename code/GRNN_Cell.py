import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, TransformerConv, GATConv, GATv2Conv
from torch_geometric.utils import dropout_adj
from torch.nn import ReLU


class GRNN_Cell(torch.nn.Module):
    '''
    adapted from https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/recurrent/dcrnn.html#DCRNN
    '''
    def __init__(self, in_channels: int, out_channels: int, K: int = 1, bias: bool = True, graph_layer = 'ConGAE', dropout = 0, self_loop = True, head_concat = False, aggr = 'mean'):
        '''
        in_channels: dimension of input feature
        out_channel: dimension of output (= dimension of hidden state)
        K : number of heads
        head_concat: concatenate the attention head and go through fc layer (v.s. take the avg of the attention head)
        graph_layer: SAGEConv, TransformerConv, GATConv,  GATv2Conv
        '''
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.bias = bias
        self.graph_layer = graph_layer
        self.dropout = dropout
        self.self_loop = self_loop
        self.head_concat = head_concat
        self.aggr = aggr

        self._create_parameters_and_layers()

    def _create_parameters_and_layers(self):
        if self.head_concat and self.graph_layer in ['GATConv', 'GATv2Conv','TransformerConv']:
            self.fc_z = nn.Linear(self.out_channels*self.K,self.out_channels)
            self.fc_r = nn.Linear(self.out_channels*self.K,self.out_channels)
            self.fc_h = nn.Linear(self.out_channels*self.K,self.out_channels)
        if self.graph_layer == 'GATConv':
            self.conv_x_z = GATConv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, heads = self.K,concat=self.head_concat, dropout = self.dropout,add_self_loops = self.self_loop)

            self.conv_x_r = GATConv(in_channels=self.in_channels + self.out_channels,
                out_channels=self.out_channels,  heads = self.K,concat=self.head_concat, dropout = self.dropout,add_self_loops = self.self_loop)

            self.conv_x_h = GATConv(in_channels=self.in_channels + self.out_channels,
                out_channels=self.out_channels, heads = self.K,concat=self.head_concat, dropout = self.dropout,add_self_loops = self.self_loop)
            
        elif self.graph_layer == 'GATv2Conv':
            self.conv_x_z = GATv2Conv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, heads = self.K,concat=self.head_concat, dropout = self.dropout,add_self_loops = self.self_loop)

            self.conv_x_r = GATv2Conv(in_channels=self.in_channels + self.out_channels,
                out_channels=self.out_channels,  heads = self.K,concat=self.head_concat, dropout = self.dropout,add_self_loops = self.self_loop)

            self.conv_x_h = GATv2Conv(in_channels=self.in_channels + self.out_channels,
                out_channels=self.out_channels, heads = self.K,concat=self.head_concat, dropout = self.dropout,add_self_loops = self.self_loop)
            
        elif self.graph_layer == 'TransformerConv':
            self.conv_x_z = TransformerConv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, heads = self.K,concat=self.head_concat,  dropout = self.dropout)

            self.conv_x_r = TransformerConv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, heads = self.K,concat=self.head_concat, dropout = self.dropout)

            self.conv_x_h = TransformerConv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, concat=self.head_concat, heads = self.K,  dropout = self.dropout)
     
        elif self.graph_layer == 'SAGEConv':
            self.conv_x_z = GraphConv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, aggr = self.aggr)
            self.conv_x_h = GraphConv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, aggr = self.aggr)
            self.conv_x_r = GraphConv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, aggr = self.aggr)
        else:
            raise NotImplementedError("graph layer not implemented")


    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, H,edge_weight):
        Z = torch.cat([X, H], dim=1)
        Z = self.conv_x_z(Z, edge_index,edge_weight )
        if self.head_concat:
            Z =  F.relu(Z)
            Z = self.fc_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, H,edge_weight):
        R = torch.cat([X, H], dim=1)
        R = self.conv_x_r(R, edge_index,edge_weight)
        if self.head_concat:
            R =  F.relu(R)
            R = self.fc_r(R)
        R = torch.sigmoid(R)
        
        return R

    def _calculate_candidate_state(self, X, edge_index, H, R,edge_weight):
        H_tilde = torch.cat([X, H * R], dim=1)
        H_tilde = self.conv_x_h(H_tilde, edge_index,edge_weight)
        if self.head_concat:
            H_tilde =  F.relu(H_tilde)
            H_tilde = self.fc_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(self, X, edge_index, H = None, edge_weight = None) :
        r"""Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **edge_index** (PyTorch Long Tensor) - Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional) - Edge weight vector.
            * **H** (PyTorch Float Tensor, optional) - Hidden state matrix for all nodes.
        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, H, edge_weight)
        R = self._calculate_reset_gate(X, edge_index,  H, edge_weight)
        H_tilde = self._calculate_candidate_state(X, edge_index, H, R, edge_weight)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H
    
    
    def check_attention(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        H: torch.FloatTensor = None,
        edge_weight = None) :
        """
        check_attention scores for each edge
        """
        H = self._set_hidden_state(X, H)
        XH = torch.cat([X, H], dim=1)
        _, attention_z = self.conv_x_z(XH, edge_index,edge_weight,return_attention_weights = True )
        _, attention_r = self.conv_x_r(XH, edge_index,edge_weight,return_attention_weights = True)
        
        Z = self._calculate_update_gate(X, edge_index, H, edge_weight)
        R = self._calculate_reset_gate(X, edge_index,  H, edge_weight)
        H_tilde = torch.cat([X, H * R], dim=1)
        _, attention_h = self.conv_x_h(H_tilde, edge_index,edge_weight,return_attention_weights = True)
        
        return attention_z, attention_r, attention_h

