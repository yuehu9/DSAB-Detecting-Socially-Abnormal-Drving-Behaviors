import torch
from torch.utils import data
import numpy as np
from torch_geometric.data import InMemoryDataset, Dataset, Data, HeteroData
import pickle

class trafficGraphDataset(InMemoryDataset):
    def __init__(self, root, data_list, source_dir = '../Data/simulation_processed/training/',window_size = 15, sample_time = 1, label = False, edge_attr = None, mask = False, metadata = False):
        super().__init__(root)
        '''
        data_list - list of sample indexes.
        source_dir - source_dir to read the data
        window_size - time window size (s)
        sample_time - sample frequency, e.g. 2 means 1 sample every 2 seconds
        label - read label for each car each time. True for test set
        edge_attr - predefined edge weights, 'sim', 'exp' or None. sim: 1/distance; exp: exp(- distance ** 2)
        mask - mask for padded trajectories because of cars leaving/entering the stretch
        metadata - read metadata
        '''
        self.data_list = data_list 
        self.dir = source_dir  # read from directory
        self.window_size = window_size
        self.sample_time = sample_time
        self.edge_attr = edge_attr
        self.label = label
        self.mask = mask
        self.metadata = metadata

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self): 
        # Download to `self.raw_dir`.
        pass

    def process(self):
        pass

    def len(self):
        'Denotes the total number of samples'
        return len(self.data_list)

    def get(self, idx):
        'Generates one sample of data'
        i = self.data_list[idx]
        graph_list = []
        # load data 
        for t in range(0, self.window_size, self.sample_time):
            # node features
            x = np.load(self.dir +"x_{}_{}.npy".format(i, t))
            x = torch.from_numpy(x).float()

            # edge
            edge_index = np.load(self.dir + "adj_list_{}_{}.npy".format(i, t))
            edge_index = torch.from_numpy(edge_index).type(torch.LongTensor)
            graph = Data(x=x, edge_index=edge_index)

            # edge attributes
            if self.edge_attr:
                if self.edge_attr == 'sim':
                    dist_edge_attr = np.load(self.dir + "edge_attr_sim_{}_{}.npy".format(i, t))
                elif self.edge_attr == 'exp':
                    dist_edge_attr = np.load(self.dir + "edge_attr_exp_{}_{}.npy".format(i, t))
                    dist_edge_attr = np.expand_dims(dist_edge_attr, axis=-1)  
                else:
                    raise NotImplementedError("edge attr not implemented")
                dist_edge_attr = torch.from_numpy(dist_edge_attr).float()
                graph.edge_attr = dist_edge_attr
            
            if self.label:
                label = np.load(self.dir + "label_{}_{}.npy".format(i, t))
                label = torch.from_numpy(label).float()
                graph.label = label
            if self.mask:
                mask = np.load(self.dir + "mask_{}_{}.npy".format(i,t))
                mask = torch.from_numpy(mask)
                graph.mask = mask
                
            if self.metadata and t==0:
                times_range = range(0, self.window_size, self.sample_time)
                times = np.load(self.dir+"time_stamp_{}.npy".format(i))
                times = times[times_range]
                IDs = np.load(self.dir+"IDs_{}.npy".format(i))
                df_n = np.load(self.dir+"df_{}.npy".format(i))
                graph.times = times
                graph.IDs = IDs
                graph.df_n = df_n
                

            graph_list.append(graph)
        return graph_list
