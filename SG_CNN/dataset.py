import os
import os.path as osp
import multiprocessing as mp
import numpy as np
import pandas as pd
import h5py
import itertools
import random
from tqdm import tqdm
from glob import glob
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler, Subset
from torch._C import NoneType
import torch.nn as nn
from torch.nn import BatchNorm1d, init
from torch.nn import Parameter as Param
import torch.nn.functional as F
from torch_sparse import coalesce
from torch.optim import Adam, lr_scheduler
from torch_geometric.data import DataLoader as GeometricDataLoader
from torch_geometric.data import Data, Batch, DataListLoader
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.nn import GCNConv, GlobalAttention, global_add_pool, NNConv, avg_pool_x, avg_pool, max_pool_x
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform, reset
from torch_geometric.utils import softmax, dense_to_sparse
from torch_geometric.utils import (to_dense_batch, add_self_loops, remove_self_loops, normalized_cut, dense_to_sparse, 
                                   is_undirected, to_undirected, scatter_, contains_self_loops)

""" Define a class to contain the data that will be included in the dataloader 
sent to the SG CNN model """
class SGCNN_Dataset(Dataset):
    def __init__(self, data_file, output_info=False, cache_data=True):
        super(SGCNN_Dataset, self).__init__()
        self.data_file = data_file
        self.output_info = output_info
        self.cache_data = cache_data
        self.data_dict = {}  # used to store data
        self.data_list = []  # used to store id's for data
        
        # retrieve PDB id's and affinities from h5py file
        with h5py.File(data_file, "r") as f:
            for pdbid in f.keys():
                affinity = np.asarray(f[pdbid].attrs["affinity"]).reshape(1, -1)
                self.data_list.append((pdbid, affinity))  

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        if self.cache_data:
            if item in self.data_dict.keys():
                return self.data_dict[item]
            else:
                pass       
        pdbid, affinity = self.data_list[item]
        node_feats, coords = None, None
        
        with h5py.File(self.data_file, "r") as f:
            if (not pdbid in f.keys()):
                print(pdbid + 'was not found in the data file')
                return None
            data = f[pdbid]
            vdw_radii = (f[pdbid].attrs["van_der_waals"].reshape(-1, 1))
            coords = data[:, 0:3]
            node_feats = np.concatenate([vdw_radii, data[:, 3:22]], axis=1)
        
        # account for the vdw radii in distance calculations (consider each atom as a sphere)
        dists = pairwise_distances(coords, metric="euclidean")
        edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())
        x = torch.from_numpy(node_feats).float()
        y = torch.FloatTensor(affinity).view(-1, 1)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y)
        
        if self.cache_data:
            if self.output_info:
                self.data_dict[item] = (pdbid, data)
            else:
                self.data_dict[item] = data
            return self.data_dict[item]
        else:
            if self.output_info:
                return (pdbid, data)
            else:
                return data
