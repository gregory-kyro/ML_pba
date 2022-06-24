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
                                   is_undirected, to_undirected, contains_self_loops)

class GatedGraphConv(MessagePassing):
    """ The gated graph convolution operator from 'Gated Graph Sequence Neural Networks':
    <https://arxiv.org/abs/1511.05493> """

    def __init__(self, out_channels, num_layers, edge_network, aggr="add", bias=True):
        super(GatedGraphConv, self).__init__(aggr)
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.edge_network = edge_network
        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        size = self.out_channels
        uniform(size, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        assert h.size(1) <= self.out_channels
        # if input size is < out_channels, pad input with 0s to achieve the same size
        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)
        for i in range(self.num_layers):
            m = torch.matmul(h, self.weight[i])
            m = self.propagate(edge_index=edge_index, x=h, aggr="add")
            h = self.rnn(m, h)
        return h

    def message(self, x_j): 
        # constructs messages
        return x_j

    def update(self, aggr_out):
        # updates node embeddings
        return aggr_out

    def __repr__(self):
        return "{}({}, num_layers={})".format(self.__class__.__name__, self.out_channels, self.num_layers)

class GlobalAttention(torch.nn.Module):
    """ The global soft attention layer from 'Gated Graph Sequence Neural Networks'
    <https://arxiv.org/abs/1511.05493> """

    def __init__(self, gate_nn, nn=None):
        super(GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size
        gate = self.gate_nn(x)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
        gate = softmax(gate, batch, size)
        out = torch.scatter(gate * x, batch, size, reduce = 'add')
        return out

    def __repr__(self):
        return "{}(gate_nn={}, nn={})".format(self.__class__.__name__, self.gate_nn, self.nn)

class PotentialNetAttention(torch.nn.Module):
    def __init__(self, net_i, net_j):
        super(PotentialNetAttention, self).__init__()
        self.net_i = net_i
        self.net_j = net_j

    def forward(self, h_i, h_j):
        return torch.nn.Softmax(dim=1)(self.net_i(torch.cat([h_i, h_j], dim=1))) * self.net_j(h_j)

class PotentialNetPropagation(torch.nn.Module):
    def __init__(self, feat_size=19, gather_width=64, k=2, neighbor_threshold=None, output_pool_result=False, bn_track_running_stats=False):
        super(PotentialNetPropagation, self).__init__()
        assert neighbor_threshold is not None
        self.neighbor_threshold = neighbor_threshold
        self.bn_track_running_stats = bn_track_running_stats
        self.edge_attr_size = 1
        self.k = k
        self.gather_width = gather_width
        self.feat_size = feat_size
        self.edge_network_nn = nn.Sequential(nn.Linear(self.edge_attr_size, int(self.feat_size / 2)), nn.Softsign(),
                                             nn.Linear(int(self.feat_size / 2), self.feat_size), nn.Softsign())
        self.edge_network = NNConv(self.feat_size, self.edge_attr_size * self.feat_size, nn=self.edge_network_nn, root_weight=True, aggr="add")
        self.gate = GatedGraphConv(self.feat_size, self.k, edge_network=self.edge_network)
        self.attention = PotentialNetAttention(net_i=nn.Sequential(nn.Linear(self.feat_size * 2, self.feat_size), nn.Softsign(), 
                                                                   nn.Linear(self.feat_size, self.gather_width), nn.Softsign()),
                                               net_j=nn.Sequential(nn.Linear(self.feat_size, self.gather_width), nn.Softsign()))
        self.output_pool_result = output_pool_result
        if self.output_pool_result:
            self.global_add_pool = global_add_pool

    def forward(self, data, edge_index, edge_attr):
        # propagtion
        h_0 = data
        h_1 = self.gate(h_0, edge_index, edge_attr)
        h_1 = self.attention(h_1, h_0)
        return h_1
      
# define function to be used in GraphThreshold (below)
def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]

class GraphThreshold(torch.nn.Module):
    def __init__(self, t):
        super(GraphThreshold, self).__init__()
        if torch.cuda.is_available():
            self.t = nn.Parameter(t, requires_grad=True).cuda()
        else:
            self.t = nn.Parameter(t, requires_grad=True)

    def filter_adj(self, row, col, edge_attr, mask):
        mask = mask.squeeze()
        return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]

    def forward(self, edge_index, edge_attr):
        # randomly drops edges from the adjacency matrix
        N = maybe_num_nodes(edge_index, None)
        row, col = edge_index
        mask = edge_attr <= self.t
        row, col, edge_attr = self.filter_adj(row, col, edge_attr, mask)
        edge_index = torch.stack([torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)], dim=0)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
        return edge_index, edge_attr

class PotentialNetFullyConnected(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PotentialNetFullyConnected, self).__init__()
        self.output = nn.Sequential(nn.Linear(in_channels, int(in_channels / 1.5)), nn.ReLU(), 
                                    nn.Linear(int(in_channels / 1.5), int(in_channels / 2)), nn.ReLU(), 
                                    nn.Linear(int(in_channels / 2), out_channels))

    def forward(self, data, return_hidden_feature=False):
        if return_hidden_feature:
            return self.output[:-2](data), self.output[:-4](data), self.output(data)
        else:
            return self.output(data)

class PotentialNetParallel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, covalent_gather_width=128, non_covalent_gather_width=64,
                 covalent_k=1, non_covalent_k=1, covalent_neighbor_threshold=None, 
                 non_covalent_neighbor_threshold=None, always_return_hidden_feature=False):
        super(PotentialNetParallel, self).__init__()
        assert (covalent_neighbor_threshold is not None and non_covalent_neighbor_threshold is not None)

        if torch.cuda.is_available():
            self.covalent_neighbor_threshold = GraphThreshold(torch.ones(1).cuda() * covalent_neighbor_threshold)
        else:
            self.covalent_neighbor_threshold = GraphThreshold(torch.ones(1) * covalent_neighbor_threshold)

        if torch.cuda.is_available():
            self.non_covalent_neighbor_threshold = GraphThreshold(torch.ones(1).cuda() * non_covalent_neighbor_threshold)
        else:
            self.non_covalent_neighbor_threshold = GraphThreshold(torch.ones(1) * non_covalent_neighbor_threshold)

        self.always_return_hidden_feature = always_return_hidden_feature
        self.global_add_pool = global_add_pool
        self.covalent_propagation = PotentialNetPropagation(feat_size=in_channels, gather_width=covalent_gather_width,
                                                            neighbor_threshold=self.covalent_neighbor_threshold, k=covalent_k)
        self.non_covalent_propagation = PotentialNetPropagation(feat_size=covalent_gather_width, gather_width=non_covalent_gather_width, 
                                                                neighbor_threshold=self.non_covalent_neighbor_threshold, k=non_covalent_k)
        self.global_add_pool = global_add_pool
        self.output = PotentialNetFullyConnected(non_covalent_gather_width, out_channels)

    def forward(self, data, return_hidden_feature=False):
        if torch.cuda.is_available():
            data.x = data.x.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.edge_index = data.edge_index.cuda()
            data.batch = data.batch.cuda()

        # make sure that graph is undirected
        if not is_undirected(data.edge_index):
            data.edge_index = to_undirected(data.edge_index)

        # make sure that nodes can propagate messages to themselves
        if not contains_self_loops(data.edge_index):
            data.edge_index, data.edge_attr = add_self_loops(data.edge_index, data.edge_attr.view(-1))

        # covalent_propagation, add self loops to enable self-propagation
        covalent_edge_index, covalent_edge_attr = self.covalent_neighbor_threshold(data.edge_index, data.edge_attr)
        (non_covalent_edge_index, non_covalent_edge_attr) = self.non_covalent_neighbor_threshold(data.edge_index, data.edge_attr)

        # covalent_propagation and non_covalent_propagation
        covalent_x = self.covalent_propagation(data.x, covalent_edge_index, covalent_edge_attr)
        non_covalent_x = self.non_covalent_propagation(covalent_x, non_covalent_edge_index, non_covalent_edge_attr)

        # zero out the protein features then do ligand only gathering
        non_covalent_ligand_only_x = non_covalent_x
        non_covalent_ligand_only_x[data.x[:, 14] == -1] = 0
        pool_x = self.global_add_pool(non_covalent_ligand_only_x, data.batch)

        # fully connected and output layers
        if return_hidden_feature or self.always_return_hidden_feature:
            avg_covalent_x, _ = avg_pool_x(data.batch, covalent_x, data.batch)
            avg_non_covalent_x, _ = avg_pool_x(data.batch, non_covalent_x, data.batch)
            fc0_x, fc1_x, output_x = self.output(pool_x, return_hidden_feature=True)
            return avg_covalent_x, avg_non_covalent_x, pool_x, fc0_x, fc1_x, output_x
        else:
            return self.output(pool_x)
