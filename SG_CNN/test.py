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



def test_sgcnn(data_dir, test_data, checkpoint_file):
    """
    Define a function to test the 3D CNN model

    inputs:
    1) data_dir: path to hdf data
    2) test_data: testing hdf file name
    3) checkpoint_file: checkpoint file
    """

    # set directory to path containing data
    os.chdir(data_dir)

    # define parameters
    checkpoint = True              # boolean flag for checkpoints
    checkpoint_iter = 100           # number of epochs per checkpoint
    epochs = 100                   # number of training epochs
    num_workers = 24               # number of workers for datloader
    batch_size = 32               # batch size to use for training
    lr = 1e-3                      # learning rate to use for training
    covalent_gather_width = 128
    non_covalent_gather_width = 128
    covalent_k = 1
    non_covalent_k = 1
    covalent_threshold = 1.5
    non_covalent_threshold = 7.5
    feature_size = 20      
    output_file_name = None

    # load checkpoint file
    if torch.cuda.is_available():
        model_train_dict = torch.load(checkpoint_file)
    else:
        model_train_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        
    # construct model
    model = GeometricDataParallel(PotentialNetParallel(in_channels=feature_size, out_channels=1, covalent_gather_width=covalent_gather_width,
                                                        non_covalent_gather_width=non_covalent_gather_width, covalent_k=covalent_k, 
                                                        non_covalent_k=non_covalent_k, covalent_neighbor_threshold=covalent_threshold, 
                                                        non_covalent_neighbor_threshold=non_covalent_threshold)).float()

    model.load_state_dict(model_train_dict["model_state_dict"])

    # construct dataset
    dataset = SGCNN_Dataset(data_file= test_data, output_info=True, cache_data=False)

    # construct dataloader
    dataloader = DataListLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = [x[1] for x in batch if x is not None]
            y_ = model(data)
            y = torch.cat([x[1].y for x in batch])
            y_true.append(y.cpu().data.numpy())
            y_pred.append(y_.cpu().data.numpy())
    y_true = np.concatenate(y_true).reshape(-1, 1)
    y_pred = np.concatenate(y_pred).reshape(-1, 1)
    # compute r^2
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    # compute mae
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    # compute mse
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    # compute pearson correlation coefficient
    pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
    # compute spearman correlation coefficient
    spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))
    # print evaluation summary for testing
    print("r2: {}\tmae: {}\tmse: {}\tpearsonr: {}\t spearmanr: {}".format(r2, mae, mse, pearsonr, spearmanr))
