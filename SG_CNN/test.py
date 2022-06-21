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

def test_sgcnn(data_dir, test_data, model_dir):
  """
  Define a function to test the 3D CNN model
  
  inputs:
  1) data_dir: path to hdf data
  2) test_data: testing hdf file name
  3) model_dir: 'directory/to/checkpoint/file.pt'
  
  output:
  1) csv file containing summary of evaluation (prediction, label),
     saves as: '3d_cnn_evaluation.csv'
  2) np file containing summary of feature information, saves as:
     '3d_cnn_evaluation.np'
  """
  
  # set directory to path containing data

  # define parameters
  save_pred = True        # whether to save prediction results in csv file















'''This is my version of a testing module, based on theirs for the 3dcnn'''

#Unnecessary if everything is in the same notebook
# from data_utils import PDBBindDataset
# from model import PotentialNetParallel

test_data = 'cleaned_testing_set.hdf'
load_checkpoint = 'sgcnn_checkpoint'
output_file_name = None

def test():

    if torch.cuda.is_available():

        model_train_dict = torch.load(load_checkpoint)

    else:
        model_train_dict = torch.load(load_checkpoint, map_location=torch.device('cpu'))


    model = GeometricDataParallel(
            PotentialNetParallel(
                in_channels=feature_size,
                out_channels=1,
                covalent_gather_width=covalent_gather_width,
                non_covalent_gather_width=non_covalent_gather_width,
                covalent_k=covalent_k,
                non_covalent_k=non_covalent_k,
                covalent_neighbor_threshold=covalent_threshold,
                non_covalent_neighbor_threshold=non_covalent_threshold,
            )
        ).float()


    model.load_state_dict(model_train_dict["model_state_dict"])



    
    dataset = PDBBindDataset(
                data_file= test_data,
                output_info=True,
                cache_data=False,
            )
    
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

    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
    spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))



    print("r2: {}\tmae: {}\tmse: {}\tpearsonr: {}\t spearmanr: {}".format(r2, mae, mse, pearsonr, spearmanr))



if __name__ == "__main__": 
    test()
