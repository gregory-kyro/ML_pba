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

def train_sgcnn(data_dir, train_data, val_data, checkpoint_dir):
  """
  Define a function to train the SG CNN model
  
  inputs:
  1) data_dir: path to hdf data
  2) train_data: training hdf file name
  3) val_data: validation hdf file name
  4) checkpoint_dir: path to save checkpoint file: 'path/to/file.pt'
  
  output:
  1) checkpoint file, to load into testing function: saved as:
     '/best_checkpoint.pth'
  """
  
  # set directory to path containing data
  os.chdir(data_dir)
  
  # define parameters
  checkpoint = True              # boolean flag for checkpoints
  checkpoint_iter = 10           # number of epochs per checkpoint
  epochs = 100                   # number of training epochs
  num_workers = 24               # number of workers for datloader
  batch_size = 32                # batch size to use for training
  lr = 1e-3                      # learning rate to use for training
  covalent_gather_width = 128
  non_covalent_gather_width = 128
  covalent_k = 1
  non_covalent_k = 1
  covalent_threshold = 1.5
  non_covalent_threshold = 7.5
  feature_size = 20              # this is 20 to account for vdw radii

  # seed all random number generators and set cudnn settings for deterministic
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  torch.cuda.manual_seed_all(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False 
  os.environ["PYTHONHASHSEED"] = "0"
  
  def worker_init_fn(worker_id):
    np.random.seed(int(0))
  
  # construct datasets fromt training and validation data
  train_dataset = SGCNN_Dataset(data_file=train_data, output_info=True)
  val_dataset = SGCNN_Dataset(data_file=val_data, output_info=True)
  
  
  # construct training and validation dataloaders to be fed to model
  train_dataloader = DataListLoader(train_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, drop_last=True)
  val_dataloader = DataListLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, drop_last=True)
  
  # print statement of complexes for sanity check
  tqdm.write("{} complexes in training dataset".format(len(train_dataset)))
  tqdm.write("{} complexes in validation dataset".format(len(val_dataset)))
  
  # construct SG CNN model
  model = GeometricDataParallel(PotentialNetParallel(in_channels=feature_size, out_channels=1, covalent_gather_width=covalent_gather_width,
                                                     non_covalent_gather_width=non_covalent_gather_width, covalent_k=covalent_k, 
                                                     non_covalent_k=non_covalent_k, covalent_neighbor_threshold=covalent_threshold,
                                                     non_covalent_neighbor_threshold=non_covalent_threshold)).float()
  model.train()
  model.to(0)
  
  # set loss as MSE
  criterion = nn.MSELoss().float()
  # set Adam optimizer
  optimizer = Adam(model.parameters(), lr=lr) 
  
  # initialize checkpoint parameters
  best_checkpoint_dict = None
  best_checkpoint_epoch = 0
  best_checkpoint_step = 0
  best_checkpoint_r2 = -100000000000
  
  # train model
  step = 0
  for epoch in range(epochs):
      losses = []
      for batch in tqdm(train_dataloader):
          batch = [x for x in batch if x is not None]
          if len(batch) < 1:
              print("empty batch, skipping to next batch")
              continue
          optimizer.zero_grad()
          data = [x[1] for x in batch]
          y_ = model(data)
          y = torch.cat([x[1].y for x in batch])
          # compute loss
          loss = criterion(y.float(), y_.cpu().float())
          losses.append(loss.cpu().data.item())
          loss.backward()
          y_true = y.cpu().data.numpy()
          y_pred = y_.cpu().data.numpy()
          # compute r^2
          r2 = r2_score(y_true=y_true, y_pred=y_pred)
          # compute mae
          mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
          # compute pearson correlation coefficient
          pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
          # compute spearman correlation coefficient
          spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))
          # write training summary for each epoch
          tqdm.write("epoch: {}\tloss:{:0.4f}\tr2: {:0.4f}\t pearsonr: {:0.4f}\tspearmanr: {:0.4f}\tmae: {:0.4f}\tpred stdev: {:0.4f}"
                     "\t pred mean: {:0.4f} \tcovalent_threshold: {:0.4f} \tnon covalent threshold: {:0.4f}".format(
                       epoch, loss.cpu().data.numpy(), r2, float(pearsonr[0]), float(spearmanr[0]), float(mae), np.std(y_pred), np.mean(y_pred),
                       model.module.covalent_neighbor_threshold.t.cpu().data.item(), model.module.non_covalent_neighbor_threshold.t.cpu().data.item()))

          if checkpoint:
              if step % checkpoint_iter == 0:
                  checkpoint_dict = checkpoint_model(model, val_dataloader, epoch, step)
                  if checkpoint_dict["validate_dict"]["r2"] > best_checkpoint_r2:
                      best_checkpoint_step = step
                      best_checkpoint_epoch = epoch
                      best_checkpoint_r2 = checkpoint_dict["validate_dict"]["r2"]
                      best_checkpoint_dict = checkpoint_dict

          optimizer.step()
          step += 1

      if checkpoint:
          checkpoint_dict = checkpoint_model(model, val_dataloader, epoch, step)
          if checkpoint_dict["validate_dict"]["r2"] > best_checkpoint_r2:
              best_checkpoint_step = step
              best_checkpoint_epoch = epoch
              best_checkpoint_r2 = checkpoint_dict["validate_dict"]["r2"]
              best_checkpoint_dict = checkpoint_dict

  if checkpoint:
      # once broken out of the loop, save last model
      checkpoint_dict = checkpoint_model(model, val_dataloader, epoch, step)

      if checkpoint_dict["validate_dict"]["r2"] > best_checkpoint_r2:
          best_checkpoint_step = step
          best_checkpoint_epoch = epoch
          best_checkpoint_r2 = checkpoint_dict["validate_dict"]["r2"]
          best_checkpoint_dict = checkpoint_dict

  if checkpoint:
      torch.save(best_checkpoint_dict, checkpoint_dir + '/best_checkpoint.pth')
  print("best training checkpoint epoch {}/step {} with r2: {}".format(best_checkpoint_epoch, best_checkpoint_step, best_checkpoint_r2))
  
  # define function to perform validation
  def validate(model, val_dataloader):
      # initialize
      model.eval()
      y_true = []
      y_pred = []
      pdbid_list = []
      pose_list = []
      # validation
      for batch in tqdm(val_dataloader):
          data = [x[1] for x in batch if x is not None]
          y_ = model(data)
          y = torch.cat([x[1].y for x in batch])
          pdbid_list.extend([x[0] for x in batch])
          pose_list.extend([x[1] for x in batch])
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
      # compte spearman correlation coefficient
      spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))
      # write out metrics
      tqdm.write(str(
              "r2: {}\tmae: {}\tmse: {}\tpearsonr: {}\t spearmanr: {}".format(r2, mae, mse, pearsonr, spearmanr)))
      
      model.train()
      return {"r2": r2, "mse": mse, "mae": mae, "pearsonr": pearsonr, "spearmanr": spearmanr,
              "y_true": y_true, "y_pred": y_pred, "pdbid": pdbid_list, "pose": pose_list}

  # define function to return checkpoint dictionary
  def checkpoint_model(model, dataloader, epoch, step):
      validate_dict = validate(model, dataloader)
      model.train()
      checkpoint_dict = {"model_state_dict": model.state_dict(), "args": NoneType, "step": step, "epoch": epoch, "validate_dict": validate_dict}
      torch.save(checkpoint_dict, './sgcnn_checkpoint')
      # return the computed metrics so it can be used to update the training loop
      return checkpoint_dict
