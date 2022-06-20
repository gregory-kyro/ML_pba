import os
import sys
sys.stdout.flush()
import csv
import h5py
import numpy as np
import math
import numbers
import scipy as sp
from scipy.stats import *
import random
import pandas as pd
from sklearn.metrics import *
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, RMSprop, lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset

""" Define function to strip prefix if present in checkpoint file """
def strip_prefix_if_present(state_dict, prefix):
	keys = sorted(state_dict.keys())
	if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
		return
	for key in keys:
		newkey = key[len(prefix) :]
		state_dict[newkey] = state_dict.pop(key)
	try:
		metadata = state_dict._metadata
	except AttributeError:
		pass
	else:
		for key in list(metadata.keys()):
			if len(key) == 0:
				continue
			newkey = key[len(prefix) :]
			metadata[newkey] = metadata.pop(key)

def test_3dcnn(data_dir, thdf_fn, model_dir):
  """
  Define a function to test the 3D CNN model
  
  inputs:
  1) data_dir: path to hdf data
  2) thdf_fn: testing hdf file name
  3) model_dir: 'directory/to/checkpoint/file.pt'
  
  output:
  1) checkpoint file, to load into testing function
  """
  
  # define parameters
  save_pred = True        # whether to save prediction results in csv file
  save_feat = True        # whether to save fully connected features in npy file
  multi_gpus = False
  verbose=False
  batch_size = 50
  device_name = "cuda:0"

  # set CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  cuda_count = torch.cuda.device_count()
  if use_cuda:
    device = torch.device(device_name)
    torch.cuda.set_device(int(device_name.split(':')[1]))
  else:
    device = torch.device("cpu")
  print(use_cuda, cuda_count, device)
