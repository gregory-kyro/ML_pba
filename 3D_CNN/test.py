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
  1) csv file containing summary of evaluation (prediction, label),
		 saves as: '3d_cnn_evaluation.csv'
	2) np file containing summary of feature information, saves as:
	   '3d_cnn_evaluation.np'
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
     
	# load testing dataset
	dataset = Dataset_hdf(os.path.join(data_dir, thdf_fn))

	# check multi-gpus
	num_workers = 0
	if multi_gpus and cuda_count > 1:
		num_workers = cuda_count

	# initialize testing data loader
	batch_count = len(dataset) // batch_size
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=None)

	# define gaussian_filter
	gaussian_filter = GaussianFilter(dim=3, channels=19, kernel_size=5, sigma=1, use_cuda=use_cuda)

	# define model
	model = Model_3DCNN(use_cuda=use_cuda, verbose=False)
	
	if multi_gpus and cuda_count > 1:
		model = nn.DataParallel(model)
	model.to(device)

	if isinstance(model, (DistributedDataParallel, DataParallel)):
		model = model.module
	
	# load checkpoint file
	checkpoint = torch.load(model_dir, map_location=device)
	
	# model state dict
	model_state_dict = checkpoint.pop("model_state_dict")
	strip_prefix_if_present(model_state_dict, "module.")
	model.load_state_dict(model_state_dict, strict=False)

	vol_batch = torch.zeros((batch_size,19,48,48,48)).float().to(device)
	
	# create empty arrays to hold predicted and true values
	ytrue_arr = np.zeros((len(dataset),), dtype=np.float32)
	ypred_arr = np.zeros((len(dataset),), dtype=np.float32)
	zfeat_arr = np.zeros((len(dataset), 100), dtype=np.float32)
	pred_list = []
	
	model.eval()
	with torch.no_grad():
		for batch_ind, batch in enumerate(dataloader):
			# transfer to GPU
			x_batch_cpu, y_batch_cpu = batch
			x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
			# arrange and filter
			bsize = x_batch.shape[0]
			vol_batch = gaussian_filter(x_batch)
			
			ypred_batch, zfeat_batch = model(vol_batch[:x_batch.shape[0]])
			ytrue = y_batch_cpu.float().data.numpy()[:,0]
			ypred = ypred_batch.cpu().float().data.numpy()[:,0]
			zfeat = zfeat_batch.cpu().float().data.numpy()
			ytrue_arr[batch_ind*batch_size:batch_ind*batch_size+bsize] = ytrue
			ypred_arr[batch_ind*batch_size:batch_ind*batch_size+bsize] = ypred
			zfeat_arr[batch_ind*batch_size:batch_ind*batch_size+bsize] = zfeat
			
			if save_pred:
				for i in range(bsize):
					pred_list.append([batch_ind + i, ytrue[i], ypred[i]])

			print("[%d/%d] evaluating" % (batch_ind+1, batch_count))
	
	# define rmse
	rmse = math.sqrt(mean_squared_error(ytrue_arr, ypred_arr))
	# define mae
	mae = mean_absolute_error(ytrue_arr, ypred_arr)
	# define r^2
	r2 = r2_score(ytrue_arr, ypred_arr)
	# define pearson correlation coefficient
	pearson, ppval = pearsonr(ytrue_arr, ypred_arr)
	# define spearman correlation coefficient
	spearman, spval = spearmanr(ytrue_arr, ypred_arr)
	# define mean
	mean = np.mean(ypred_arr)
	# define standard deviation
	std = np.std(ypred_arr)
	
	# print evaluation summary after validation has finished
	print("Evaluation Summary:")
	print("RMSE: %.3f, MAE: %.3f, R^2 score: %.3f, Pearson: %.3f, Spearman: %.3f, mean/std: %.3f/%.3f" % (rmse, mae, r2, pearson, spearman, mean, std))
	
	# write csv file containing evaluation summary information
	if save_pred:
		csv_fpath = '3d_cnn_evaluation.csv'
		df = pd.DataFrame(pred_list, columns=["cid", "label", "pred"])
		df.to_csv(csv_fpath, index=False)
	
	# write np file containing feature information
	if save_feat:
		npy_fpath = '3d_cnn_evaluation.np'
		np.save(npy_fpath, zfeat_arr)
