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

def train_3dcnn(data_dir, hdf_fn, vhdf_fn, thdf_fn, checkpoint_dir):
  """
  Define a function to train the 3D CNN model
  
  inputs:
  1) data_dir: path to hdf data
  2) hdf_fn: training hdf file name
  3) vhdf_fn: validation hdf file name
  4) thdf_fn: testing hdf file name
  5) checkpoint_dir: path to save checkpoint file: 'path/to/file.pt'
  
  output:
  1) checkpoint file, to load into testing function
  """
  
  # define parameters
  batch_size = 50
  learning_rate = .0007
  decay_iter = 100
  decay_rate = 0.95
  epoch_count = 50
  checkpoint_iter = 50
  device_name = "cuda:0"
  multi_gpus = False
  verbose=False
  
  # set CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  cuda_count = torch.cuda.device_count()
  if use_cuda:
    device = torch.device(device_name)
    torch.cuda.set_device(int(device_name.split(':')[1]))
  else:
    device = torch.device("cpu")
  print(use_cuda, cuda_count, device)

  def worker_init_fn(worker_id):
    np.random.seed(int(0))
    
  # build training dataset variable
  dataset = Dataset_hdf(os.path.join(data_dir, hdf_fn))
  
  # build validation dataset variable
  val_dataset = Dataset_hdf(os.path.join(data_dir, vhdf_fn))
  
  # check multi-gpus
	num_workers = 0
	if multi_gpus and cuda_count > 1:
		num_workers = cuda_count
    
  # initialize training data loader
	batch_count = len(dataset.data_info_list) // batch_size
	dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=None)
  
  # initialize validation data loader
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=None)
  
  # define gaussian_filter
	gaussian_filter = GaussianFilter(dim=3, channels=19, kernel_size=11, sigma=1, use_cuda=use_cuda)
  
  # define model
	model = Model_3DCNN(use_cuda=use_cuda, verbose=verbose)

	if multi_gpus and cuda_count > 1:
		model = nn.DataParallel(model)
	model.to(device)

	if isinstance(model, (DistributedDataParallel, DataParallel)):
		model = model.module
    
  # define loss
  loss_fn = nn.MSELoss().float()
  
  # define optimizer
  optimizer = RMSprop(model.parameters(), lr=learning_rate)
  
  # define scheduler
  scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_iter, gamma=decay_rate)
  
  # train model
  epoch_start = 0
	step = 0
  
	for epoch_ind in range(epoch_start, epoch_count):
		vol_batch = torch.zeros((batch_size,19,48,48,48)).float().to(device)
		losses = []
		model.train()
		for batch_ind, batch in enumerate(dataloader):
      # transfer to GPU
			x_batch_cpu, y_batch_cpu = batch
			x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
      # gaussian filter
			vol_batch = gaussian_filter(x_batch)
			ypred_batch, _ = model(vol_batch[:x_batch.shape[0]])
      # compute loss
			loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float())
			losses.append(loss.cpu().data.item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
      
      # print information during training
			print("[%d/%d-%d/%d] training, loss: %.3f, lr: %.7f" % (epoch_ind+1, epoch_count, batch_ind+1, batch_count, loss.cpu().data.item(), optimizer.param_groups[0]['lr']))
			if step % checkpoint_iter == 0:
				checkpoint_dict = {
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"loss": loss,
					"step": step,
					"epoch": epoch_ind
				}
				torch.save(checkpoint_dict, checkpoint_dir)
				print("checkpoint saved: %s" % checkpoint_dir)
			step += 1
      
    # print training information for each epoch
		print("[%d/%d] training, epoch loss: %.3f" % (epoch_ind+1, epoch_count, np.mean(losses)))
    
    # validation step
		if val_dataset: 
			val_losses = []
			model.eval()
			with torch.no_grad():
				for batch_ind, batch in enumerate(val_dataloader):
          # transfer to GPU
					x_batch_cpu, y_batch_cpu = batch
					x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
          # apply gaussian filter
					vol_batch = gaussian_filter(x_batch)
					ypred_batch, _ = model(vol_batch[:x_batch.shape[0]])
          # compute loss
					loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float())
					val_losses.append(loss.cpu().data.item())
          # print information during training
					print("[%d/%d-%d/%d] validation, loss: %.3f" % (epoch_ind+1, epoch_count, batch_ind+1, batch_count, loss.cpu().data.item()))
        
        # print validation information for each epoch
				print("[%d/%d] validation, epoch loss: %.3f" % (epoch_ind+1, epoch_count, np.mean(val_losses)))
        
  # close dataset
	dataset.close()
	if (val_dataset):
		val_dataset.close()
