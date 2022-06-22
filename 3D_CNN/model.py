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

""" Define a Gaussian Filter to smoothen and remove noise from data """
class GaussianFilter(nn.Module):
	def __init__(self, dim=2, channels=3, kernel_size=11, sigma=1, use_cuda=True):
		super(GaussianFilter, self).__init__()
		self.use_cuda = use_cuda
		if isinstance(kernel_size, numbers.Number):
			self.kernel_size = [kernel_size] * dim
		if isinstance(sigma, numbers.Number):
			self.sigma = [sigma] * dim	
		self.padding = kernel_size // 2

		# Gaussian kernel is the product of the gaussian filter for each dimension.
		kernel = 1
		meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in self.kernel_size])
		for size, std, mgrid in zip(self.kernel_size, self.sigma, meshgrids):
			mean = (size - 1) / 2
			kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
		kernel = kernel / torch.sum(kernel)

		# reshape kernel to depth-wise convolutional weight
		kernel = kernel.view(1, 1, *kernel.size())
		kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
		if self.use_cuda:
			kernel = kernel.cuda()
		self.register_buffer('weight', kernel)
		self.groups = channels
		if dim == 1:
			self.conv = nn.functional.conv1d
		elif dim == 2:
			self.conv = nn.functional.conv2d
		elif dim == 3:
			self.conv = nn.functional.conv3d

	def forward(self, input):
		return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)
	
""" Construct 3D CNN model """
class Model_3DCNN(nn.Module):
	def __init__(self, feat_dim=19, output_dim=1, num_filters=[64,128,256], use_cuda=True, verbose=False):
		super(Model_3DCNN, self).__init__()     
		self.feat_dim = feat_dim
		self.output_dim = output_dim
		self.num_filters = num_filters
		self.use_cuda = use_cuda
		self.verbose = verbose

		self.conv_block1 = self.__conv_layer_set__(self.feat_dim, self.num_filters[0], 7, 2, 3)
		self.res_block1 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[0], 7, 1, 3)
		self.res_block2 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[0], 7, 1, 3)
		self.conv_block2 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[1], 7, 3, 3)
		self.max_pool2 = nn.MaxPool3d(2)
		self.conv_block3 = self.__conv_layer_set__(self.num_filters[1], self.num_filters[2], 5, 2, 2)
		self.max_pool3 = nn.MaxPool3d(2)
		self.fc1 = nn.Linear(2048, 100)
		torch.nn.init.normal_(self.fc1.weight, 0, 1)
		self.fc1_bn = nn.BatchNorm1d(num_features=100, affine=True, momentum=0.1).train()
		self.fc2 = nn.Linear(100, 1)
		torch.nn.init.normal_(self.fc2.weight, 0, 1)
		self.relu = nn.ReLU()

	def __conv_layer_set__(self, in_c, out_c, k_size, stride, padding):
		conv_layer = nn.Sequential(
			nn.Conv3d(in_c, out_c, kernel_size=k_size, stride=stride, padding=padding, bias=True),
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(out_c))
		return conv_layer

	def forward(self, x):
		if x.dim() == 1:
			x = x.unsqueeze(-1)
		conv1_h = self.conv_block1(x)
		if self.verbose != 0:
			print(conv1_h.shape)

		conv1_res1_h = self.res_block1(conv1_h)
		if self.verbose != 0:
			print(conv1_res1_h.shape)

		conv1_res1_h2 = conv1_res1_h + conv1_h
		if self.verbose != 0:
			print(conv1_res1_h2.shape)

		conv1_res2_h = self.res_block2(conv1_res1_h2)
		if self.verbose != 0:
			print(conv1_res2_h.shape)

		conv1_res2_h2 = conv1_res2_h + conv1_h
		if self.verbose != 0:
			print(conv1_res2_h2.shape)

		conv2_h = self.conv_block2(conv1_res2_h2)
		if self.verbose != 0:
			print(conv2_h.shape)

		pool2_h = self.max_pool2(conv2_h)
		if self.verbose != 0:
			print(pool2_h.shape)

		conv3_h = self.conv_block3(pool2_h)
		if self.verbose != 0:
			print(conv3_h.shape)

		pool3_h = conv3_h
		#pool3_h = self.max_pool3(conv3_h)
		#if self.verbose != 0:
		#	print(pool3_h.shape)

		flatten_h = pool3_h.view(pool3_h.size(0), -1)
		if self.verbose != 0:
			print(flatten_h.shape)

		fc1_z = self.fc1(flatten_h)
		fc1_y = self.relu(fc1_z)
		fc1_h = self.fc1_bn(fc1_y) if fc1_y.shape[0]>1 else fc1_y  #batchnorm train require more than 1 batch
		if self.verbose != 0:
			print(fc1_h.shape)

		fc2_z = self.fc2(fc1_h)
		if self.verbose != 0:
			print(fc2_z.shape)

		return fc2_z, fc1_z
