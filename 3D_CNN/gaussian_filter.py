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