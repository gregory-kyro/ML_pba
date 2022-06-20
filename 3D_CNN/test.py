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

