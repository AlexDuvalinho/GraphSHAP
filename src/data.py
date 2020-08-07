import numpy as np
import os as os

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def prepare_data(dataset='Cora'):
	"""
	:param dataset: name of the dataset used
	:return: data, in the correct format
	"""
	# Retrieve main path of project
	dirname = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	# Store path to data location 
	path = os.path.join(dirname, 'data', 'Planetoid')
	# Download and store dataset
	dataset = Planetoid(path, dataset, T.NormalizeFeatures())
	# Get it in right format
	data = dataset[0]

	# data = modify_train_mask(data)
    # data = add_noise_features(data, args.num_noise)
	
	return data


def modify_train_mask(data):
	"""
	:param data: dataset downloaded above
	:return: same dataset but different validation/test/train split
	"""
	val_mask = data.val_mask
	test_mask = data.test_mask
	new_train_mask = ~(val_mask + test_mask)
	data.train_mask = new_train_mask

	return data


def add_noise_features(data, num_noise):
	"""
	:param data: downloaded dataset 
	:param num_noise: number of noise features we want to add
	:return: new dataset with additional noisy features
	"""
	# Do nothing if no noise feature to add 
	if not num_noise: 
		return data 

	# Number of nodes in the dataset
	num_nodes = data.x.size(0)

	# Define some random features (randomly), in addition to existing ones
	noise_feat = torch.randn((num_nodes, num_noise))
	noise_feat = noise_feat - noise_feat.mean(1, keepdim=True)
	data.x = torch.cat([data.x, noise_feat], dim=-1)

	return data


def extract_test_nodes(data, num_samples):
	"""
	:param num_samples: 
	:return: 
	"""
	test_indices = data.test_mask.cpu().numpu().nonzero()[0]
	node_indices = np.random.choice(test_indices, num_samples).tolist()
	
	return node_indices

