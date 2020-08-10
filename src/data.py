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

	# Update the training mask of the dataset 
	data = modify_train_mask(data)
	print('Train mask is of size: ', data.train_mask[data.train_mask==True].shape)
    # data = add_noise_features(data, args.num_noise)
	
	return data


def modify_train_mask(data):
	"""
	:param data: dataset downloaded above
	:return: same dataset but with extended train set mask
	"""
	# We define the new train mask on all observations not part of the validation/test set 
	new_train_mask = ~(data.val_mask + data.test_mask)
	data.train_mask = new_train_mask

	# Train_mask was of size 140 over 2708. val_mask was/is 500 and test_mask 1000. 
	# New train mask is of size 1208 (1208 + 1500 = 2708)
	
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

