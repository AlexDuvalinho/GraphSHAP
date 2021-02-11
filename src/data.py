"""data.py

	Import and process relevant datasets 
"""
from torch_geometric.datasets import PPI, Amazon, Planetoid, Reddit
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import torch
import numpy as np
import os as os
import random
from types import SimpleNamespace
import warnings

warnings.filterwarnings('ignore')


def prepare_data(dataset, seed):
	"""Import, save and process dataset

	Args:
			dataset (str): name of the dataset used
			seed (int): seed number

	Returns:
			[torch_geometric.Data]: dataset in the correct format 
			with required attributes and train/test/val split
	"""
	# Retrieve main path of project
	dirname = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

	# Download and store dataset at chosen location
	if dataset == 'Cora' or dataset == 'PubMed' or dataset == 'Citeseer':
		path = os.path.join(dirname, 'data')
		data = Planetoid(path, name=dataset, split='full')[0]
		data.name = dataset
		data.num_classes = (max(data.y)+1).item()
		# data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())
		# data = Planetoid(path, name=dataset, split='public', transform=T.NormalizeFeatures(), num_train_per_class=20, num_val=500, num_test=1000)

	elif dataset == 'Amazon':
		path = os.path.join(dirname, 'data', 'Amazon')
		data = Amazon(path, 'photo')[0]
		data.name = dataset
		data.num_classes = (max(data.y)+1).item()
		data.train_mask, data.val_mask, data.test_mask = split_function(
			data.y.numpy(), seed=seed)
		# Amazon: 4896 train, 1224 val, 1530 test

	return data


def _get_train_val_test_masks(total_size, y_true, val_fraction, test_fraction, seed):
	"""Performs stratified train/test/val split

	Args:
		total_size (int): dataset total number of instances
		y_true (numpy array): labels
		val_fraction (int): validation set proportion
		test_fraction (int): test set proportion
		seed (int): seed value

	Returns:
		[torch.tensors]: train, validation and test masks - boolean values
	"""
	# Split into a train, val and test set
	# Store indexes of the nodes belong to train, val and test set
	indexes = range(total_size)
	indexes_train, indexes_test = train_test_split(
		indexes, test_size=test_fraction, stratify=y_true, random_state=seed)
	indexes_train, indexes_val = train_test_split(indexes_train, test_size=val_fraction, stratify=y_true[indexes_train],
												  random_state=seed)
	# Init masks
	train_idxs = np.zeros(total_size, dtype=np.bool)
	val_idxs = np.zeros(total_size, dtype=bool)
	test_idxs = np.zeros(total_size, dtype=np.bool)

	# Update masks using corresponding indexes
	train_idxs[indexes_train] = True
	val_idxs[indexes_val] = True
	test_idxs[indexes_test] = True

	return torch.from_numpy(train_idxs), torch.from_numpy(val_idxs), torch.from_numpy(test_idxs)


def split_function(y, args_train_ratio=0.6, seed=10):
	return _get_train_val_test_masks(y.shape[0], y, (1-args_train_ratio)/2, (1-args_train_ratio), seed=seed)


def add_noise_features(data, num_noise, binary=False, p=0.5):
	"""Add noisy features to original dataset

	Args:
		data (torch_geometric.Data): downloaded dataset 
		num_noise ([type]): number of noise features we want to add
		binary (bool, optional): True if want binary node features. Defaults to False.
		p (float, optional): Proportion of 1s for new features. Defaults to 0.5.

	Returns:
		[torch_geometric.Data]: dataset with additional noisy features
	"""

	# Do nothing if no noise feature to add
	if not num_noise:
		return data, None

	# Number of nodes in the dataset
	num_nodes = data.x.size(0)

	# Define some random features, in addition to existing ones
	m = torch.distributions.bernoulli.Bernoulli(torch.tensor([p]))
	noise_feat = m.sample((num_noise, num_nodes)).T[0]
	# noise_feat = torch.randint(2,size=(num_nodes, num_noise))
	if not binary:
		noise_feat_bis = torch.rand((num_nodes, num_noise))
		# noise_feat_bis = noise_feat_bis - noise_feat_bis.mean(1, keepdim=True)
		noise_feat = torch.min(noise_feat, noise_feat_bis)
	data.x = torch.cat([noise_feat, data.x], dim=-1)

	return data, noise_feat


def add_noise_neighbours(data, num_noise, node_indices, binary=False, p=0.5, connectedness='medium', c=0.001):
	"""Add noisy nodes to original dataset

	Args:
		data (torch_geometric.Data): downloaded dataset 
		num_noise (int): number of noise features we want to add
		node_indices (list): list of test samples 
		binary (bool, optional): True if want binary node features. Defaults to False.
		p (float, optional): proba that each binary feature = 1. Defaults to 0.5.
		connectedness (str, optional): how connected are new nodes, either 'low', 'medium' or 'high'.
			Defaults to 'high'.

	Returns:
		[torch_geometric.Data]: dataset with additional nodes, with random features and connections
	"""
	if not num_noise:
		return data

	# Number of features in the dataset
	num_feat = data.x.size(1)
	num_nodes = data.x.size(0)

	# Add new nodes with random features
	m = torch.distributions.bernoulli.Bernoulli(torch.tensor([p]))
	noise_nei_feat = m.sample((num_feat, num_noise)).T[0]
	if not binary:
		noise_nei_feat_bis = torch.rand((num_noise, num_feat))
		noise_nei_feat = torch.min(noise_nei_feat, noise_nei_feat_bis)
	data.x = torch.cat([data.x, noise_nei_feat], dim=0)
	new_num_nodes = data.x.size(0)

	# Add random edges incident to these nodes - according to desired level of connectivity
	if connectedness == 'high':  # few highly connected new nodes
		adj_matrix = torch.randint(2, size=(num_noise, new_num_nodes))

	elif connectedness == 'medium':  # more sparser nodes, connected to targeted nodes of interest
		m = torch.distributions.bernoulli.Bernoulli(torch.tensor([c]))
		adj_matrix = m.sample((new_num_nodes, num_noise)).T[0]
		# each node of interest has at least one noisy neighbour
		for i, idx in enumerate(node_indices):
			try:
				adj_matrix[i, idx] = 1
			except IndexError:  # in case num_noise < test_samples
				pass
	# low connectivity
	else: 
		adj_matrix = torch.zeros((num_noise, new_num_nodes))
		for i, idx in enumerate(node_indices):
			try:
				adj_matrix[i, idx] = 1
			except IndexError:
				pass
		while num_noise > i+1:
			l = node_indices + list(range(num_nodes, (num_nodes+i)))
			i += 1
			idx = random.sample(l, 2)
			adj_matrix[i, idx[0]] = 1
			adj_matrix[i, idx[1]] = 1

	# Add defined edges to data adjacency matrix, in the correct form
	for i, row in enumerate(adj_matrix):
		indices = (row == 1).nonzero()
		indices = torch.transpose(indices, 0, 1)
		a = torch.full_like(indices, i + num_nodes)
		adj_row = torch.cat((a, indices), 0)
		data.edge_index = torch.cat((data.edge_index, adj_row), 1)
		adj_row = torch.cat((indices, a), 0)
		data.edge_index = torch.cat((data.edge_index, adj_row), 1)

	# Update train/test/val masks - don't include these new nodes anywhere as there have no labels
	test_mask = torch.empty(num_noise)
	test_mask = torch.full_like(test_mask, False).bool()
	data.train_mask = torch.cat((data.train_mask, test_mask), -1)
	data.val_mask = torch.cat((data.val_mask, test_mask), -1)
	data.test_mask = torch.cat((data.test_mask, test_mask), -1)
	# Update labels randomly - no effect on the rest
	data.y = torch.cat((data.y, test_mask), -1)

	return data


def extract_test_nodes(data, num_samples, seed):
	"""Select some test samples - without repetition

	Args:
		num_samples (int): number of test samples desired

	Returns:
		[list]: list of indexes representing nodes used as test samples
	"""
	np.random.seed(seed)
	test_indices = data.test_mask.cpu().numpy().nonzero()[0]
	node_indices = np.random.choice(test_indices, num_samples, replace=False).tolist()

	return node_indices
