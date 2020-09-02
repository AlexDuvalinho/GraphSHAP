import numpy as np
import os as os
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
import random

import torch
from torch_geometric.datasets import Planetoid, PPI, Amazon, Reddit
import torch_geometric.transforms as T
from torch_geometric.data import Data


def prepare_data(dataset, seed):
	"""
	:param dataset: name of the dataset used
	:return: data, in the correct format
	"""
	# Retrieve main path of project
	dirname = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

	# Download and store dataset at chosen location
	if dataset == 'Cora' or dataset=='PubMed' or dataset=='Citeseer':
		path = os.path.join(dirname, 'data')
		data = Planetoid(path, name=dataset, split='full', transform=T.NormalizeFeatures())[0]
		# dataset = Planetoid(path, name=dataset, split='public', transform=T.NormalizeFeatures(), num_train_per_class=20, num_val=500, num_test=1000)
		# data = modify_train_mask(data)

	elif dataset == 'Amazon':
		path = os.path.join(dirname, 'data', 'Amazon')
		data = Amazon(path, 'photo')[0]
		data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())
		# Amazon: 4896 train, 1224 val, 1530 test

	elif dataset == 'Reddit':
		path = os.path.join(dirname, 'data', 'Reedit')
		data = Reddit(path)[0]
		data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())

	elif dataset == 'PPI':
		path = os.path.join(dirname, 'data', 'PPI')
		data, data.graphs = ppi_prepoc(dirname, seed)
		#data = PPI(path, split='train')[0]
		#data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())
		
	#elif dataset = 'MUTAG'

	# Get it in right format
	if dataset != 'PPI':
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

	# For Cora, train_mask is of size 1208 (prevoulsy 140). val_mask was/is 500 and test_mask 1000. 
	# For PubMed, train_mask is 18217, val mask 500 and test mask 1000
	
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



def _get_train_val_test_masks(total_size, y_true, val_fraction, test_fraction, seed):
	"""
	Get train, val and test split masks for `total_size` examples with the labels `y_true`. Performs stratified
	splitting over the labels `y_true`. `y_true` is a numpy array.
	"""
	# Split into a train, val and test set 
	# Store indexes of the nodes belong to train, val and test set 
	indexes = range(total_size)
	indexes_train, indexes_test = train_test_split(indexes, test_size=test_fraction, stratify=y_true, random_state=0)
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

def split_function(y):
	return _get_train_val_test_masks(y.shape[0], y, 0.2, 0.2, seed=0)



def ppi_prepoc(dirname, seed):
	# 20 protein graphs - some set as validation, some as train, some as test. 
	# Need to create the relevant masks for each graph
	data = SimpleNamespace()
	data.graphs = []
	for split in ['train', 'val', 'test']:
		split_data = PPI(root=dirname, split=split, pre_transform=T.NormalizeFeatures())
		x_idxs = split_data.slices['x'].numpy()
		edge_idxs = split_data.slices['edge_index'].numpy()
		split_data = split_data.data
		for x_start, x_end, e_start, e_end in zip(x_idxs, x_idxs[1:], edge_idxs, edge_idxs[1:]):
			graph = Data(split_data.x[x_start:x_end], split_data.edge_index[:, e_start:e_end],
							y=split_data.y[x_start:x_end])
			graph.num_nodes = int(x_end - x_start)
			graph.split = split
			all_true = torch.ones(graph.num_nodes).bool()
			all_false = torch.zeros(graph.num_nodes).bool()
			graph.train_mask = all_true if split == 'train' else all_false
			graph.val_mask = all_true if split == 'val' else all_false
			graph.test_mask = all_true if split == 'test' else all_false
			data.graphs.append(graph)
	if seed != 0:
		temp_random = random.Random(seed)
		val_graphs = temp_random.sample(range(len(data.graphs)), 2)
		test_candidates = [graph_idx for graph_idx in range(len(data.graphs)) if graph_idx not in val_graphs]
		test_graphs = temp_random.sample(test_candidates, 2)
		for graph_idx, graph in enumerate(data.graphs):
			all_true = torch.ones(graph.num_nodes).bool()
			all_false = torch.zeros(graph.num_nodes).bool()
			graph.split = 'test' if graph_idx in test_graphs else 'val' if graph_idx in val_graphs else 'train'
			graph.train_mask = all_true if graph.split == 'train' else all_false
			graph.val_mask = all_true if graph.split == 'val' else all_false
			graph.test_mask = all_true if graph.split == 'test' else all_false

	return data, data.graphs