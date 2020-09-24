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
		data = Planetoid(path, name=dataset, split='full')[0]
		# data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())
		data.num_classes = (max(data.y)+1).item() 
		# dataset = Planetoid(path, name=dataset, split='public', transform=T.NormalizeFeatures(), num_train_per_class=20, num_val=500, num_test=1000)
		# data = modify_train_mask(data)

	elif dataset == 'Amazon':
		path = os.path.join(dirname, 'data', 'Amazon')
		data = Amazon(path, 'photo')[0]
		data.num_classes = (max(data.y)+1).item() 
		data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())
		# Amazon: 4896 train, 1224 val, 1530 test

	elif dataset == 'Reddit': 
		path = os.path.join(dirname, 'data', 'Reedit')
		data = Reddit(path)[0]
		data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())

	elif dataset == 'PPI':
		path = os.path.join(dirname, 'data', 'PPI')
		data = ppi_prepoc(path, seed)
		data.x = data.graphs[0].x
		data.num_classes = data.graphs[0].y.size(1)
		for df in data.graphs: 
			df.num_classes = data.num_classes
	
	elif dataset in ['syn1', 'syn2', 'syn4', 'syn5']:
		data = synthetic_data(dataset, dirname, args_input_dim = 10, args_train_ratio = 0.6)
	
	#elif dataset = 'MUTAG'

	# Get it in right format
	if dataset != 'PPI':
		print('Train mask is of size: ', data.train_mask[data.train_mask==True].shape)
	
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


def _get_train_val_test_masks(total_size, y_true, val_fraction, test_fraction, seed):
	"""
	Get train, val and test split masks for `total_size` examples with the labels `y_true`. Performs stratified
	splitting over the labels `y_true`. `y_true` is a numpy array.
	"""
	# Split into a train, val and test set 
	# Store indexes of the nodes belong to train, val and test set 
	indexes = range(total_size)
	indexes_train, indexes_test = train_test_split(indexes, test_size=test_fraction, stratify=y_true, random_state=seed)
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
	return _get_train_val_test_masks(y.shape[0], y, 0.2, 0.2, seed=10)


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

	return data


def add_noise_features(data, num_noise, binary=False, p=0.5):
	"""
	:param data: downloaded dataset 
	:param num_noise: number of noise features we want to add
	:param binary: True if want binary node features 
	:return: dataset with additional noisy features
	"""

	# Do nothing if no noise feature to add 
	if not num_noise: 
		return data 

	# Number of nodes in the dataset
	num_nodes = data.x.size(0)

	# Define some random features (randomly), in addition to existing ones
	m = torch.distributions.bernoulli.Bernoulli(torch.tensor([p]))
	noise_feat = m.sample((num_noise, num_nodes)).T[0]
	#noise_feat = torch.randint(2,size=(num_nodes, num_noise))
	if not binary: 
		noise_feat_bis = torch.rand((num_nodes, num_noise))
		# noise_feat_bis = noise_feat_bis - noise_feat_bis.mean(1, keepdim=True)
		noise_feat = torch.min(noise_feat, noise_feat_bis)
	data.x = torch.cat([noise_feat, data.x], dim=-1)

	return data, noise_feat


def add_noise_neighbours(data, num_noise, node_indices, binary=False, p=0.5, connectedness='high'):
	"""
	:param data: downloaded dataset 
	:param num_noise: number of noise features we want to add
	:param node_indices: list of test samples 
	:param binary: True if want binary node features 
	:param p: for binary features, proba that each feature = 1
	:param connectedness: how connected are new nodes, either 'low', 'medium' or 'high' 
	:return: dataset with additional nodes, with noisy features and connections; 
	and  noisy nodes features
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
	if connectedness == 'high': # few highly connected new nodes
		adj_matrix = torch.randint(2,size=(num_noise, new_num_nodes))

	elif connectedness == 'medium': # more sparser nodes, connected to targeted nodes of interest
		m = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1]))
		adj_matrix = m.sample((new_num_nodes, num_noise)).T[0]
		for i,idx in enumerate(node_indices): # each node of interest has at least one noisy neighbour
			try: 
				adj_matrix[i,idx]=1
			except IndexError: # in case num_noise < test_samples
				pass
	else: 
		adj_matrix = torch.zeros((num_noise, new_num_nodes))
		for i,idx in enumerate(node_indices):
			try:
				adj_matrix[i,idx]=1
			except IndexError:
				pass
		while num_noise > i+1: 
			l = node_indices + list(range(num_nodes,(num_nodes+i)))
			i += 1
			idx = random.sample(l,2)
			adj_matrix[i,idx[0]]=1
			adj_matrix[i,idx[1]]=1
	
	# Add defined edges to data adjacency matrix, in the correct form
	for i, row in enumerate(adj_matrix):
		indices = (row == 1).nonzero()
		indices = torch.transpose(indices, 0, 1)
		a = torch.full_like(indices, i + num_nodes)
		adj_row = torch.cat((a,indices),0)
		data.edge_index = torch.cat((data.edge_index,adj_row),1)
		adj_row = torch.cat((indices,a),0)
		data.edge_index = torch.cat((data.edge_index,adj_row),1)

	# Update train/test/val masks - don't include these new nodes anywhere as there have no labels
	test_mask = torch.empty(num_noise)
	test_mask = torch.full_like(test_mask, False).bool()
	data.train_mask = torch.cat((data.train_mask,test_mask),-1)
	data.val_mask = torch.cat((data.val_mask,test_mask),-1)
	data.test_mask = torch.cat((data.test_mask,test_mask),-1)
	# Update labels randomly - no effect on the rest
	data.y = torch.cat((data.y, test_mask),-1)

	return data


def extract_test_nodes(data, num_samples):
	"""
	:param data: dataset
	:param num_samples: number of test samples desired
	:return: list of indexes representing nodes used as test samples
	"""
	test_indices = data.test_mask.cpu().numpy().nonzero()[0]
	node_indices = np.random.choice(test_indices, num_samples).tolist()
	
	return node_indices

# Files required 
from src.train import train_and_val
import src.gengraph as gengraph
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import utils.featgen as featgen
import configs


def synthetic_data(dataset, dirname, args_input_dim = 10, args_train_ratio = 0.6):
	"""
	Create synthetic data, similarly to what was done in GNNExplainer
	Pipeline was adapted so as to fit ours. 
	"""
	# Define path where dataset should be saved 
	data_path = "data/{}.pth".format(dataset)

	# If already created, do not recreate 
	if os.path.exists(data_path):
		data = torch.load(data_path)

	else:
		# Construct graph
		if dataset =='syn1':	
			G, labels, name = gengraph.gen_syn1( feature_generator=featgen.ConstFeatureGen(np.ones(args_input_dim)) )
		elif dataset =='syn4':
			G, labels, name = gengraph.gen_syn4( feature_generator=featgen.ConstFeatureGen(np.ones(args_input_dim, dtype=float)) )
		elif dataset == 'syn5':
			G, labels, name = gengraph.gen_syn5( feature_generator=featgen.ConstFeatureGen(np.ones(args_input_dim, dtype=float)) )
		elif dataset =='syn2':
			G, labels, name = gengraph.gen_syn2()
			args_input_dim = len(G.nodes[0]["feat"])

		# Create dataset
		data = SimpleNamespace()
		data.x, data.edge_index, data.y = gengraph.preprocess_input_graph(G, labels)
		a = torch.randperm(data.x.size()[0])
		data.y, data.x = data.y[a], data.x[a,:]
		data.num_classes = max(labels) + 1
		data.num_features = args_input_dim
		data.num_nodes = G.number_of_nodes()
			
		# Train/test split only for nodes
		data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())

		# Save data
		torch.save(data, data_path)

	return data 
