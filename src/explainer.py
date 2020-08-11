# Import libraries 
import scipy.special
import numpy as np
from copy import deepcopy
import torch_geometric


class GraphSHAP():

	def __init__(self, data, model):
		self.model = model
		self.data = data
		self.model.eval()

	def explainer(self, node_index=0, hops=1, num_samples=10):
		"""
		:param node_index: index of the node of interest
		:param hops: number k of k-hop neighbours to considere in the subgraph
		:param num_samples: number of samples we want to form GraphSHAP's new dataset
		"""
		# Create a variable to store node features 
		X = deepcopy(data.x)
		x = X[node_index,:]

		# Construct k hop subgraph of node of interest (denoted v)
		neighbours, adj, _, edge_mask =\
			 torch_geometric.utils.k_hop_subgraph(node_idx=node_index, 
												num_hops=hops, 
												edge_index= data.edge_index)

		# Get neighbours of node v (need to exclude v)
		neighbours = neighbours[neighbours!=node_index]
		D = neighbours.shape[0] 

	 	# Number of non-zero entries for the feature vector x of node v
		F = x[x==1].shape[0]
		# Corresponding indexes of these entries
		idx = torch.nonzero(x)

		# Total number of features + neighbours considered for node of interest
		M = F+D

		# Sample z' 
		z_ = torch.empty(num_samples, M).random_(2)
		# Compute |z'| for each sample z'
		s = (z_ != 0).sum(dim=1)

		# Define weights associated with each sample
		weights = shapley_kernel(M,s)

		
		###  Create dataset from z'.
		# Reference sample is set to 0 everywhere since it is a categorical var.

		# Define new node features dataset, where, for each new sample
		# only the features where z_ == 1 are kept 
		new_X = torch.zeros([num_samples,data.num_features])
		for i in range(num_samples):
			for j in range(F):
				if z_[i,j].item()==1: 
					new_X[i,idx[j].item()]=1
		
		# Subsample neighbors (neighbours and )
		for i in range(num_samples):
			for j in range(D):
				if z_[i,F+j]==1:
					node_id = neighbours[j].item()
		# Need to store pair (node_index, node_id) and find a way to remove
		# it form adjacency matrix 	(do it when computing f from stored pairs)





	@ static
	def shapley_kernel(M, s):
		"""
		:param M: number of features + number of neighbours
		:param s: dimension of z' (number of features + neighbours included)
		:return: [scalar] value of shapley value 
		"""
		shap_kernel = []
		# Loop around elements of s in order to specify a special case
		# Otherwise could have procedeed with tensor s direclty
		for i in range(s.shape[0]):
			a = s[i].item()
			# Put an emphasis on samples where all or none features are included
			if a == 0 or a == M: 
				shap_kernel.append(1000)
			else: 
				shap_kernel.append((M-1)/(scipy.special.binom(M,a)*a*(M-a)))
		return torch.tensor(shap_kernel)