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
		X = deepcopy(data.x)
		x = X[node_index,:]

		# k-hop-subgraph
		neighbours, adj, _, edge_mask =\
			 torch_geometric.utils.k_hop_subgraph(node_idx=node_index, 
												num_hops=hops, 
												edge_index= data.edge_index)

		# Get neighbours of node_index
		D = neighbours.shape[0]

	 	# Number of non-zero entries
		F = x[x==1].shape[0]
		# Corresponding indexes of these entires
		idx = torch.nonzero(x)

		# Total number of features + neighbours considered for node of interest
		M = D+F

		# Sample z' - use node neighbours + use self.data.num_features
		z_ = torch.empty(M, num_samples).random_(2)
		s = (z_ != 0).sum(dim=0)

		# Define weights associated with each sample
		weights = shapley_kernel(M,s)

		# Create dataset from z'.
		# Need position of each feature/neighbour included (neighbours), 
		# contained in neighbours and in idx (need to mask maybe)


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