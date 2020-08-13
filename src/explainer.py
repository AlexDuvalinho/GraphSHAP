# Import libraries 
import scipy.special
import numpy as np
from copy import deepcopy
import torch_geometric
import torch


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
		X = deepcopy(self.data.x)
		A = deepcopy(self.data.edge_index)
		x = X[node_index,:]
		num_classes = (max(self.data.y)+1).item() # add if Cora, use probas.shape of test.py or else

		# Construct k hop subgraph of node of interest (denoted v)
		neighbours, adj, _, edge_mask =\
			 torch_geometric.utils.k_hop_subgraph(node_idx=node_index, 
												num_hops=hops, 
												edge_index= self.data.edge_index)

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
		weights = self.shapley_kernel(M,s)

		
		###  Create dataset from z'.
		# Reference sample is set to 0 everywhere since it is a categorical var.

		# Define new node features dataset, where, for each new sample
		# only the features where z_ == 1 are kept 
		new_X = torch.zeros([num_samples,self.data.num_features])
		for i in range(num_samples):
			for j in range(F):
				if z_[i,j].item()==1: 
					new_X[i,idx[j].item()]=1
		
		# Subsample neighbours - store index of neighbours which need to be shut down
		excluded_nei = {}
		for i in range(num_samples):
			nodes_id = []
			for j in range(D):
				if z_[i,F+j]==0:
					node_id = neighbours[j].item()
					nodes_id.append(node_id)
			# Store in a dico with key = num_sample id, value = excluded neigh. index
			excluded_nei[i] = nodes_id
		
		
		# Next, find a way to remove all edges incident to selected neighbours
		# from edge_index = adj matrix. Want to isolate these nodes to prevent them 
		# from influencing prediction related to node v. 

		
		# Def label dataset
		fz = torch.zeros((num_samples, num_classes))
		# fz = np.zeros((num_samples, num_classes))

		# Create new matrix A and X - for each sample 
		for key, value in excluded_nei.items():
			
			# Distinguish (maybe) between case where all neighbours
			if value == []: 
				pass
				# Need to compute predictions as well and
				# Maybe not needed as a special case in fact
			else:
				positions = []
				for val in value:
					# Retrieve column index in adjacency matrix of undesired neighbours
					pos = (A==val).nonzero()[:,1].tolist()
					positions += pos
				# Create new adjacency matrix for that sample
				positions = list(set(positions))
				A_ = np.array(A)
				A_ = np.delete(A_, pos, axis=1)
				A_ = torch.tensor(A_)
			
			
			# Change feature vector for node of interest 
			# NOTE: maybe change values of all nodes for features not inlcuded, not just x_v
			X_ = deepcopy(X)
			X_[node_index,:] = new_X[key,:]

			# Apply GCN model with A_, X_ as input. 
			log_logits = self.model(x=X_, edge_index=A_) # [2708, 7]
			proba = log_logits.exp()[node_index]
			# p, class_indices = torch.topk(proba, k=1)
			
			# Store predicted class label in fz 
			fz[key] = proba
			
		# Final dataset for SHAP is stored as (z_, fz)

		
		# OLS to estimate parameter of Weighted Linear Regression

		tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
		phi = np.dot(tmp, np.dot(np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))
		
		print('Explanations include {} node features and {} neighbours for this node\
		for {} classes'.format(F, D, num_classes))
		
		# Maybe return only explanations of the class chosen by the model 
		# Not fz, but pred for original data - real class chosen. Can explain why.

		return phi

	def shapley_kernel(self, M, s):
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