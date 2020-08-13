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
		:return: shapley values for features/neighbours that influence node v's pred
		"""

		### Determine z' => features and neighbours whose importance is investigated

		# Create a variable to store node features 
		x = self.data.x[node_index,:]

		# Store number of classes (TODO: condition on Cora or find other way to get this info)
		num_classes = (max(self.data.y)+1).item() # if Cora, use probas.shape of test.py or else

		# Construct k hop subgraph of node of interest (denoted v)
		neighbours, _, _, _ =\
			 torch_geometric.utils.k_hop_subgraph(node_idx=node_index, 
												num_hops=hops, 
												edge_index= self.data.edge_index)
		# Store the indexes of the neighbours of v (+ index of v itself)

		# Remove node v index from neighbours and store their number in D
		neighbours = neighbours[neighbours!=node_index]
		D = neighbours.shape[0] 

	 	# Number of non-zero entries for the feature vector x_v
		F = x[x==1].shape[0]
		# Store indexes of these non zero feature values
		feat_idx = torch.nonzero(x)

		# Total number of features + neighbours considered for node v
		M = F+D

		# Sample z' - binary vector of dimension (num_samples, M)
		z_ = torch.empty(num_samples, M).random_(2)
		# Compute |z'| for each sample z'
		s = (z_ != 0).sum(dim=1)


		### Define weights associated with each sample using shapley kernel formula
		weights = self.shapley_kernel(M,s)


		###  Create dataset (z', f(z)), stored as (z_, fz)
		# Retrive z from z' and x_v, then compute f(z)
		fz = self.compute_pred(node_index, num_classes, num_samples, F, D, z_, neighbours, feat_idx)
		

		### OLS estimator for weighted linear regression
		phi = self.OLS(z_, weights, fz)
		

		# Print some information
		print('Explanations include {} node features and {} neighbours for this node\
		for {} classes'.format(F, D, num_classes))
		
		# Compare with true prediction of the model - see what class should truly be explained
		true_conf, true_pred  = self.model(x=self.data.x, edge_index=self.data.edge_index).exp()[node_index].max(dim=0)
		print('Prediction of orignal model is class {} with confidence {}'.format(true_pred, true_conf))


		### Visualisation 
		# Call visu function

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

	def compute_pred(self, node_index, num_classes, num_samples, F, D, z_, neighbours, feat_idx):
		"""
		Variables are exactly as defined in explainer function, where compute_pred is used
		This function aims to construct z (from z' and x_v) and then to compute f(z), 
		meaning the prediction of the new instances with our original model. 
		In fact, it builds the dataset (z', f(z)), required to train the weighted linear model.
		"""
		# This implies retrieving z from z' - wrt sampled neighbours and node features
		# We start this process here by storing new node features for v and neigbours to 
		# isolate
		X_v = torch.zeros([num_samples,self.data.num_features])
		excluded_nei = {}

		# Do it for each sample
		for i in range(num_samples):
			
			# Define new node features dataset (we only modify x_v for now)
			# Features where z_j == 1 are kept, others are set to 0 
			for j in range(F):
				if z_[i,j].item()==1: 
					X_v[i,feat_idx[j].item()]=1
		
			# Define new neighbourhood
			# Store index of neighbours that need to be shut down (not sampled, z_j=0)
			nodes_id = []
			for j in range(D):
				if z_[i,F+j]==0:
					node_id = neighbours[j].item()
					nodes_id.append(node_id)
			# Dico with key = num_sample id, value = excluded neighbour index
			excluded_nei[i] = nodes_id

		# Init label f(z) for graphshap dataset - consider all classes
		fz = torch.zeros((num_samples, num_classes))
		# Init final predicted class for each sample (informative)
		classes_labels = torch.zeros(num_samples)
		pred_confidence = torch.zeros(num_samples)

		# Create new matrix A and X - for each sample ≈ reform z from z'
		for key, value in excluded_nei.items():
			
			positions = []
			# For each excluded neighbour, retrieve the column index of each occurence 
			# in the adj matrix - store in positions (list)
			for val in value: 
				pos = (self.data.edge_index==val).nonzero()[:,1].tolist()
				positions += pos
			# Create new adjacency matrix for that sample
			positions = list(set(positions))
			A = np.array(self.data.edge_index)
			A = np.delete(A, positions, axis=1)
			A = torch.tensor(A)
			
			# Change feature vector for node of interest 
			# NOTE: maybe change values of all nodes for features not inlcuded, not just x_v
			X = deepcopy(self.data.x)
			X[node_index,:] = X_v[key,:]

			# Apply model on (X,A) as input. 
			proba = self.model(x=X, edge_index=A).exp()[node_index] 
			
			# Store final class prediction and confience level
			pred_confidence[key], classes_labels[key] = torch.topk(proba, k=1) # optional
			
			# Store predicted class label in fz 
			fz[key] = proba

		return fz

	def OLS(self, z_, weights, fz):
		"""
		:param z_: z' - binary vector  
		:param weights: shapley kernel weights for z'
		:param fz: f(z) where z is a new instance - formed from z' and x
		:return: estimated coefficients of our weighted linear regression - on (z', f(z))
		"""
		# OLS to estimate parameter of Weighted Linear Regression
		tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
		phi = np.dot(tmp, np.dot(np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))
		return phi

	def vizu(self, true_pred, phi, feat_idx, neighbours):
		"""
		:param true_pred: class predicted by original model for node of interest
		:param phi: shapley values
		:param feat_idx: index of features whose importance is assessed
		:param neighbours: index of nodes whose importance is assessed
		:return: nice visualisation (like SHAP) of each feature/neigbour's average
		marginal contribution towards the prediction 
		"""
		# Explanation for this class - TODO: improve => print in for loop item next to influence
		# Even do vizualisation of result like SHAPE does
		print('Explanation for class {} are {}. They regard these features {} and these neighbours {}'\
			.format(true_pred, phi[true_pred],feat_idx, neighbours))