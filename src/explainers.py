""" explainers.py

	Define the different explainers: GraphSHAP + benchmarks
"""

from src.train import accuracy
from src.models import LinearRegressionModel
from sklearn.metrics import r2_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import warnings
import time
import random

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import torch
import torch_geometric
from sklearn.linear_model import LassoLars, Ridge, Lasso
from itertools import combinations

# GraphLIME
from src.plots import visualize_subgraph, k_hop_subgraph, denoise_graph, log_graph

# GNNExplainer
from torch_geometric.nn import GNNExplainer as GNNE
from torch_geometric.nn import MessagePassing



class GraphSHAP():

	def __init__(self, data, model):
		self.model = model
		self.data = data
		self.F = None  # number of non zero node features
		self.neighbours = None  # neighbours considered
		self.M = None  # number of nonzero features - for each node index

		self.model.eval()

	def explain(self, node_index=0, hops=2, num_samples=10, info=True):
		""" Explain prediction for a particular node - GraphSHAP method

		Args:
			node_index (int, optional): index of the node of interest. Defaults to 0.
			hops (int, optional): number k of k-hop neighbours to consider in the subgraph 
													around node_index. Defaults to 2.
			num_samples (int, optional): number of samples we want to form GraphSHAP's new dataset. 
													Defaults to 10.
			info (bool, optional): Print information about explainer's inner workings. 
													And include vizualisation. Defaults to True.

		Returns:
				[type]: shapley values for features/neighbours that influence node v's pred
		"""
		# Time
		start = time.time()

		# Determine z' => features and neighbours whose importance is investigated

		# Create a variable to store node features
		x = self.data.x[node_index, :]

		# Number of non-zero entries for the feature vector x_v
		self.F = x[x != 0].shape[0]
		# Store indexes of these non zero feature values
		feat_idx = torch.nonzero(x)

		# Construct k hop subgraph of node of interest (denoted v)
		self.neighbours, _, _, edge_mask =\
			torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
												 num_hops=hops,
												 edge_index=self.data.edge_index)
		# Store the indexes of the neighbours of v (+ index of v itself)

		# Remove node v index from neighbours and store their number in D
		self.neighbours = self.neighbours[self.neighbours != node_index]
		D = self.neighbours.shape[0]

		# Total number of features + neighbours considered for node v
		self.M = self.F+D

		# F node features first, then D neighbours
		#z_ = torch.empty(num_samples, self.M).random_(2)
		#z_[0, :] = torch.ones(self.M)
		#z_[1, :] = torch.zeros(self.M)
			
		# Sample z' - binary vector of dimension (num_samples, M)
		z_ = self.coalition_sampler(num_samples)
		# Compute |z'| for each sample z'
		s = (z_ != 0).sum(dim=1)

		# Compute true prediction of model, for original instance
		with torch.no_grad():
			true_conf, true_pred = self.model(
				x=self.data.x, edge_index=self.data.edge_index).exp()[node_index].max(dim=0)

		# Define weights associated with each sample using shapley kernel formula
		weights = self.shapley_kernel(s)

		# Create dataset (z', f(z)), stored as (z_, fz)
		# Retrive z from z' and x_v, then compute f(z)
		fz = self.compute_pred(node_index, num_samples, D, z_, feat_idx)
		
		# Weighted linear regression 
		# phi = self.WLR(z_, weights, fz)

		# OLS estimator for weighted linear regression
		phi, base_value = self.OLS(z_, weights, fz)  # dim (M*num_classes)
		print('Base value', base_value[true_pred], 'for class ', true_pred.item())

		# Print some information
		if info:
			self.print_info(D, node_index, phi, feat_idx, true_pred, true_conf)

		# Visualisation
			self.vizu(edge_mask, node_index, phi, true_pred, hops)
		
		# Time
		end = time.time()
		print('Time: ', end - start)

		return phi

	def coalition_sampler(self, num_samples):
		""" Sample coalitions cleverly given shapley kernel def

		Args:
			num_samples ([int]): total number of coalitions z_

		Returns:
			[tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
		"""
		z_ = torch.ones(num_samples, self.M)
		z_[1::2] = torch.zeros(num_samples//2, self.M)
		k = 1
		i = 2
		while i < num_samples:
			if i + 2 * self.M < num_samples and k == 1:
				z_[i:i+self.M, :] = torch.ones(self.M, self.M)
				z_[i:i+self.M, :].fill_diagonal_(0)
				z_[i+self.M:i+2*self.M, :] = torch.zeros(self.M, self.M)
				z_[i+self.M:i+2*self.M, :].fill_diagonal_(1)
				i += 2 * self.M
				k += 1
			elif k == 1:
				M = list(range(self.M))
				random.shuffle(M)
				for j in range(self.M):
					z_[i, M[j]] = torch.zeros(1)
					i += 1
					if i == num_samples:
						return z_
					z_[i, M[j]] = torch.ones(1)
					i += 1
					if i == num_samples:
						return z_
				k += 1
			elif k == 2:
				M = list(combinations(range(self.M), 2))[:num_samples-i+1]
				random.shuffle(M)
				for j in range(len(M)):
					z_[i, M[j][0]] = torch.tensor(0)
					z_[i, M[j][1]] = torch.tensor(0)
					i += 1
					if i == num_samples:
						return z_
					z_[i, M[j][0]] = torch.tensor(1)
					z_[i, M[j][1]] = torch.tensor(1)
					i += 1
					if i == num_samples:
						return z_
				k += 1
			else:
				z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
				return z_
			
		return z_

	def shapley_kernel(self, s):
		""" Computes a weight for each newly created sample 

		Args:
			s (tensor): contains dimension of z' for all instances
				(number of features + neighbours included)

		Returns:
				[tensor]: shapley kernel value for each sample
		"""
		shap_kernel = []
		# Loop around elements of s in order to specify a special case
		# Otherwise could have procedeed with tensor s direclty
		for i in range(s.shape[0]):
			a = s[i].item()
			# Put an emphasis on samples where all or none features are included
			if a == 0 or a == self.M:
				shap_kernel.append(10000)
			elif scipy.special.binom(self.M, a) == float('+inf'):
				shap_kernel.append(1)
			else:
				shap_kernel.append(
					(self.M-1)/(scipy.special.binom(self.M, a)*a*(self.M-a)))
		return torch.tensor(shap_kernel)


	def compute_pred(self, node_index, num_samples, D, z_, feat_idx):
		""" Construct z from z' and compute prediction f(z) for each sample z'
			In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

		Args: 
				Variables are defined exactly as defined in explainer function 

		Returns: 
				(tensor): f(z) - probability of belonging to each target classes, for all samples z
				Dimension (N * C) where N is num_samples and C num_classses. 
		"""
		# We need to recover z from z' - wrt sampled neighbours and node features
		# Initialise new node feature vectors and neighbours to disregard
		av_feat_values = list(self.data.x.mean(dim=0))
		# or random feature vector made of random value across each col of X 
		excluded_feat = {}
		excluded_nei = {}

		for i in range(num_samples):

			# Define new node features dataset (we only modify x_v for now)
			# Features where z_j == 1 are kept, others are set to 0
			feats_id = []
			for j in range(self.F):
				if z_[i, j].item() == 0:
					feats_id.append(feat_idx[j].item())
			excluded_feat[i] = feats_id

			# Define new neighbourhood
			# Store index of neighbours that need to be shut down (not sampled, z_j=0)
			nodes_id = []
			for j in range(D):
				if z_[i, self.F+j] == 0:
					nodes_id.append(self.neighbours[j].item())
			# Dico with key = num_sample id, value = excluded neighbour index
			excluded_nei[i] = nodes_id

		# Init label f(z) for graphshap dataset - consider all classes
		fz = torch.zeros((num_samples, self.data.num_classes))
		# Init final predicted class for each sample (informative)
		classes_labels = torch.zeros(num_samples)
		pred_confidence = torch.zeros(num_samples)

		# Create new matrix A and X - for each sample ≈ reform z from z'
		for (key, value), (_, value1)  in zip(excluded_nei.items(), excluded_feat.items()):

			positions = []
			# For each excluded neighbour, retrieve the column indices of its occurences
			# in the adj matrix - store them in positions (list)
			for val in value:
				pos = (self.data.edge_index == val).nonzero()[:, 1].tolist()
				positions += pos
			# Create new adjacency matrix for that sample
			positions = list(set(positions))
			A = np.array(self.data.edge_index)
			A = np.delete(A, positions, axis=1)
			A = torch.tensor(A)

			# Change feature vector for node of interest
			# NOTE: maybe change values of all nodes for features not inlcuded, not just x_v
			X = deepcopy(self.data.x)
			for val in value1:
				#X[:, val] = torch.tensor([av_feat_values[val]]*X.shape[0])
				X[self.neighbours, val] = torch.tensor([av_feat_values[val]]*D)  # 0
				X[node_index, val] = torch.tensor([av_feat_values[val]])#0

				
			# Apply model on (X,A) as input.
			with torch.no_grad():
				proba = self.model(x=X, edge_index=A).exp()[node_index]

			# Store final class prediction and confience level
			pred_confidence[key], classes_labels[key] = torch.topk(
				proba, k=1)  # optional
			# NOTE: maybe only consider predicted class for explanations

			# Store predicted class label in fz
			fz[key] = proba

		return fz
	
	def WLR(self, z_, weights, fz):
		"""Train a weighted linear regression

		Args:
			z_ (torch.tensor): data
			weights (torch.tensor): weights of each sample
			fz (torch.tensor): y data 
		"""
		# Define model 
		our_model = LinearRegressionModel(z_.shape[1], self.data.num_classes)

		# Define optimizer and loss function
		def weighted_mse_loss(input, target, weight):
			return (weight * (input - target) ** 2).mean()

		criterion = torch.nn.MSELoss()
		optimizer = torch.optim.SGD(our_model.parameters(), lr=0.01)

		# Dataloader 
		train = torch.utils.data.TensorDataset(z_, fz)
		train_loader = torch.utils.data.DataLoader(train, batch_size=1)
		
		# Repeat for several epochs
		for epoch in range(100):

			av_loss = []
			#for x,y,w in zip(z_,fz, weights):
			for batch_idx, (dat, target) in enumerate(train_loader):
				x, y = Variable(dat), Variable(target)
			
				# Forward pass: Compute predicted y by passing x to the model 
				pred_y = our_model(x)

				# Compute loss
				#loss = weighted_mse_loss(pred_y, y, weights[batch_idx])
				loss = criterion(pred_y,y)
				
				# Zero gradients, perform a backward pass, and update the weights.
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				# Store batch loss
				av_loss.append(loss.item())
			print('av loss epoch: ', np.mean(av_loss))

		# Evaluate model
		our_model.eval()
		with torch.no_grad():
			pred = our_model(z_)  
		print('r2 score: ', r2_score(pred, fz, multioutput='variance_weighted'))
		print(r2_score(pred, fz, multioutput='raw_values'))

		phi, base_value = [param.T for _,param in our_model.named_parameters()]
		return phi.detach().numpy().astype('float64')


	def OLS(self, z_, weights, fz):
		""" Ordinary Least Squares Method, weighted
			Estimates shapely value coefficients

		Args:
			z_ (tensor): binary vector representing the new instance
			weights ([type]): shapley kernel weights for z'
			fz ([type]): prediction f(z) where z is a new instance - formed from z' and x

		Returns:
			[tensor]: estimated coefficients of our weighted linear regression - on (z', f(z))
			Dimension (M * num_classes)
		"""
		# Add constant term 
		z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)

		# WLS to estimate parameters 
		try:
			tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
		except np.linalg.LinAlgError:  # matrix not invertible
			print('WLS: Matrix not invertible')
			tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
			tmp = np.linalg.inv(tmp + np.diag(np.random.randn(tmp.shape[1])))
		phi = np.dot(tmp, np.dot(
			np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))
		return phi[:-1,:], phi[-1,:]

	def print_info(self, D, node_index, phi, feat_idx, true_pred, true_conf):
		"""
		Displays some information about explanations - for a better comprehension and audit
		"""

		# Print some information
		print('Explanations include {} node features and {} neighbours for this node\
		for {} classes'.format(self.F, D, self.data.num_classes))

		# Compare with true prediction of the model - see what class should truly be explained
		print('Prediction of orignal model is class {} with confidence {}, while label is {}'
			  .format(true_pred, true_conf, self.data.y[node_index]))

		# Isolate explanations for predicted class - explain model choices
		pred_explanation = phi[:, true_pred]
		# print('Explanation for the class predicted by the model:', pred_explanation)

		# Look at repartition of weights among neighbours and node features
		# Motivation for regularisation
		sum_feat = sum_nei = 0
		for i in range(len(pred_explanation)):
			if i < self.F:
				sum_feat += np.abs(pred_explanation[i])
			else:
				sum_nei += np.abs(pred_explanation[i])
		print('Total weights for node features: ', sum_feat)
		print('Total weights for neighbours: ', sum_nei)

		# Note we focus on explanation for class predicted by the model here, so there is a bias towards
		# positive weights in our explanations (proba is close to 1 everytime).
		# Alternative is to view a class at random or the second best class

		# Select most influential neighbours and/or features (+ or -)
		if self.F + D > 10:
			_, idxs = torch.topk(torch.from_numpy(np.abs(pred_explanation)), 6)
			vals = [pred_explanation[idx] for idx in idxs]
			influential_feat = {}
			influential_nei = {}
			for idx, val in zip(idxs, vals):
				if idx.item() < self.F:
					influential_feat[feat_idx[idx]] = val
				else:
					influential_nei[self.neighbours[idx-self.F]] = val
			print( 'Most influential features: ', len([(item[0].item(), item[1].item()) for item in list(influential_feat.items())]),
				  'and neighbours', len([(item[0].item(), item[1].item()) for item in list(influential_nei.items())]) )

		# Most influential features splitted bewteen neighbours and features
		if self.F > 5:
			_, idxs = torch.topk(torch.from_numpy(
				np.abs(pred_explanation[:self.F])), 3)
			vals = [pred_explanation[idx] for idx in idxs]
			influential_feat = {}
			for idx, val in zip(idxs, vals):
				influential_feat[feat_idx[idx]] = val
			print('Most influential features: ', [
				  (item[0].item(), item[1].item()) for item in list(influential_feat.items())])

		# Most influential features splitted bewteen neighbours and features
		if D > 5:
			_, idxs = torch.topk(torch.from_numpy(
				np.abs(pred_explanation[self.F:])), 3)
			vals = [pred_explanation[self.F + idx] for idx in idxs]
			influential_nei = {}
			for idx, val in zip(idxs, vals):
				influential_nei[self.neighbours[idx]] = val
			print('Most influential neighbours: ', [
				  (item[0].item(), item[1].item()) for item in list(influential_nei.items())])

	def vizu(self, edge_mask, node_index, phi, predicted_class, hops):
		""" Vizu of important nodes in subgraph around node_index

		Args:
			edge_mask ([type]): vector of size data.edge_index with False 
											if edge is not included in subgraph around node_index
			node_index ([type]): node of interest index
			phi ([type]): explanations for node of interest
			predicted_class ([type]): class predicted by model for node of interest 
			hops ([type]):  number of hops considered for subgraph around node of interest 
		"""
		# Replace False by 0, True by 1 in edge_mask
		mask = torch.zeros(self.data.edge_index.shape[1])
		for i, val in enumerate(edge_mask):
			if val.item() == True:
				mask[i] = 1

		# Identify one-hop neighbour
		one_hop_nei, _, _, _ = k_hop_subgraph(
						node_index, 1, self.data.edge_index, relabel_nodes=True,
						num_nodes=None)

		# Attribute phi to edges in subgraph bsed on the incident node phi value
		for i, nei in enumerate(self.neighbours):
			list_indexes = (self.data.edge_index[0, :] == nei).nonzero()
			for idx in list_indexes:
				# Remove importance of 1-hop neighbours to 2-hop nei.
				if nei in one_hop_nei:
					if self.data.edge_index[1, idx] in one_hop_nei:
						mask[idx] = phi[self.F + i, predicted_class]
					else:
						pass
				elif mask[idx] == 1:
					mask[idx] = phi[self.F + i, predicted_class]
			#mask[mask.nonzero()[i].item()]=phi[i, predicted_class]

		# Set to 0 importance of edges related to 0
		mask[mask == 1] = 0

		# Increase coef for visibility and consider absolute contribution
		mask = torch.abs(mask)

		# Vizu nodes 
		ax, G = visualize_subgraph(self.model,
								   node_index,
								   self.data.edge_index,
								   mask,
								   hops,
								   y=self.data.y,
								   threshold=None)
		
		plt.savefig('results/GS1_{}_{}_{}'.format(self.data.name,
                                           self.model.__class__.__name__,
                                           node_index),
                    bbox_inches='tight')

		# Other visualisation 
		G = denoise_graph(self.data, mask, phi[self.F:,predicted_class], self.neighbours, node_index, feat=None, label=self.data.y, threshold_num=10)
		
		log_graph(G,
					identify_self=True,
					nodecolor="label",
					epoch=0,
					fig_size=(4, 3),
					dpi=300,
					label_node_feat=False,
					edge_vmax=None,
					args=None)

		plt.savefig('results/GS_{}_{}_{}'.format(self.data.name,
												 self.model.__class__.__name__,
												  node_index),
												  bbox_inches='tight')
		

		#plt.show()



class Greedy:

	def __init__(self, data, model):
		self.model = model
		self.data = data
		self.neighbours = None
		self.M = None
		self.F = None

		self.model.eval()

	def explain(self, node_index=0, hops=2, num_samples=0, info=True):
		"""
		Greedy explainer - only considers node features for explanations
		Computes the prediction proba with and without the targeted feature (repeat for all feat)
		This feature's importance is set as the normalised absolute difference in predictions above
		:param num_samples, info: useless here (simply to match GraphSHAP structure)
		"""
		# Create a variable to store node features
		x = self.data.x[node_index, :]
		# Number of non-zero entries for the feature vector x_v
		self.M = x[x != 0].shape[0]
		self.F = self.M
		# Store indexes of these non zero feature values
		feat_idx = x.nonzero()

		# Compute predictions
		with torch.no_grad():
			probas = self.model(x=self.data.x, edge_index=self.data.edge_index).exp()[
				node_index]
		pred_confidence, label = torch.topk(probas, k=1)

		# Init explanations vector
		coefs = np.zeros([self.M, self.data.num_classes])  # (m, #feats)
		#coef_pred_class = np.zeros(self.data.x.size(1))

		# Loop on all features - consider all classes
		for i, idx in enumerate(feat_idx):
			idx = idx.item()
			x_ = deepcopy(self.data.x)
			x_[:, idx] = 0.0  # set feat of interest to 0
			with torch.no_grad():
				probas_ = self.model(x=x_, edge_index=self.data.edge_index).exp()[
					node_index]  # [label].item()
			# Compute explanations with the following formula
			coefs[i] = (torch.abs(probas-probas_)/probas).detach().numpy()
			#coef_pred_class[i] = (torch.abs(
			#	pred_confidence - probas_[label].item()) / pred_confidence).detach().numpy()

		return coefs  # , coef_pred_class

	def explain_nei(self, node_index=0, hops=2, num_samples=0, info=True):
		# Create a variable to store node features
		x = self.data.x[node_index, :]

		# Construct k hop subgraph of node of interest (denoted v)
		neighbours, _, _, edge_mask =\
			torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
												 num_hops=hops,
												 edge_index=self.data.edge_index)
		# Store the indexes of the neighbours of v (+ index of v itself)

		# Remove node v index from neighbours and store their number in D
		neighbours = neighbours[neighbours != node_index]
		self.neighbours = neighbours
		self.M = neighbours.shape[0]

		# Compute predictions
		with torch.no_grad():
			probas = self.model(x=self.data.x, edge_index=self.data.edge_index).exp()[
				node_index]
		pred_confidence, label = torch.topk(probas, k=1)

		# Init explanations vector
		coefs = np.zeros([self.M, self.data.num_classes])  # (m, #feats)
		#coef_pred_class = np.zeros(self.data.x.size(1))

		# Loop on all neighbours - consider all classes
		for i, nei_idx in enumerate(self.neighbours):
			nei_idx = nei_idx.item()
			A_ = deepcopy(self.data.edge_index)

			# Find all edges incident to the isolated neighbour
			pos = (self.data.edge_index == nei_idx).nonzero()[:, 1].tolist()

			# Create new adjacency matrix where this neighbour is isolated
			A_ = np.array(self.data.edge_index)
			A_ = np.delete(A_, pos, axis=1)
			A_ = torch.tensor(A_)

			# Compute new prediction with updated adj matrix (without this neighbour)
			with torch.no_grad():
				probas_ = self.model(x=self.data.x, edge_index=A_).exp()[
					node_index]  # [label].item()

			# Compute explanations with the following formula
			coefs[i] = (torch.abs(probas-probas_)/probas).detach().numpy()
			#coef_pred_class[i] = (torch.abs(
			#	pred_confidence - probas_[label].item()) / pred_confidence).detach().numpy()

		return coefs


class Random:

	def __init__(self, num_feats, K):
		self.num_feats = num_feats
		self.K = K

	def explain(self):
		return np.random.choice(self.num_feats, self.K)


class GraphLIME:

	def __init__(self, data, model, hop=2, rho=0.1, cached=True):
		self.data = data
		self.model = model
		self.hop = hop
		self.rho = rho
		self.cached = cached
		self.cached_result = None
		self.M = data.x.size(1)
		self.F = data.x.size(1)

		self.model.eval()

	def __flow__(self):
		for module in self.model.modules():
			if isinstance(module, MessagePassing):
				return module.flow
		return 'source_to_target'

	def __subgraph__(self, node_idx, x, y, edge_index, **kwargs):
		num_nodes, num_edges = x.size(0), edge_index.size(1)

		subset, edge_index, mapping, edge_mask = k_hop_subgraph(
			node_idx, self.hop, edge_index, relabel_nodes=True,
			num_nodes=num_nodes, flow=self.__flow__())

		x = x[subset]
		y = y[subset]

		for key, item in kwargs:
			if torch.is_tensor(item) and item.size(0) == num_nodes:
				item = item[subset]
			elif torch.is_tensor(item) and item.size(0) == num_edges:
				item = item[edge_mask]
			kwargs[key] = item

		return x, y, edge_index, mapping, edge_mask, kwargs

	def __init_predict__(self, x, edge_index, **kwargs):
		if self.cached and self.cached_result is not None:
			if x.size(0) != self.cached_result.size(0):
				raise RuntimeError(
					'Cached {} number of nodes, but found {}.'.format(
						x.size(0), self.cached_result.size(0)))

		if not self.cached or self.cached_result is None:
			# Get the initial prediction.
			with torch.no_grad():
				log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
				probas = log_logits.exp()

			self.cached_result = probas

		return self.cached_result

	def __compute_kernel__(self, x, reduce):
		assert x.ndim == 2, x.shape

		n, d = x.shape

		dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
		dist = dist ** 2

		if reduce:
			dist = np.sum(dist, axis=-1, keepdims=True)  # (n, n, 1)

		std = np.sqrt(d)

		# (n, n, 1) or (n, n, d)
		K = np.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))

		return K

	def __compute_gram_matrix__(self, x):
		# unstable implementation due to matrix product (HxH)
		# n = x.shape[0]
		# H = np.eye(n, dtype=np.float) - 1.0 / n * np.ones(n, dtype=np.float)
		# G = np.dot(np.dot(H, x), H)

		# more stable and accurate implementation
		G = x - np.mean(x, axis=0, keepdims=True)
		G = G - np.mean(G, axis=1, keepdims=True)

		G = G / (np.linalg.norm(G, ord='fro', axis=(0, 1), keepdims=True) + 1e-10)

		return G

	def explain(self, node_index, hops, num_samples, info=False, **kwargs):
		# hops, num_samples, info are useless: just to copy graphshap pipeline
		x = self.data.x
		edge_index = self.data.edge_index

		probas = self.__init_predict__(x, edge_index, **kwargs)

		x, probas, _, _, _, _ = self.__subgraph__(
			node_index, x, probas, edge_index, **kwargs)

		x = x.detach().numpy()  # (n, d)
		y = probas.detach().numpy()  # (n, classes)

		n, d = x.shape

		K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
		L = self.__compute_kernel__(y, reduce=False)  # (n, n, 1)

		K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
		L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

		K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
		L_bar = L_bar.reshape(n ** 2, self.data.num_classes)  # (n ** 2,) 

		solver = LassoLars(self.rho, fit_intercept=False,
						   normalize=False, positive=True)
		solver.fit(K_bar * n, L_bar * n)

		return solver.coef_.T


class LIME:

	def __init__(self, data, model, cached=True):
		self.data = data
		self.model = model
		self.M = data.x.size(1)
		self.F = data.x.size(1)

		self.model.eval()

	def __init_predict__(self, x, edge_index, **kwargs):
		
		# Get the initial prediction.
		with torch.no_grad():
			log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
			probas = log_logits.exp()

		return probas

	def explain(self, node_index, hops, num_samples, info=False, **kwargs):
		x = self.data.x
		edge_index = self.data.edge_index

		probas = self.__init_predict__(x, edge_index, **kwargs)
		proba, label = probas[node_index, :].max(dim=0)

		x_ = deepcopy(x)
		original_feats = x[node_index, :]

		sample_x = [original_feats.detach().numpy()]
		#sample_y = [proba.item()]
		sample_y = [probas[node_index, :].detach().numpy()]

		for _ in range(num_samples):
			x_[node_index, :] = original_feats + \
				torch.randn_like(original_feats)

			with torch.no_grad():
				log_logits = self.model(x=x_, edge_index=edge_index, **kwargs)
				probas_ = log_logits.exp()

			#proba_ = probas_[node_index, label]
			proba_ = probas_[node_index]

			sample_x.append(x_[node_index, :].detach().numpy())
			# sample_y.append(proba_.item())
			sample_y.append(proba_.detach().numpy())

		sample_x = np.array(sample_x)
		sample_y = np.array(sample_y)

		solver = Ridge(alpha=0.1)
		solver.fit(sample_x, sample_y)

		return solver.coef_.T


class SHAP():

	def __init__(self, data, model):
		self.model = model
		self.data = data
		self.M = None  # number of nonzero features - for each node index
		self.neighbours = None
		self.F = None

		self.model.eval()

	def explain(self, node_index=0, hops=2, num_samples=10, info=True):
		"""
		:param node_index: index of the node of interest
		:param hops: number k of k-hop neighbours to consider in the subgraph around node_index
		:param num_samples: number of samples we want to form GraphSHAP's new dataset 
		:return: shapley values for features that influence node v's pred
		"""

		# Determine z' => features and neighbours whose importance is investigated

		# Create a variable to store node features
		x = self.data.x[node_index, :]

		# Number of non-zero entries for the feature vector x_v
		F = x[x != 0].shape[0]
		self.F = F
		# Store indexes of these non zero feature values
		feat_idx = torch.nonzero(x)

		# Total number of features + neighbours considered for node v
		self.M = F

		# Sample z' - binary vector of dimension (num_samples, M)
		# F node features first, then D neighbours
		z_ = torch.empty(num_samples, self.M).random_(2)
		# Compute |z'| for each sample z'
		s = (z_ != 0).sum(dim=1)

		# Define weights associated with each sample using shapley kernel formula
		weights = self.shapley_kernel(s)

		# Create dataset (z', f(z)), stored as (z_, fz)
		# Retrive z from z' and x_v, then compute f(z)
		fz = self.compute_pred(node_index, num_samples, F, z_, feat_idx)

		# OLS estimator for weighted linear regression
		phi = self.OLS(z_, weights, fz)  # dim (M*num_classes)

		# Visualisation
		# Call visu function
		# Pass it true_pred

		return phi

	def shapley_kernel(self, s):
		"""
		:param s: dimension of z' (number of features + neighbours included)
		:return: [scalar] value of shapley value 
		"""
		shap_kernel = []
		# Loop around elements of s in order to specify a special case
		# Otherwise could have procedeed with tensor s direclty
		for i in range(s.shape[0]):
			a = s[i].item()
			# Put an emphasis on samples where all or none features are included
			if a == 0 or a == self.M:
				shap_kernel.append(1000)
			elif scipy.special.binom(self.M, a) == float('+inf'):
				shap_kernel.append(1)
			else:
				shap_kernel.append(
					(self.M-1)/(scipy.special.binom(self.M, a)*a*(self.M-a)))
		return torch.tensor(shap_kernel)

	def compute_pred(self, node_index, num_samples, F, z_, feat_idx):
		"""
		Variables are exactly as defined in explainer function, where compute_pred is used
		This function aims to construct z (from z' and x_v) and then to compute f(z), 
		meaning the prediction of the new instances with our original model. 
		In fact, it builds the dataset (z', f(z)), required to train the weighted linear model.
		:return fz: probability of belonging to each target classes, for all samples z
		fz is of dimension N*C where N is num_samples and C num_classses. 
		"""
		# This implies retrieving z from z' - wrt sampled neighbours and node features
		# We start this process here by storing new node features for v and neigbours to
		# isolate
		X_v = torch.zeros([num_samples, self.data.num_features])

		# Feature matrix
		A = np.array(self.data.edge_index)
		A = torch.tensor(A)

		# Init label f(z) for graphshap dataset - consider all classes
		fz = torch.zeros((num_samples, self.data.num_classes))
		# Init final predicted class for each sample (informative)
		classes_labels = torch.zeros(num_samples)
		pred_confidence = torch.zeros(num_samples)

		# Do it for each sample
		for i in range(num_samples):

			# Define new node features dataset (we only modify x_v for now)
			# Features where z_j == 1 are kept, others are set to 0
			for j in range(F):
				if z_[i, j].item() == 1:
					X_v[i, feat_idx[j].item()] = 1

			# Change feature vector for node of interest
			# NOTE: maybe change values of all nodes for features not inlcuded, not just x_v
			X = deepcopy(self.data.x)
			X[node_index, :] = X_v[i, :]

			# Apply model on (X,A) as input.
			with torch.no_grad():
				proba = self.model(x=X, edge_index=A).exp()[node_index]

			# Store final class prediction and confience level
			# pred_confidence[i], classes_labels[i] = torch.topk(proba, k=1) # optional
			# NOTE: maybe only consider predicted class for explanations

			# Store predicted class label in fz
			fz[i] = proba

		return fz

	def OLS(self, z_, weights, fz):
		"""
		:param z_: z' - binary vector  
		:param weights: shapley kernel weights for z'
		:param fz: f(z) where z is a new instance - formed from z' and x
		:return: estimated coefficients of our weighted linear regression - on (z', f(z))
		phi is of dimension (M * num_classes)
		"""
		# OLS to estimate parameter of Weighted Linear Regression
		try:
			tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
		except np.linalg.LinAlgError:  # matrix not invertible
			tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
			tmp = np.linalg.inv(tmp + np.diag(np.random.randn(tmp.shape[1])))
		phi = np.dot(tmp, np.dot(
			np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))

		return phi


class GNNExplainer():

	def __init__(self, data, model):
		self.data = data
		self.model = model
		self.M = data.x.size(0) + data.x.size(1)
		# self.coefs = torch.zeros(data.x.size(0), self.data.num_classes)
		self.coefs = None  # node importance derived from edge importance
		self.edge_mask = None
		self.neighbours = None
		self.F = data.x.size(1)

		self.model.eval()

	def explain(self, node_index, hops, num_samples, info=False):
		# Use GNNE open source implem - outputs features's and edges importance
		explainer = GNNE(self.model, epochs=num_samples)
		node_feat_mask, self.edge_mask = explainer.explain_node(
			node_index, self.data.x, self.data.edge_index)

		# Transfer edge importance to node importance
		dico = {}
		for idx in torch.nonzero(self.edge_mask):
			node = self.data.edge_index[0, idx].item()
			if not node in dico.keys():
				dico[node] = [self.edge_mask[idx]]
			else:
				dico[node].append(self.edge_mask[idx])
		# Count neighbours in the subgraph
		self.neighbours = torch.tensor([index for index in dico.keys()])
		# Attribute an importance measure to each node = sum of incident edges' importance
		self.coefs = torch.zeros(
			self.neighbours.shape[0], self.data.num_classes)
		# for key, val in dico.items():
		for i, val in enumerate(dico.values()):
			#self.coefs[key,:] = sum(val)
			self.coefs[i, :] = sum(val)

		# Eliminate node_index from neighbourhood
		self.neighbours = self.neighbours[self.neighbours != node_index]
		self.coefs = self.coefs[1:]

		if info == True:
			self.vizu(self.edge_mask, node_index, self.coefs[0], hops)

		return torch.stack([node_feat_mask]*self.data.num_classes, 1)

	def vizu(self, edge_mask, node_index, phi, hops):
		"""
		Visualize the importance of neighbours in the subgraph of node_index
		"""
		# Replace False by 0, True by 1
		mask = torch.zeros(self.data.edge_index.shape[1])
		for i, val in enumerate(edge_mask):
			if val.item() != 0:
				mask[i] = 1

		# Attribute phi to edges in subgraph bsed on the incident node phi value
		for i, nei in enumerate(self.neighbours):
			list_indexes = (self.data.edge_index[0, :] == nei).nonzero()
			for idx in list_indexes:
				if self.data.edge_index[1, idx] == node_index:
					mask[idx] = edge_mask[idx]
					break
				elif mask[idx] != 0:
					mask[idx] = edge_mask[idx]

		# Set to 0 importance of edges related to 0
		mask[mask == 1] = 0

		# Vizu nodes and
		ax, G = visualize_subgraph(self.model,
								   node_index,
								   self.data.edge_index,
								   mask,
								   hops,
								   y=self.data.y,
								   threshold=None)

		plt.savefig('results/GS1_{}_{}_{}'.format(self.data.name,
                                            self.model.__class__.__name__,
                                            node_index),
                    bbox_inches='tight')
		#plt.show()
