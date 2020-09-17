# Import libraries 
import scipy.special
import numpy as np
from copy import deepcopy
import torch_geometric
import torch
import tqdm
import matplotlib.pyplot as plt

# GraphLIME
from src.utils import k_hop_subgraph
from src.plots import visualize_subgraph
from torch_geometric.nn import MessagePassing
from sklearn.linear_model import Ridge, LassoLars

# GNNExplainer
from torch_geometric.nn import GNNExplainer as GNNE



class GraphSHAP():

	def __init__(self, data, model):
		self.model = model
		self.data = data
		self.model.eval()
		self.M = None # number of nonzero features - for each node index
		self.neighbors = None

	def explain(self, node_index=0, hops=2, num_samples=10, info=True):
		"""
		:param node_index: index of the node of interest
		:param hops: number k of k-hop neighbours to consider in the subgraph around node_index
		:param num_samples: number of samples we want to form GraphSHAP's new dataset 
		:return: shapley values for features/neighbours that influence node v's pred
		"""

		### Determine z' => features and neighbours whose importance is investigated

		# Create a variable to store node features 
		x = self.data.x[node_index,:]

		# Number of non-zero entries for the feature vector x_v
		F = x[x!=0].shape[0]
		# Store indexes of these non zero feature values
		feat_idx = torch.nonzero(x)

		# Construct k hop subgraph of node of interest (denoted v)
		neighbours, _, _, edge_mask =\
			 torch_geometric.utils.k_hop_subgraph(node_idx=node_index, 
												num_hops=hops, 
												edge_index= self.data.edge_index)
		# Store the indexes of the neighbours of v (+ index of v itself)

		# Remove node v index from neighbours and store their number in D
		neighbours = neighbours[neighbours!=node_index]
		self.neighbors = neighbours
		D = neighbours.shape[0] 

		# Total number of features + neighbours considered for node v
		self.M = F+D

		# Sample z' - binary vector of dimension (num_samples, M)
		# F node features first, then D neighbors
		z_ = torch.empty(num_samples, self.M).random_(2)
		# Compute |z'| for each sample z'
		s = (z_ != 0).sum(dim=1)
		
		# Compute true prediction of model, for original instance
		true_conf, true_pred  = self.model(x=self.data.x, edge_index=self.data.edge_index).exp()[node_index].max(dim=0)

		### Define weights associated with each sample using shapley kernel formula
		weights = self.shapley_kernel(s)

		###  Create dataset (z', f(z)), stored as (z_, fz)
		# Retrive z from z' and x_v, then compute f(z)
		fz = self.compute_pred(node_index, num_samples, F, D, z_, neighbours, feat_idx)
		
		### OLS estimator for weighted linear regression
		phi = self.OLS(z_, weights, fz) # dim (M*num_classes)
		
		### Print some information 
		if info:
			self.print_info(F, D, node_index, phi, feat_idx, neighbours)

		### Visualisation 
			self.vizu(edge_mask, node_index, neighbours, phi, true_pred, hops)

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
			elif scipy.special.binom(self.M,a) == float('+inf'):
				shap_kernel.append(1)
			else: 
				shap_kernel.append((self.M-1)/(scipy.special.binom(self.M,a)*a*(self.M-a)))
		return torch.tensor(shap_kernel)


	def compute_pred(self, node_index, num_samples, F, D, z_, neighbours, feat_idx):
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
		fz = torch.zeros((num_samples, self.data.num_classes))
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
			# NOTE: maybe only consider predicted class for explanations

			# Store predicted class label in fz 
			fz[key] = proba

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
		except np.linalg.LinAlgError: # matrix not invertible
			tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_) 
			tmp = np.linalg.inv(tmp + np.diag(np.random.randn(tmp.shape[1]))) 
		phi = np.dot(tmp, np.dot(np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))
		return phi


	def print_info(self, F, D, node_index, phi, feat_idx, neighbour):
		"""
		Displays some information about explanations - for a better comprehension and audit
		"""

		# Print some information
		print('Explanations include {} node features and {} neighbours for this node\
		for {} classes'.format(F, D, self.data.num_classes))
		
		# Compare with true prediction of the model - see what class should truly be explained
		true_conf, true_pred  = self.model(x=self.data.x, edge_index=self.data.edge_index).exp()[node_index].max(dim=0)
		print('Prediction of orignal model is class {} with confidence {}'.format(true_pred, true_conf))

		# Isolate explanations for predicted class - explain model choices
		pred_explanation = phi[:,true_pred]
		print('Explanation for the class predicted by the model:', pred_explanation)

		# Look at repartition of weights among neighbours and node features
		# Motivation for regularisation 
		sum_feat = sum_nei = 0 
		for i in range(len(pred_explanation)):
			if i < F: 
				sum_feat += np.abs(pred_explanation[i])
			else: 
				sum_nei += np.abs(pred_explanation[i])
		print('Total weights for node features: ', sum_feat)
		print('Total weights for neighbours: ', sum_nei)

		# Note we focus on explanation for class predicted by the model here, so there is a bias towards 
		# positive weights in our explanations (proba is close to 1 everytime). 
		# Alternative is to view a class at random or the second best class 

		# Select most influential neighbors and/or features (+ or -)
		if F + D > 10: 
			_, idxs = torch.topk(torch.from_numpy(np.abs(pred_explanation)), 6)
			vals = [pred_explanation[idx] for idx in idxs]
			influential_feat = {}
			influential_nei = {}
			for idx, val in zip(idxs, vals): 
				if idx.item() < F: 
					influential_feat[feat_idx[idx]] = val
				else: 
					influential_nei[neighbour[idx-F]] = val
			print('Most influential features: ', [(item[0].item(),item[1].item()) for item in list(influential_feat.items())], 
			'and neighbours', [(item[0].item(),item[1].item()) for item in list(influential_nei.items())])

		# Most influential features splitted bewteen neighbours and features
		if F > 5:
			_, idxs = torch.topk(torch.from_numpy(np.abs(pred_explanation[:F])), 3)
			vals = [pred_explanation[idx] for idx in idxs]
			influential_feat = {}
			for idx, val in zip(idxs, vals): 
				influential_feat[feat_idx[idx]] = val
			print('Most influential features: ', [(item[0].item(),item[1].item()) for item in list(influential_feat.items())] )

		# Most influential features splitted bewteen neighbours and features
		if D > 5: 
			_, idxs = torch.topk(torch.from_numpy(np.abs(pred_explanation[F:])), 3)
			vals = [pred_explanation[F + idx] for idx in idxs]
			influential_nei = {}
			for idx, val in zip(idxs, vals): 
				influential_nei[neighbour[idx]] = val
			print('Most influential neighbours: ', [(item[0].item(),item[1].item()) for item in list(influential_nei.items())] )
			


	def vizu(self, edge_mask, node_index, neighbours, phi, predicted_class, hops):
		"""
		:param true_pred: class predicted by original model for node of interest
		:param phi: shapley values
		:param feat_idx: index of features whose importance is assessed
		:param neighbours: index of nodes whose importance is assessed
		:return: nice visualisation (like SHAP) of each feature/neigbour's average
		marginal contribution towards the prediction 
		"""
		# Replace False by 0, True by 1
		mask = torch.zeros(self.data.edge_index.shape[1])
		for i, val in enumerate(edge_mask):
			if val.item()==True:
				mask[i]=1
				
		# Attribute phi to edges in subgraph bsed on the incident node phi value
		for i, nei in enumerate(neighbours): 
			list_indexes = (self.data.edge_index[0,:]==nei).nonzero()
			for idx in list_indexes: 
				if self.data.edge_index[1,idx]==0:
					mask[idx] = phi[self.M - len(neighbours) + i, predicted_class] 
					break
				elif mask[idx]==1:
					mask[idx] = phi[self.M - len(neighbours) + i, predicted_class] 
			#mask[mask.nonzero()[i].item()]=phi[i, predicted_class]
		
		# Set to 0 importance of edges related to 0 
		mask[mask==1]=0

		# Increase coef for visibility and consider absolute contribution
		mask = torch.abs(mask)*2

		# Vizu nodes and 
		ax, G = visualize_subgraph(self.model, 
									node_index, 
									self.data.edge_index, 
									mask, 
									hops, 
									y=self.data.y, 
									threshold=None)
		plt.show()
		

		

class Greedy:

	def __init__(self, data, model):
		self.model = model
		self.data = data
		self.model.eval()
		self.neighbors = 0
		self.M = None

	def explain(self, node_index=0, hops=2, num_samples=0, info=True):
		"""
		Greedy explainer - only considers node features for explanations
		Computes the prediction proba with and without the targeted feature (repeat for all feat)
		This feature's importance is set as the normalised absolute difference in predictions above
		:param num_samples, info: useless here (simply to match GraphSHAP structure)
		"""
		# Create a variable to store node features 
		x = self.data.x[node_index,:]
		# Number of non-zero entries for the feature vector x_v
		self.M = x[x!=0].shape[0]
		# Store indexes of these non zero feature values
		feat_indexes = x.nonzero()

		### Compute predictions
		self.model.eval()		
		probas = self.model(x=self.data.x, edge_index=self.data.edge_index).exp()[node_index]
		pred_confidence, label = torch.topk(probas, k=1) 
		
		# Init explanations vector
		coefs = np.zeros([self.M, self.data.num_classes])  # (m, #feats)
		coef_pred_class = np.zeros(self.data.x.size(1))

		# Loop on all features - consider all classes
		for i, feat_idx in enumerate(feat_indexes):
			feat_idx = feat_idx.item()
			x_ = deepcopy(self.data.x)
			x_[:, feat_idx] = 0.0 # set feat of interest to 0 
			probas_ = self.model(x=x_, edge_index=self.data.edge_index).exp()[node_index] # [label].item()
			# Compute explanations with the following formula
			coefs[i] = (torch.abs(probas-probas_)/probas).detach().numpy()
			coef_pred_class[i] = (torch.abs(pred_confidence - probas_[label].item()) / pred_confidence).detach().numpy()

		return coefs #, coef_pred_class


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
			
		K = np.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))  # (n, n, 1) or (n, n, d)

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
		L_bar = L_bar.reshape(n ** 2, 7)  # (n ** 2,)

		solver = LassoLars(self.rho, fit_intercept=False, normalize=False, positive=True)
		solver.fit(K_bar * n, L_bar * n)

		return solver.coef_.T


class LIME:

	def __init__(self, data, model, cached=True):
		self.data = data
		self.model = model
		self.cached = cached
		self.cached_result = None
		self.M = data.x.size(1)

		self.model.eval()

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
			x_[node_index, :] = original_feats + torch.randn_like(original_feats)
			
			with torch.no_grad():
				log_logits = self.model(x=x_, edge_index=edge_index, **kwargs)
				probas_ = log_logits.exp()

			#proba_ = probas_[node_index, label]
			proba_ = probas_[node_index]

			sample_x.append(x_[node_index, :].detach().numpy())
			#sample_y.append(proba_.item())
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
		self.model.eval()
		self.M = None # number of nonzero features - for each node index
		self.neighbors = None

	def explain(self, node_index=0, hops=2, num_samples=10, info=True):
		"""
		:param node_index: index of the node of interest
		:param hops: number k of k-hop neighbours to consider in the subgraph around node_index
		:param num_samples: number of samples we want to form GraphSHAP's new dataset 
		:return: shapley values for features that influence node v's pred
		"""

		### Determine z' => features and neighbours whose importance is investigated

		# Create a variable to store node features 
		x = self.data.x[node_index,:]

		# Number of non-zero entries for the feature vector x_v
		F = x[x!=0].shape[0]
		# Store indexes of these non zero feature values
		feat_idx = torch.nonzero(x)

		# Total number of features + neighbours considered for node v
		self.M = F

		# Sample z' - binary vector of dimension (num_samples, M)
		# F node features first, then D neighbors
		z_ = torch.empty(num_samples, self.M).random_(2)
		# Compute |z'| for each sample z'
		s = (z_ != 0).sum(dim=1)


		### Define weights associated with each sample using shapley kernel formula
		weights = self.shapley_kernel(s)

		###  Create dataset (z', f(z)), stored as (z_, fz)
		# Retrive z from z' and x_v, then compute f(z)
		fz = self.compute_pred(node_index, num_samples, F, z_, feat_idx)
		
		### OLS estimator for weighted linear regression
		phi = self.OLS(z_, weights, fz) # dim (M*num_classes)

		### Visualisation 
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
			elif scipy.special.binom(self.M,a) == float('+inf'):
				shap_kernel.append(1)
			else: 
				shap_kernel.append((self.M-1)/(scipy.special.binom(self.M,a)*a*(self.M-a)))
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
		X_v = torch.zeros([num_samples,self.data.num_features])

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
				if z_[i,j].item()==1: 
					X_v[i,feat_idx[j].item()]=1

			# Change feature vector for node of interest 
			# NOTE: maybe change values of all nodes for features not inlcuded, not just x_v
			X = deepcopy(self.data.x)
			X[node_index,:] = X_v[i,:]

			# Apply model on (X,A) as input. 
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
		except np.linalg.LinAlgError: # matrix not invertible
			tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_) 
			tmp = np.linalg.inv(tmp + np.diag(np.random.randn(tmp.shape[1]))) 
		phi = np.dot(tmp, np.dot(np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))

		return phi


class GNNExplainer():

	def __init__(self, data, model):
		self.data = data
		self.model = model
		self.M = data.x.size(1)
		self.edge_mask = None

	def explain(self, node_index, hops, num_samples, info=False):
		explainer = GNNE(self.model, epochs=200)
		node_feat_mask, self.edge_mask = explainer.explain_node(node_index, self.data.x, self.data.edge_index)
	
		return torch.stack([node_feat_mask]*7, 1)