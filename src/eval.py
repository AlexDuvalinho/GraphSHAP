import matplotlib.pyplot as plt
from torch_geometric.nn import GNNExplainer as GNNE

import src.gengraph
from src.data import (add_noise_features, add_noise_neighbours,
					  extract_test_nodes, prepare_data)
from src.explainers import (LIME, SHAP, GNNExplainer, GraphLIME, GraphSHAP,
							Greedy, Random)
from src.models import GAT, GCN
from src.plots import plot_dist
from src.train import *
from src.utils import *

def filter_useless_features(args_model,
							args_dataset,
							args_explainers,
							args_hops,
							args_num_samples,
							args_test_samples,
							args_K,
							args_num_noise_feat,
							args_p,
							args_binary,
							node_indices,
							multiclass=True,
							info=True):
	"""
	Arguments defined in argument parser of script_eval.py
	Add noisy features to dataset and check how many are included in explanations
	The fewest, the better the explainer.
	"""
	'''
	####### Input in script_eval file
	args_dataset = 'Cora'
	args_model = 'GCN'
	args_explainers = ['GraphSHAP', 'Greedy', 'GraphLIME', 'LIME', 'GNNE']
	args_hops = 2
	args_num_samples = 100 # size shap dataset
	args_test_samples = 20 # number of test samples
	args_num_noise_feat= 25 # number of noisy features
	args_K= 5 # maybe def depending on M
	args_binary = True 
	args_p = 0.5
	info=True

	node_indices= [2420,2455,1783,2165,2628,1822,2682,2261,1896,1880,2137,2237,2313,2218,1822,1719,1763,2263,2020,1988]
	node_indices = [10, 18, 89, 178, 333, 356, 378, 456, 500, 2222, 1220, 1900, 1328, 189, 1111]
	node_indices = [1834,2512,2591,2101,1848,1853,2326,1987,2359,2453,2230,2267,2399, 2150,2400]
	'''

	# Define dataset - include noisy features
	data = prepare_data(args_dataset, seed=10)
	data, noise_feat = add_noise_features(
		data, num_noise=args_num_noise_feat, binary=args_binary, p=args_p)

	# Define training parameters depending on (model-dataset) couple
	hyperparam = ''.join(['hparams_', args_dataset, '_', args_model])
	param = ''.join(['params_', args_dataset, '_', args_model])

	# Define the model
	if args_model == 'GCN':
		model = GCN(input_dim=data.x.size(
			1), output_dim=data.num_classes, **eval(hyperparam))
	else:
		model = GAT(input_dim=data.x.size(
			1), output_dim=data.num_classes,  **eval(hyperparam))

	# Re-train the model on dataset with noisy features
	train_and_val(model, data, **eval(param))

	# Select random subset of nodes to eval the explainer on.
	if not node_indices:
		node_indices = extract_test_nodes(data, args_test_samples)

	# Loop on different explainers selected
	for c, explainer_name in enumerate(args_explainers):

		# Define explainer
		explainer = eval(explainer_name)(data, model)

		# count noisy features found in explanations for each test sample (for each class)
		total_num_noise_feats = []
		# count noisy features found in explanations for each test sample for class of interest
		pred_class_num_noise_feats = []
		# count number of noisy features considered for each test sample (for each class)
		total_num_noise_feat_considered = []
		F = []  # count number of non zero features for each test sample

		# Loop on each test sample and store how many times do noise features appear among
		# K most influential features in our explanations
		for node_idx in tqdm(node_indices, desc='explain node', leave=False):

			# Explanations via GraphSHAP
			coefs = explainer.explain(node_index=node_idx,
									  hops=args_hops,
									  num_samples=args_num_samples,
									  info=False)

			# Check how many non zero features
			F.append(explainer.F)

			# Number of non zero noisy features
			# Dfferent for explainers with all features considered vs non zero features only (shap,graphshap)
			# if explainer.F != data.x.size(1)
			if explainer_name == 'GraphSHAP' or explainer_name == 'SHAP':
				num_noise_feat_considered = len(
					[val for val in noise_feat[node_idx] if val != 0])
			else:
				num_noise_feat_considered = args_num_noise_feat

			# Multilabel classification - consider all classes instead of focusing on the
			# class that is predicted by our model
			num_noise_feats = []
			true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[
				node_idx].max(dim=0)

			for i in range(data.num_classes):

				# Store indexes of K most important node features, for each class
				feat_indices = np.abs(
					coefs[:explainer.F, i]).argsort()[-args_K:].tolist()

				# Number of noisy features that appear in explanations - use index to spot them
				num_noise_feat = sum(
					idx < num_noise_feat_considered for idx in feat_indices)
				num_noise_feats.append(num_noise_feat)

				# For predicted class only
				if i == predicted_class:
					pred_class_num_noise_feats.append(num_noise_feat)

			# Return number of times noisy features are provided as explanations
			total_num_noise_feats.append(sum(num_noise_feats))

			# Return number of noisy features considered in this test sample
			total_num_noise_feat_considered.append(num_noise_feat_considered)

		if info:
			print('Noise features included in explanations: ',
				  total_num_noise_feats)
			print('There are {} noise features found in the explanations of {} test samples, an average of {} per sample'
				  .format(sum(total_num_noise_feats), args_test_samples, sum(total_num_noise_feats)/args_test_samples))

			# Number of noisy features found in explanation for the predicted class
			print(np.sum(pred_class_num_noise_feats) /
				  args_test_samples, 'for the predicted class only')

			perc = 100 * sum(total_num_noise_feat_considered) / np.sum(F)
			print(
				'Proportion of non-zero noisy features among non-zero features: {:.2f}%'.format(perc))

			perc = 100 * sum(total_num_noise_feats) / \
				(args_K * args_test_samples * data.num_classes)
			print(
				'Proportion of explanations showing noisy features: {:.2f}%'.format(perc))

			if sum(total_num_noise_feat_considered) != 0:
				perc = 100 * sum(total_num_noise_feats) / \
					(sum(total_num_noise_feat_considered)*data.num_classes)
				perc2 = 100 * (args_K * args_test_samples * data.num_classes - sum(total_num_noise_feats)) / (
					data.num_classes * (sum(F) - sum(total_num_noise_feat_considered)))
				print('Proportion of noisy features found in explanations vs normal features (among considered ones): {:.2f}% vs {:.2f}%, over considered features only'.format(
					perc, perc2))

			print('------------------------------------')

		# Plot of kernel density estimates of number of noisy features included in explanation
		# Do for all benchmarks (with diff colors) and plt.show() to get on the same graph
		if multiclass:
			plot_dist(total_num_noise_feats,
					  label=explainer_name, color=COLOURS[c])
		else:  # consider only predicted class
			plot_dist(pred_class_num_noise_feats,
					  label=explainer_name, color=COLOURS[c])

	# Random explainer - plot estimated kernel density
	total_num_noise_feats = noise_feats_for_random(
		data, model, args_K, args_num_noise_feat, node_indices)
	plot_dist(total_num_noise_feats, label='Random', color='y')

	plt.show()
	return sum(total_num_noise_feats)


def noise_feats_for_random(data, model, args_K, args_num_noise_feat, node_indices):
	"""
	:param args_K: number of important features
	:param args_num_noise_feat: number of noisy features
	:param node_indices: indices of test nodes 
	Code to output indexes of node features 
	"""

	# Use Random explainer
	explainer = Random(data.x.size(1), args_K)

	# Loop on each test sample and store how many times do noise features appear among
	# K most influential features in our explanations
	total_num_noise_feats = []
	pred_class_num_noise_feats = []

	for node_idx in node_indices:
		num_noise_feats = []
		true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[
			node_idx].max(dim=0)

		for i in range(data.num_classes):
			# Store indexes of K most important features, for each class
			feat_indices = explainer.explain()

			# Number of noisy features that appear in explanations - use index to spot them
			num_noise_feat = sum(
				idx < args_num_noise_feat for idx in feat_indices)
			num_noise_feats.append(num_noise_feat)

			if i == predicted_class:
				pred_class_num_noise_feats.append(num_noise_feat)

		# Return this number => number of times noisy features are provided as explanations
		total_num_noise_feats.append(sum(num_noise_feats))

	#noise_feats = []
	# Do for each test sample
	# for node_idx in tqdm(range(args.test_samples), desc='explain node', leave=False):
	#	feat_indices = explainer.explain() # store indices of features provided as explanations
	#	noise_feat = (feat_indices >= INPUT_DIM[args.dataset]).sum() # check if they are noise features - not like this
	#	noise_feats.append(noise_feat)
	return total_num_noise_feats

###############################################################################


def filter_useless_nodes(args_model,
						 args_dataset,
						 args_explainers,
						 args_hops,
						 args_num_samples,
						 args_test_samples,
						 args_K,
						 args_num_noise_nodes,
						 args_p,
						 args_binary,
						 args_connectedness,
						 node_indices=None,
						 multiclass=True,
						 info=True):
	"""
	Arguments defined in argument parser in script_eval.py
	Add noisy nodes to dataset and check how many are included in explanations
	The fewest, the better the explainer.
	"""

	'''
	####### Input in script_eval file
	args_dataset = 'Cora'
	args_model = 'GAT'
	args_hops = 2
	args_num_samples = 100
	args_test_samples = 10
	args_num_noise_nodes = 20
	args_K= 5 # maybe def depending on M 
	args_p = 0.013
	args_connectedness = 'medium'
	args_binary=True
	'''

	# Define dataset
	data = prepare_data(args_dataset, seed=10)

	# Select a random subset of nodes to eval the explainer on.
	if not node_indices:
		node_indices = extract_test_nodes(data, args_test_samples)

	# Add noisy neighbours to the graph, with random features
	data = add_noise_neighbours(data, args_num_noise_nodes, node_indices,
								binary=args_binary, p=args_p, connectedness=args_connectedness)

	# Define training parameters depending on (model-dataset) couple
	hyperparam = ''.join(['hparams_', args_dataset, '_', args_model])
	param = ''.join(['params_', args_dataset, '_', args_model])

	# Define the model
	if args_model == 'GCN':
		model = GCN(input_dim=data.x.size(
			1), output_dim=data.num_classes, **eval(hyperparam))
	else:
		model = GAT(input_dim=data.x.size(
			1), output_dim=data.num_classes, **eval(hyperparam))

	# Re-train the model on dataset with noisy features
	train_and_val(model, data, **eval(param))

	# Study attention weights of noisy nodes in GAT model - compare attention with explanations
	if str(type(model)) == "<class 'src.models.GAT'>":
		study_attention_weights(data, model, args_test_samples)

	# Do for several explainers
	for c, explainer_name in enumerate(args_explainers):

		# Define the explainer
		explainer = eval(explainer_name)(data, model)

		# Loop on each test sample and store how many times do noisy nodes appear among
		# K most influential features in our explanations
		# 1 el per test sample - count number of noisy nodes in explanations
		total_num_noise_neis = []
		# 1 el per test sample - count number of noisy nodes in explanations for 1 class
		pred_class_num_noise_neis = []
		# 1 el per test sample - count number of noisy nodes in subgraph
		total_num_noisy_nei = []
		total_neigbours = []  # 1 el per test samples - number of neigbours of v in subgraph
		M = []  # 1 el per test sample - number of non zero features
		for node_idx in tqdm(node_indices, desc='explain node', leave=False):

			# Look only at coefficients for nodes (not node features)
			if explainer_name == 'Greedy':
				coefs = explainer.explain_nei(node_index=node_idx,
											  hops=args_hops,
											  num_samples=args_num_samples,
											  info=False)

			elif explainer_name == 'GNNExplainer':
				_ = explainer.explain(node_index=node_idx,
									  hops=args_hops,
									  num_samples=args_num_samples,
									  info=False)
				coefs = explainer.coefs

			else:
				# Explanations via GraphSHAP
				coefs = explainer.explain(node_index=node_idx,
										  hops=args_hops,
										  num_samples=args_num_samples,
										  info=False)
				coefs = coefs[explainer.F:]

			# Check how many non zero features
			M.append(explainer.M)

			# Number of noisy nodes in the subgraph of node_idx
			num_noisy_nodes = len(
				[n_idx for n_idx in explainer.neighbours if n_idx >= data.x.size(0)-args_num_noise_nodes])

			# Number of neighbours in the subgraph
			total_neigbours.append(len(explainer.neighbours))

			# Multilabel classification - consider all classes instead of focusing on the
			# class that is predicted by our model
			num_noise_neis = []  # one element for each class of a test sample
			true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[
				node_idx].max(dim=0)

			for i in range(data.num_classes):

				# Store indexes of K most important features, for each class
				nei_indices = np.abs(coefs[:, i]).argsort()[-args_K:].tolist()

				# Number of noisy features that appear in explanations - use index to spot them
				num_noise_nei = sum(
					idx >= (explainer.neighbours.shape[0] - num_noisy_nodes) for idx in nei_indices)
				num_noise_neis.append(num_noise_nei)

				if i == predicted_class:
					#nei_indices = coefs[:,i].argsort()[-args_K:].tolist()
					#num_noise_nei = sum(idx >= (explainer.neighbours.shape[0] - num_noisy_nodes) for idx in nei_indices)
					pred_class_num_noise_neis.append(num_noise_nei)

			# Return this number => number of times noisy neighbours are provided as explanations
			total_num_noise_neis.append(sum(num_noise_neis))
			# Return number of noisy nodes adjacent to node of interest
			total_num_noisy_nei.append(num_noisy_nodes)

		if info:
			print('Noisy neighbours included in explanations: ',
				  total_num_noise_neis)

			print('There are {} noise neighbours found in the explanations of {} test samples, an average of {} per sample'
				  .format(sum(total_num_noise_neis), args_test_samples, sum(total_num_noise_neis)/args_test_samples))

			print(np.sum(pred_class_num_noise_neis) /
				  args_test_samples, 'for the predicted class only')

			print('Proportion of explanations showing noisy neighbours: {:.2f}%'.format(
				100 * sum(total_num_noise_neis) / (args_K * args_test_samples * data.num_classes)))

			perc = 100 * sum(total_num_noise_neis) / (args_test_samples *
													  args_num_noise_nodes * data.num_classes)
			perc2 = 100 * ((args_K * args_test_samples * data.num_classes) -
						   sum(total_num_noise_neis)) / (np.sum(M) - sum(total_num_noisy_nei))
			print('Proportion of noisy neighbours found in explanations vs normal features: {:.2f}% vs {:.2f}'.format(
				perc, perc2))

			print('Proportion of nodes in subgraph that are noisy: {:.2f}%'.format(
				100 * sum(total_num_noisy_nei) / sum(total_neigbours)))

			print('Proportion of noisy neighbours in subgraph found in explanations: {:.2f}%'.format(
				100 * sum(total_num_noise_neis) / (sum(total_num_noisy_nei) * data.num_classes)))

		# Plot of kernel density estimates of number of noisy features included in explanation
		# Do for all benchmarks (with diff colors) and plt.show() to get on the same graph
		if multiclass:
			plot_dist(total_num_noise_neis,
					  label=explainer_name, color=COLOURS[c])
		else:  # consider only predicted class
			plot_dist(pred_class_num_noise_neis,
					  label=explainer_name, color=COLOURS[c])

	# Random explainer - plot estimated kernel density
	total_num_noise_neis = noise_nodes_for_random(
		data, model, args_K, args_num_noise_nodes, node_indices)
	plot_dist(total_num_noise_neis, label='Random', color='y')

	plt.show()
	return total_num_noise_neis


def noise_nodes_for_random(data, model, args_K, args_num_noise_nodes, node_indices):
	"""
	Code to output indexes of node features 
	"""

	# Use Random explainer - on neighbours (not features)
	explainer = Random(data.x.size(0), args_K)

	# Store number of noisy neighbours found in explanations (for all classes and predicted class)
	total_num_noise_neis = []
	pred_class_num_noise_neis = []

	# Check how many noisy neighbours are included in top-K explanations
	for node_idx in node_indices:
		num_noise_neis = []
		true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[
			node_idx].max(dim=0)

		for i in range(data.num_classes):
			# Store indexes of K most important features, for each class
			nei_indices = explainer.explain()

			# Number of noisy features that appear in explanations - use index to spot them
			num_noise_nei = sum(
				idx > (data.x.size(0)-args_num_noise_nodes) for idx in nei_indices)
			num_noise_neis.append(num_noise_nei)

			if i == predicted_class:
				pred_class_num_noise_neis.append(num_noise_nei)

		# Return this number => number of times noisy features are provided as explanations
		total_num_noise_neis.append(sum(num_noise_neis))

	#noise_feats = []
	# Do for each test sample
	# for node_idx in tqdm(range(args.test_samples), desc='explain node', leave=False):
	#	feat_indices = explainer.explain() # store indices of features provided as explanations
	#	noise_feat = (feat_indices >= INPUT_DIM[args.dataset]).sum() # check if they are noise features - not like this
	#	noise_feats.append(noise_feat)
	return total_num_noise_neis


def study_attention_weights(data, model, args_test_samples):
	"""
		Studies the attention weights of the GAT model
		"""
	_, alpha, alpha_bis = model(data.x, data.edge_index, att=True)

	# remove self loops att
	edges, alpha1 = alpha[0][:, :-
                          (data.x.size(0))], alpha[1][:-(data.x.size(0)), :]
	alpha2 = alpha_bis[1][:-(data.x.size(0))]

	# Look at all importance coefficients of noisy nodes towards normal nodes
	att1 = []
	att2 = []
	for i in range(data.x.size(0) - args_test_samples, (data.x.size(0)-1)):
		ind = (edges == i).nonzero()
		for j in ind[:, 1]:
			att1.append(torch.mean(alpha1[j]))
			att2.append(alpha2[j][0])
	print('shape attention noisy', len(att2))

	# It looks like these noisy nodes are very important
	print('av attention',  (torch.mean(alpha1) + torch.mean(alpha2))/2)  # 0.18
	(torch.mean(torch.stack(att1)) + torch.mean(torch.stack(att2)))/2  # 0.32

	# In fact, noisy nodes are slightly below average in terms of attention received
	# Importance of interest: look only at imp. of noisy nei for test nodes
	print('attention 1 av. for noisy nodes: ',
            torch.mean(torch.stack(att1[0::2])))
	print('attention 2 av. for noisy nodes: ',
            torch.mean(torch.stack(att2[0::2])))

	return torch.mean(alpha[1], axis=1)

############################################################################


def eval_shap(args_dataset,
			  args_model,
			  args_test_samples,
			  args_hops,
			  args_K,
			  args_num_samples,
			  node_indices=None):
	"""
	Compares SHAP and GraphSHAP on graph based datasets
	Check if they agree on features'contribution towards prediction for several test samples
	"""

	# Define dataset
	data = prepare_data(args_dataset, seed=10)

	# Select a random subset of nodes to eval the explainer on.
	if not node_indices:
		node_indices = extract_test_nodes(data, args_test_samples)

	# Define training parameters depending on (model-dataset) couple
	hyperparam = ''.join(['hparams_', args_dataset, '_', args_model])
	param = ''.join(['params_', args_dataset, '_', args_model])

	# Define the model
	if args_model == 'GCN':
		model = GCN(input_dim=data.x.size(
			1), output_dim=data.num_classes, **eval(hyperparam))
	else:
		model = GAT(input_dim=data.x.size(
			1), output_dim=data.num_classes, **eval(hyperparam))

	# Re-train the model on dataset with noisy features
	train_and_val(model, data, **eval(param))

	# Store metrics
	iou = []
	prop_contrib_diff = []

	# Iterate over test samples
	for node_idx in tqdm(node_indices, desc='explain node', leave=False):

		# Define explainers we would like to compare
		graphshap = GraphSHAP(data, model)
		shap = SHAP(data, model)

		# Explanations via GraphSHAP
		graphshap_coefs = graphshap.explain(node_index=node_idx,
											hops=args_hops,
											num_samples=args_num_samples,
											info=False)

		shap_coefs = shap.explain(node_index=node_idx,
								  hops=args_hops,
								  num_samples=args_num_samples,
								  info=False)

		# Consider node features only - for predicted class only
		true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[
			node_idx].max(dim=0)
		graphshap_coefs = graphshap_coefs[:graphshap.F, predicted_class]
		shap_coefs = shap_coefs[:, predicted_class]

		# Need to apply regularisation

		# Proportional contribution
		prop_contrib_diff.append(np.abs(graphshap_coefs.sum(
		) / np.abs(graphshap_coefs).sum() - shap_coefs.sum() / np.abs(shap_coefs).sum()))
		#print('GraphSHAP proportional contribution to pred: {:.2f}'.format(graphshap_coefs.sum() / np.abs(graphshap_coefs).sum() ))
		#print('SHAP proportional contribution to pred: {:.2f}'.format(shap_coefs.sum() / np.abs(shap_coefs).sum() ))

		# Important features
		graphshap_feat_indices = np.abs(
			graphshap_coefs).argsort()[-args_K:].tolist()
		shap_feat_indices = np.abs(shap_coefs).argsort()[-args_K:].tolist()
		iou.append(len(set(graphshap_feat_indices).intersection(set(shap_feat_indices))
					   ) / len(set(graphshap_feat_indices).union(set(shap_feat_indices))))
		#print('Iou important features: ', iou)

	print('iou av:', np.mean(iou))
	print('difference in contibutions towards pred: ', np.mean(prop_contrib_diff))


############################################################################

def eval_gnne(args_dataset, args_model, args_test_samples, node_indices):
	"""
	Evaluate GraphSHAP on GNNExplainer synthetic datasets 
	"""

	# Load / Create desired synthetic dataset
	data = prepare_data(args_dataset, 10)

	# Train model
	hyperparam = ''.join(['hparams_', args_dataset, '_', args_model])
	param = ''.join(['params_', args_dataset, '_', args_model])
	model = eval(args_model)(input_dim=data.x.size(1),
				output_dim=data.num_classes, **eval(hyperparam))
	train_and_val(model, data, **eval(param))

	# Select a random subset of nodes to eval the explainer on.
	if not node_indices:
		node_indices = extract_test_nodes(data, args_test_samples)

	for node_index in node_indices:
		_, predicted_class = model(data.x, data.edge_index).exp()[
			node_index].max(dim=0)

		# Explain graphshap and vizu important structure
		graphshap = GraphSHAP(data, model)
		graphshap_explanations = graphshap.explain(node_index,
												   hops=2,
												   num_samples=100,
												   info=True)
		
		
		print('-------------------------------------------------------')

		# Explain GNNE and vizu important structure
		gnne = GNNExplainer(data, model)
		gnne_explanations = gnne.explain(node_index,
										 hops=2,
										 num_samples=100,
										 info=True)

		# Our way to construct the graph - suboptimal but consistent with graphshap 
		ax, G = gnne.vizu(gnne.edge_mask, node_index, gnne_explanations, hops=2)

		# Original way to construct the graph for GNN Explainer
		explainer = GNNE(model, epochs=100)
		#node_feat_mask, edge_mask = explainer.explain_node(
		#	node_index, data.x, data.edge_index)
		explainer.visualize_subgraph(
			node_index, data.edge_index, gnne.edge_mask, y=data.y, threshold=None)
		
		print('-------------------------------------------------------')

		# Refer to paper to see metrics and how to properly assess the structure of interest and comparison with bench
		alpha = edge_attention(data, model, args_test_samples)
		explainer.visualize_subgraph(
			node_index, data.edge_index, alpha, y=data.y, threshold=None)
		
		# Vizu 
		explainer.visualize_subgraph(
                    node_index, data.edge_index, alpha, y=data.y, threshold=None)

		print('-------------------------------------------------------')
		
		# Metric to assess quality of predictions

def edge_attention(data, model, args_test_samples):
	"""
		Studies the attention weights of the GAT model
		"""
	_, alpha, alpha_bis = model(data.x, data.edge_index, att=True)

	# Remove self loops att
	alpha = alpha[1][:-(data.x.size(0)), :]
	alpha_bis = alpha_bis[1][:-(data.x.size(0))]

	return ( torch.mean(alpha, axis=1) + torch.mean(alpha_bis, axis=1) ) / torch.tensor(2)

