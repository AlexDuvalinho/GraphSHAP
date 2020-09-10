from src.explainer import GraphSHAP
from src.data import add_noise_features, prepare_data, extract_test_nodes, add_noise_neighbors
from src.utils import *
from src.models import GCN, GAT
from src.train import *
from src.plots import plot_dist


def filter_useless_feature(info=True):
	"""
	Add noisy features to dataset and check how many are included in explanations
	The fewest, the better the explainer.
	"""
	####### Input in script_eval file
	args_dataset = 'Cora'
	args_model = 'GCN'
	args_hops = 2
	args_num_samples = 100
	args_test_samples = 100
	args_num_noise_feat= 10
	args_K= 5 # maybe def depending on M 

	#### Create function from here. Maybe create training fct first, to avoid retraining the model.

	# Define dataset - include noisy features 
	data = prepare_data(args_dataset, seed=10)
	data, noise_feat = add_noise_features(data, num_noise=args_num_noise_feat, binary=True)

	# Define training parameters depending on (model-dataset) couple
	hyperparam = ''.join(['hparams_',args_dataset,'_', args_model])
	param = ''.join(['params_',args_dataset,'_', args_model])

	# Define the model
	if args_model == 'GCN':
		model = GCN(input_dim=data.x.size(1), output_dim= data.num_classes, **eval(hyperparam) )
	else:
		model = GAT(input_dim=data.x.size(1), output_dim= data.num_classes,  **eval(hyperparam) )

	# Re-train the model on dataset with noisy features 
	train_and_val(model, data, **eval(param))

	# Define explainer
	graphshap = GraphSHAP(data, model)

	# Select random subset of nodes to eval the explainer on. 
	node_indices = extract_test_nodes(data, args_test_samples)

	# Loop on each test sample and store how many times do noise features appear among
	# K most influential features in our explanations 
	total_num_noise_feats = []
	M=[]
	pred_class_num_noise_feats=[]
	for node_idx in tqdm(node_indices, desc='explain node', leave=False):
		
		# Explanations via GraphSHAP
		coefs = graphshap.explainer(node_index= node_idx, 
									hops=args_hops, 
									num_samples=args_num_samples,
									info=False)
		
		# Check how many non zero features 
		M.append(graphshap.M)

		# Multilabel classification - consider all classes instead of focusing on the
		# class that is predicted by our model 
		num_noise_feats = []
		true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[node_idx].max(dim=0)
		for i in range(data.num_classes):

			# Store indexes of K most important features, for each class
			feat_indices = np.abs(coefs[:,i]).argsort()[-args_K:].tolist()  
			
			# Number of non zero noisy features
			num_non_zero_noise_feat = len([val for val in noise_feat[node_idx] if val != 0])

			# Number of noisy features that appear in explanations - use index to spot them
			num_noise_feat = sum(idx < num_non_zero_noise_feat for idx in feat_indices)
			num_noise_feats.append(num_noise_feat)

			if i==predicted_class:
				pred_class_num_noise_feats.append(num_noise_feat)
		
		# Return this number => number of times noisy features are provided as explanations
		total_num_noise_feats.append(sum(num_noise_feats))

	if info:		
		print('Noise features included in explanations: ', total_num_noise_feats)
		print('There are {} noise features found in the explanations of {} test samples, an average of {} per sample'\
			.format(sum(total_num_noise_feats),args_test_samples,sum(total_num_noise_feats)/args_test_samples) )

		# Number of noisy features found in explanation for the predicted class
		print(np.sum(pred_class_num_noise_feats), 'for the predicted class only' )

		prop = 100 * args_num_noise_feat* args_test_samples / np.sum(M)
		print('Overall proportion of noisy features: {:.3f}%'.format(prop) )
		
		perc = 100 * sum(total_num_noise_feats) / ( args_K* args_test_samples* data.num_classes)
		print('Percentage of explanations showing noisy features: {:.3f}%'.format(perc) )

		percen = 100 * sum(total_num_noise_feats) / ( args_num_noise_feat* args_test_samples* data.num_classes)
		print('Proportion of noisy features found in explanations: {:.3f}%'.format(percen) )

		# Plot of kernel density estimates of number of noisy features included in explanation
		# Do for all benchmarks (with diff colors) and plt.show() to get on the same graph 
	plot_dist(total_num_noise_feats, label='GraphSHAP', color='g')

	return sum(total_num_noise_feats)




def filter_useless_nodes(info=True):
	"""
	Add noisy features to dataset and check how many are included in explanations
	The fewest, the better the explainer.
	"""
	####### Input in script_eval file
	args_dataset = 'Cora'
	args_model = 'GAT'
	args_hops = 2
	args_num_samples = 100
	args_test_samples = 10
	args_num_noise_nei = 20
	args_K= 5 # maybe def depending on M 

	#### Create function from here. Maybe create training fct first, to avoid retraining the model.

	# Define dataset 
	data = prepare_data(args_dataset, seed=10)

	# Select random subset of nodes to eval the explainer on. 
	node_indices = extract_test_nodes(data, args_test_samples)

	# Include noisy neighbours
	data = add_noise_neighbors(data, args_num_noise_nei, node_indices, binary=True, connectedness='low')
	# data, noise_feat = add_noise_features(data, num_noise=args_num_noise_feat, binary=True)
	
	# Define training parameters depending on (model-dataset) couple
	hyperparam = ''.join(['hparams_',args_dataset,'_', args_model])
	param = ''.join(['params_',args_dataset,'_', args_model])

	# Define the model
	if args_model == 'GCN':
		model = GCN(input_dim=data.x.size(1), output_dim= data.num_classes, **eval(hyperparam) )
	else:
		model = GAT(input_dim=data.x.size(1), output_dim= data.num_classes,  **eval(hyperparam) )

	# Re-train the model on dataset with noisy features 
	train_and_val(model, data, **eval(param))

	# Define explainer
	graphshap = GraphSHAP(data, model)

	# Loop on each test sample and store how many times do noise features appear among
	# K most influential features in our explanations 
	total_num_noise_neis = [] # 1 el per test sample - count number of noisy nodes in explanations
	pred_class_num_noise_neis=[] # 1 el per test sample - count number of noisy nodes in explanations for 1 class
	total_num_noisy_nei = [] # 1 el per test sample - count number of noisy nodes in subgraph
	total_neigbours = [] # 1 el per test samples - number of neigbours of v in subgraph
	M=[] # 1 el per test sample - number of non zero features 
	for node_idx in tqdm(node_indices, desc='explain node', leave=False):
		
		# Explanations via GraphSHAP
		coefs = graphshap.explainer(node_index= node_idx, 
									hops=args_hops, 
									num_samples=args_num_samples,
									info=False)
		
		# Check how many non zero features 
		M.append(graphshap.M)

		# Number of noisy nodes in the subgraph of node_idx 
		num_noisy_nodes = len([n_idx for n_idx in graphshap.neighbors if n_idx >= data.x.size(0)-args_num_noise_nei])

		total_neigbours.append(len(graphshap.neighbors))

		# Multilabel classification - consider all classes instead of focusing on the
		# class that is predicted by our model 
		num_noise_neis = [] # one element for each class of a test sample
		true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[node_idx].max(dim=0)
		for i in range(data.num_classes):

			# Store indexes of K most important features, for each class
			nei_indices = np.abs(coefs[:,i]).argsort()[-args_K:].tolist()  
			
			# Number of noisy features that appear in explanations - use index to spot them
			num_noise_nei = sum(idx >= graphshap.M - num_noisy_nodes for idx in nei_indices)
			num_noise_neis.append(num_noise_nei)

			if i==predicted_class:
				pred_class_num_noise_neis.append(num_noise_nei)
		
		# Return this number => number of times noisy neighbours are provided as explanations
		total_num_noise_neis.append(sum(num_noise_neis))
		# Return number of noisy nodes adjacent to node of interest
		total_num_noisy_nei.append(num_noisy_nodes)

	if info:		
		print('Noisy neighbours included in explanations: ', total_num_noise_neis)

		print('There are {} noise neighbours found in the explanations of {} test samples, an average of {} per sample'\
			.format(sum(total_num_noise_neis),args_test_samples,sum(total_num_noise_neis)/args_test_samples) )

		print(np.sum(pred_class_num_noise_neis), 'for the predicted class only' )

		print('Proportion of explanations showing noisy neighbours: {:.3f}%'.format(100 * sum(total_num_noise_neis) / ( args_K* args_test_samples* data.num_classes)) )

		print('Proportion of noisy neighbours found in explanations : {:.3f}%'.format(100 * sum(total_num_noise_neis) / (args_test_samples * args_num_noise_nei * data.num_classes)))

		print('Proportion of neigbours that are noisy (in subgraphs): {:.3f}%'.format(100 * sum(total_num_noisy_nei) / sum(total_neigbours)))
		
		print('Proportion of noisy neighbours among features: {:.3f}%'.format(100 * args_num_noise_nei* args_test_samples / np.sum(M)) )
		
		

		# Plot of kernel density estimates of number of noisy features included in explanation
		# Do for all benchmarks (with diff colors) and plt.show() to get on the same graph 
	plot_dist(total_num_noise_neis, label='GraphSHAP', color='g')

	return sum(total_num_noise_neis)

