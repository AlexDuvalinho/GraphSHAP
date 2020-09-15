from src.explainer import GraphSHAP
from src.data import add_noise_features, prepare_data, extract_test_nodes, add_noise_neighbors
from src.utils import *
from src.models import GCN, GAT
from src.train import *
from src.plots import plot_dist
import matplotlib.pyplot as plt


def filter_useless_features(args_model,
							args_dataset,
							args_hops,
							args_num_samples,
							args_test_samples,
							args_K,
							args_num_noise_feat,
							args_p,
							args_binary,
							node_indices,
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
	args_hops = 2
	args_num_samples = 100 # size shap dataset
	args_test_samples = 20 # number of test samples
	args_num_noise_feat= 25 # number of noisy features
	args_K= 5 # maybe def depending on M 
	info=True

	node_indices= [2420,2455,1783,2165,2628,1822,2682,2261,1896,1880,2137,2237,2313,2218,1822,1719,1763,2263,2020,1988]
	node_indices = [10, 18, 89, 178, 333, 356, 378, 456, 500, 2222, 1220, 1900, 1328, 189, 1111]
	node_indices = [1834,2512,2591,2101,1848,1853,2326,1987,2359,2453,2230,2267,2399, 2150,2400]
	'''
	
	#### Create function from here. Maybe create training fct first, to avoid retraining the model.

	# Define dataset - include noisy features
	data = prepare_data(args_dataset, seed=10)
	data, noise_feat = add_noise_features(data, num_noise=args_num_noise_feat, binary=args_binary, p=args_p)

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
	if not node_indices:
		node_indices = extract_test_nodes(data, args_test_samples)

	total_num_noise_feats = [] # count noisy features found in explanations for each test sample (for each class)
	pred_class_num_noise_feats=[] # count noisy features found in explanations for each test sample for class of interest
	total_num_non_zero_noise_feat =[] # count number of noisy features considered for each test sample
	M=[] # count number of non zero features for each test sample
	
	# Loop on each test sample and store how many times do noise features appear among
	# K most influential features in our explanations 
	for node_idx in tqdm(node_indices, desc='explain node', leave=False):
		
		# Explanations via GraphSHAP
		coefs = graphshap.explainer(node_index= node_idx, 
									hops=args_hops, 
									num_samples=args_num_samples,
									info=False)
		
		# Check how many non zero features 
		M.append(graphshap.M)

		# Number of non zero noisy features
		num_non_zero_noise_feat = len([val for val in noise_feat[node_idx] if val != 0])

		# Multilabel classification - consider all classes instead of focusing on the
		# class that is predicted by our model 
		num_noise_feats = []
		true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[node_idx].max(dim=0)

		for i in range(data.num_classes):

			# Store indexes of K most important features, for each class
			feat_indices = np.abs(coefs[:,i]).argsort()[-args_K:].tolist()  

			# Number of noisy features that appear in explanations - use index to spot them
			num_noise_feat = sum(idx < num_non_zero_noise_feat for idx in feat_indices)
			num_noise_feats.append(num_noise_feat)

			if i==predicted_class:
				pred_class_num_noise_feats.append(num_noise_feat)
		
		# Return this number => number of times noisy features are provided as explanations
		total_num_noise_feats.append(sum(num_noise_feats))
		# Return number of noisy features considered in this test sample
		total_num_non_zero_noise_feat.append(num_non_zero_noise_feat)

	if info:		
		print('Noise features included in explanations: ', total_num_noise_feats)
		print('There are {} noise features found in the explanations of {} test samples, an average of {} per sample'\
			.format(sum(total_num_noise_feats),args_test_samples,sum(total_num_noise_feats)/args_test_samples) )

		# Number of noisy features found in explanation for the predicted class
		print(np.sum(pred_class_num_noise_feats)/args_test_samples, 'for the predicted class only' )

		perc = 100 * sum(total_num_non_zero_noise_feat) / np.sum(M)
		print('Overall proportion of considered noisy features : {:.2f}%'.format(perc) )
		
		perc = 100 * sum(total_num_noise_feats) / ( args_K* args_test_samples* data.num_classes)
		print('Percentage of explanations showing noisy features: {:.2f}%'.format(perc) )

		if sum(total_num_non_zero_noise_feat) != 0: 
			perc = 100 * sum(total_num_noise_feats) / (sum(total_num_non_zero_noise_feat)*data.num_classes)
			perc2 = 100 * (args_K* args_test_samples* data.num_classes - sum(total_num_noise_feats) ) / (data.num_classes * (sum(M) - sum(total_num_non_zero_noise_feat)))
			print('Proportion of noisy features found in explanations vs normal features: {:.2f}% vs {:.2f}%, over considered features only'.format(perc, perc2) )

		# Plot of kernel density estimates of number of noisy features included in explanation
		# Do for all benchmarks (with diff colors) and plt.show() to get on the same graph 
	plot_dist(total_num_noise_feats, label='GraphSHAP', color='g')

	return sum(total_num_noise_feats)




def filter_useless_nodes(args_model,
							args_dataset,
							args_hops,
							args_num_samples,
							args_test_samples,
							args_K,
							args_num_noise_nodes,
							args_p,
							args_binary,
							args_connectedness,
							node_indices=None,
							info=True):
	"""
	Arguments defined in argument parser in script_eval.py
	Add noisy features to dataset and check how many are included in explanations
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
	#### Create function from here. Maybe create training fct first, to avoid retraining the model.

	# Define dataset 
	data = prepare_data(args_dataset, seed=10)

	# Select random subset of nodes to eval the explainer on. 
	if not node_indices:
		node_indices = extract_test_nodes(data, args_test_samples)

	# Include noisy neighbours
	data = add_noise_neighbors(data, args_num_noise_nodes, node_indices, binary=args_binary, p=args_p, connectedness=args_connectedness)
	# data, noise_feat = add_noise_features(data, num_noise=args_num_noise_feat, binary=True)
	
	# Define training parameters depending on (model-dataset) couple
	hyperparam = ''.join(['hparams_',args_dataset,'_', args_model])
	param = ''.join(['params_',args_dataset,'_', args_model])

	# Define the model
	if args_model == 'GCN':
		model = GCN(input_dim=data.x.size(1), output_dim=data.num_classes, **eval(hyperparam) )
	else:
		model = GAT(input_dim=data.x.size(1), output_dim=data.num_classes, **eval(hyperparam) )

	# Re-train the model on dataset with noisy features 
	train_and_val(model, data, **eval(param))

	# Study attention weights of noisy nodes - for 20 new nodes
	def study_attention_weights(data, model):
		"""
		Studies the attention weights of the GAT model 
		"""
		_, alpha, alpha_bis = model(data.x, data.edge_index, att=True)

		edges, alpha1 = alpha[0][:, :-(data.x.size(0)-1)], alpha[1][:-(data.x.size(0)-1), :] # remove self loops att
		alpha2 = alpha_bis[1][:-(data.x.size(0)-1)]
		
		att1 = []
		att2 = []
		for i in range( data.x.size(0) - args_test_samples, (data.x.size(0)-1)):
			ind = (edges==i).nonzero()
			for j in ind[:,1]:
				att1.append(torch.mean(alpha1[j]))
				att2.append(alpha2[j][0])
		print('shape attention noisy', len(att2))

		# It looks like these noisy nodes are very important 
		print('av attention',  (torch.mean(alpha1) + torch.mean(alpha2))/2 )  # 0.18
		(torch.mean(torch.stack(att1)) + torch.mean(torch.stack(att2)))/2 # 0.32

		# In fact, noisy nodes are slightly below average in terms of attention received
		# Importance of interest: look only at imp. of noisy nei for test nodes
		print('attention 1 av. for noisy nodes: ', torch.mean(torch.stack(att1[0::2])))
		print('attention 2 av. for noisy nodes: ', torch.mean(torch.stack(att2[0::2])))

	# Study attention weights
	if str(type(model)) == "<class 'src.models.GAT'>":
		study_attention_weights(data, model)

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
		num_noisy_nodes = len([n_idx for n_idx in graphshap.neighbors if n_idx >= data.x.size(0)-args_num_noise_nodes])

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

		print(np.sum(pred_class_num_noise_neis)/args_test_samples, 'for the predicted class only' )

		print('Proportion of explanations showing noisy neighbours: {:.2f}%'.format(100 * sum(total_num_noise_neis) / ( args_K* args_test_samples* data.num_classes)) )

		perc = 100 * sum(total_num_noise_neis) / (args_test_samples * args_num_noise_nodes * data.num_classes)
		perc2 = 100 * (( args_K* args_test_samples* data.num_classes)  - sum(total_num_noise_neis)) / (np.sum(M) - sum(total_num_noisy_nei))
		print('Proportion of noisy neighbours found in explanations vs normal features: {:.2f}% vs {:.2f}'.format(perc, perc2))

		print('Proportion of nodes in subgraph that are noisy: {:.2f}%'.format(100 * sum(total_num_noisy_nei) / sum(total_neigbours)))
		
		print('Proportion of noisy neighbours among features: {:.2f}%'.format(100 * sum(total_num_noisy_nei) / np.sum(M)) )
		
	# Plot of kernel density estimates of number of noisy features included in explanation
	# Do for all benchmarks (with diff colors) and plt.show() to get on the same graph 
	plot_dist(total_num_noise_neis, label='GraphSHAP', color='g')
	#plt.show()

	return total_num_noise_neis

