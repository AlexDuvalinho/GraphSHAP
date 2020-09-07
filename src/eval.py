from src.explainer import GraphSHAP
from src.data import add_noise_features, prepare_data, extract_test_nodes
from src.utils import *
from src.models import GCN, GAT
from src.train import *


def filter_useless_feature():
	"""
	Add noisy features to dataset and check how many are included in explanations
	The fewest, the better the explainer.
	"""
	####### Input in script_eval file
	args_dataset = 'Cora'
	args_model = 'GCN'
	args_hops = 2
	args_num_samples = 200
	args_test_samples = 10
	args_num_noise_feat= 10
	args_K= 5 # maybe def depending on M 

	####### Create function filter_useless_feature() from here

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

	# Random subset of nodes selected to eval the explainer on. 
	# Ideally, use SP-LIME to select these nodes
	node_indices = extract_test_nodes(data, args_test_samples)

	# Loop on each test sample and store how many times do noise features appear among
	# K most influential features in our explanations 
	total_num_noise_feats = []
	for node_idx in tqdm(node_indices, desc='explain node', leave=False):
		
		# Explanations via GraphSHAP
		coefs = graphshap.explainer(node_index= node_idx, 
									hops=args_hops, 
									num_samples=args_num_samples,
									info=False)
		
		# Multilabel classification - consider all classes instead of focusing on 
		# class that is predicted by our model 
		num_noise_feats = []
		for i in range(data.num_classes):

			# Store indexes of K most important features, for each class
			feat_indices = np.abs(coefs[:,i]).argsort()[-args_K:] 
			feat_indices = [idx for idx in feat_indices] # look at most influent feat
			
			# Non zero noisy features
			non_zero_noise_feat = [val for val in noise_feat[node_idx] if val != 0]

			# Number of noisy features that appear in explanations
			num_noise_feat = sum(idx >= len(coefs[:,i]) - len(non_zero_noise_feat) for idx in feat_indices)
			num_noise_feats.append(num_noise_feat)
		
		# Return this number => number of times noisy features are provided as explanations
		total_num_noise_feats.append(sum(num_noise_feats))
	print('Noise features included in explanations: ', total_num_noise_feats)
	print('There are {} noise features found in the explanations of {} test samples'.format(sum(total_num_noise_feats),args_test_samples) )

	return sum(total_num_noise_feats)