from src.explainer import GraphSHAP
from src.data import add_noise_features, prepare_data, extract_test_nodes, add_noise_neighbors
from src.utils import *
from src.models import GCN, GAT
from src.train import *
from src.plots import plot_dist
import matplotlib.pyplot as plt

####### Input in script_eval file
args_dataset = 'PubMed'
args_model = 'GCN'
args_hops = 2
args_num_samples = 100 # size shap dataset
args_test_samples = 20 # number of test samples
args_num_noise_feat= 50 # number of noisy features
args_K= 5 # maybe def depending on M 
args_p = 0.1
args_binary = False
info=True

# Fix seed for add_noise_features
torch.manual_seed(10)

node_indices_list = [[2332,2101,1769,2546,2595,1913,1804,2419,2530,1872,2629,2272,1739,2394,1770,2030,2123,2176,1999,2608],
[10, 18, 89, 178, 333, 356, 378, 456, 500, 2222, 1220, 1900, 1328, 189, 1111, 124, 666, 684, 1556, 1881],
[1834,2512,2591,2101,1848,1853,2326,1987,2359,2453,2230,2267,2399, 2150,2400, 2546,1825,2529,2559,1883]]

node_indices = [2332,2101,1769,2546,2595,1913,1804,2419,2530,1872,2629,2272,1739,2394,1770,2030,2123,2176,1999,2608]
#node_indices= [2420,2455,1783,2165,2628,1822,2682,2261,1896,1880,2137,2237,2313,2218,1822,1719,1763,2263,2020,1988]
node_indices = [10, 18, 89, 178, 333, 356, 378, 456, 500, 2222, 1220, 1900, 1328, 189, 1111, 124, 666, 684, 1556, 1881]
node_indices = [1834,2512,2591,2101,1848,1853,2326,1987,2359,2453,2230,2267,2399, 2150,2400, 2546,1825,2529,2559,1883]



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



for node_indices in node_indices_list:

	# Select random subset of nodes to eval the explainer on. 
	# node_indices = extract_test_nodes(data, args_test_samples)

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
		print('Overall proportion of considered noisy features : {:.3f}%'.format(perc) )
		
		perc = 100 * sum(total_num_noise_feats) / ( args_K* args_test_samples* data.num_classes)
		print('Percentage of explanations showing noisy features: {:.3f}%'.format(perc) )

		perc = 100 * sum(total_num_noise_feats) / (sum(total_num_non_zero_noise_feat)*data.num_classes)
		perc2 = 100 * (args_K* args_test_samples* data.num_classes - sum(total_num_noise_feats) ) / (data.num_classes * (sum(M) - sum(total_num_non_zero_noise_feat)))
		print('Proportion of noisy features found in explanations vs normal features: {:.3f}% vs {:.3f}%, over considered feat only'.format(perc, perc2) )

		perc = 100 * sum(total_num_noise_feats) / (args_num_noise_feat*data.num_classes*args_test_samples)
		perc2 = 100 * (args_K* args_test_samples* data.num_classes - sum(total_num_noise_feats) ) / (data.num_classes * args_test_samples * (data.x.size(1) - args_num_noise_feat))
		print('Proportion of noisy features found in explanations vs normal features: {:.3f}% vs {:.3f}%'.format(perc, perc2) )

		# Plot of kernel density estimates of number of noisy features included in explanation
		# Do for all benchmarks (with diff colors) and plt.show() to get on the same graph 
	plot_dist(total_num_noise_feats, label='GraphSHAP', color='g')



"""
import torch
from src.data import prepare_data
from src.explainer import GraphSHAP
import argparse

data = prepare_data('PPI', 10)

model_path = 'models/{}_model_{}.pth'.format('GCN', 'PPI')
model = torch.load(model_path)
model.eval()

for df in data.graphs:
	graphshap = GraphSHAP(df, model)
	explanations = graphshap.explainer(10, 2, 10)
"""







class GAT(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, dropout, n_heads):
		super(GAT, self).__init__()
		self.dropout = dropout

		self.conv_in = GATConv(input_dim, hidden_dim[0], heads=n_heads[0], dropout=self.dropout)
		self.conv = [GATConv(hidden_dim[i-1] * n_heads[i-1], hidden_dim[i], heads=n_heads[i], dropout=self.dropout) for i in range(1,len(n_heads)-1)]
		self.conv_out = GATConv(hidden_dim[-1] * n_heads[-2], output_dim, heads=n_heads[-1], dropout=self.dropout, concat=False)

	def forward(self, x, edge_index, return_attention_weights=True):
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = F.elu(self.conv_in(x, edge_index))

		for attention in self.conv:
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = F.elu(attention(x, edge_index))
		
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.conv_out(x, edge_index)
	
		return F.log_softmax(x, dim=1)

model = GAT(input_dim=data.x.size(1), output_dim=data.num_classes, **eval(hyperparam) )

log_logits, x = model(x=data.x, edge_index=data.edge_index, return_attention_weights=True) # [2708, 7]


import torch
import torch_geometric as pyg

torch.manual_seed(0)
l= GATConv(data.x.size(1), data.num_classes)
x, alpha = l(data.x,data.edge_index,return_attention_weights=True)

print(alpha)








class GAT(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_heads):
		super(GAT, self).__init__()
		self.conv = GATConv(input_dim, hidden_dim, n_heads)
		self.conv1 = GATConv(hidden_dim, output_dim, n_heads, concat=False)
	
	def forward(self, x, edge_index, return_attention_weights=True):
		x, alpha = self.conv(x, edge_index, return_attention_weights=True)
		x  = self.conv1(x, edge_index)
		return F.log_softmax(x, dim=1), alpha


model = GAT(data.x.size(1), 16, data.num_classes, 1)

y, alpha = model(data.x, data.edge_index, return_attention_weights=True)
