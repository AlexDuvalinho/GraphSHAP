""" script_eval.py

	Evaluation and benchmark of GraphSHAP explainer
"""

import argparse
import warnings
import numpy as np
import random
import time

import torch

from src.eval_multiclass import filter_useless_features_multiclass, filter_useless_nodes_multiclass
from src.eval import filter_useless_features, filter_useless_nodes

warnings.filterwarnings("ignore")


def build_arguments():
	""" Build argument parser  
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str,
						help="Name of the GNN: GCN or GAT")
	parser.add_argument("--dataset", type=str,
						help="Name of the dataset among Cora, PubMed, Amazon, PPI")
	parser.add_argument("--seed", type=int,
						help="fix random state")
	parser.add_argument("--explainers", type=list, default=['GraphSHAP', 'GNNExplainer', 'GraphLIME',
															'LIME', 'SHAP', 'Greedy'],
						help="Name of the benchmarked explainers among GraphSHAP, SHAP, LIME, GraphLIME, Greedy and GNNExplainer")
	parser.add_argument("--node_explainers", type=list, default=['GraphSHAP', 'Greedy', 'GNNExplainer'],
						help="Name of the benchmarked explainers among Greedy, GNNExplainer and GraphSHAP")
	parser.add_argument("--hops", type=int,
						help='k of k-hops neighbours considered for the node of interest')
	parser.add_argument("--num_samples", type=int,
						help='number of coalitions sampled')
	parser.add_argument("--test_samples", type=int,
						help='number of test samples for evaluation')
	parser.add_argument("--K", type=int,
						help='proportion of most important features considered, among non zero ones')
	parser.add_argument("--prop_noise_feat", type=int,
						help='proportion of noisy features')
	parser.add_argument("--prop_noise_nodes", type=int,
						help='proportion of noisy nodes')
	parser.add_argument("--connectedness", type=str,
						help='how connected are the noisy nodes we define: low, high or medium')
	parser.add_argument("--multiclass", type=bool,
						help='False if we consider explanations for the predicted class only')

	parser.set_defaults(
		model='GCN',
		dataset='Cora',
		seed=10,
		explainers=['GraphSHAP', 'GNNExplainer', 'GraphLIME',
					'SHAP', 'LIME'],
		node_explainers=['GraphSHAP', 'GNNExplainer', 'Greedy'],
		hops=2,
		num_samples=500,
		test_samples=5,
		K=0.25,
		prop_noise_feat=0.20,
		prop_noise_nodes=0.20,
		connectedness='medium',
		multiclass=False
	)

	args = parser.parse_args()
	return args


def fix_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True


def main():

	args = build_arguments()
	fix_seed(args.seed)
	node_indices = None
	
	start_time = time.time()

	if args.multiclass == False:

		# Neighbours
		filter_useless_nodes(args.model,
							 args.dataset,
							 args.node_explainers,
							 args.hops,
							 args.num_samples,
							 args.test_samples,
							 args.K,
							 args.prop_noise_nodes,
							 args.connectedness,
							 node_indices,
							 info=True)

		# Features
		filter_useless_features(args.model,
								args.dataset,
								args.explainers,
								args.hops,
								args.num_samples,
								args.test_samples,
								args.K,
								args.prop_noise_feat,
								node_indices,
								info=True)

	else:
		# Neighbours
		filter_useless_nodes_multiclass(args.model,
										args.dataset,
										args.node_explainers,
										args.hops,
										args.num_samples,
										args.test_samples,
										args.prop_noise_nodes,
										args.connectedness,
										node_indices,
										5,
										args.multiclass,
										info=True)

		# Node features
		filter_useless_features_multiclass(args.model,
										   args.dataset,
										   args.explainers,
										   args.hops,
										   args.num_samples,
										   args.test_samples,
										   args.prop_noise_feat,
										   node_indices,
										   5,
										   args.multiclass,
										   info=True)

	end_time = time.time()
	print('Time: ', end_time - start_time)

if __name__ == "__main__":
	main()
