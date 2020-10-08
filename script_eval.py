""" script_eval.py

	Evaluation and benchmark of GraphSHAP explainer
"""

import argparse
import warnings
import numpy as np
import random

import torch

from src.eval_multiclass import filter_useless_features, filter_useless_nodes
from src.eval import filter_useless_features1, filter_useless_nodes1
from src.utils import DIM_FEAT_P

warnings.filterwarnings("ignore")


def build_arguments():
	""" Build argument parser  
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, 
								help= "Name of the GNN: GCN or GAT")
	parser.add_argument("--dataset", type=str, 
								help= "Name of the dataset among Cora, PubMed, Amazon, PPI")
	parser.add_argument("--seed", type=int, 
                     help="fix random state")
	parser.add_argument("--explainers", type=list, default=['GraphSHAP', 'SHAP', 'LIME', 'GraphLIME', 'Greedy', 'GNNExplainer'],
								help= "Name of the benchmarked explainers among GraphSHAP, SHAP, LIME, GraphLIME, Greedy and GNNExplainer")
	parser.add_argument("--node_explainers", type=list, default= ['GraphSHAP', 'Greedy', 'GNNExplainer'],
								help= "Name of the benchmarked explainers among Greedy, GNNExplainer and GraphSHAP")
	parser.add_argument("--hops", type=int, 
								help= 'k of k-hops neighbours considered for the node of interest')
	parser.add_argument("--num_samples", type=int, 
								help= 'number of coalitions sampled')
	parser.add_argument("--test_samples", type=int, 
								help='number of test samples for evaluation')
	parser.add_argument("--K", type=int, 
								help= 'number of most important features considered')
	parser.add_argument("--num_noise_feat", type=int, 
								help='number of noisy features')
	parser.add_argument("--num_noise_nodes", type=int,
								help= 'number of noisy nodes')
	parser.add_argument("--p", type=float, 
								help= 'proba of existance for each feature, if binary')
	parser.add_argument("--binary", type=bool, 
								help= 'if noisy features are binary or not')
	parser.add_argument("--connectedness", type=str, 
								help= 'how connected are the noisy nodes we define: low, high or medium')
	parser.add_argument("--multiclass", type=bool, 
								help= 'False if we consider explanations for the predicted class only')		
	
	parser.set_defaults(
            model='GCN',
            dataset='Cora',
            seed=10,
            explainers=['GraphSHAP', 'SHAP', 'LIME', 
						'GraphLIME', 'Greedy', 'GNNExplainer'],
            node_explainers=['GraphSHAP', 'GNNExplainer'],
            hops=2,
            num_samples=200,
            test_samples=20,
            K=0.25,
            num_noise_feat=0.2,
            num_noise_nodes=20,
            p=0.013,
            binary=True,
            connectedness='low',
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

	if args.multiclass == True: 

		# Node features
		filter_useless_features(args.model,
								args.dataset,
								args.explainers,
								args.hops,
								args.num_samples,
								args.test_samples, 
								args.K,
								args.num_noise_feat,
								args.p,
								args.binary,
								node_indices,
								args.multiclass,
								info=True)

		
		# Neighbours
		filter_useless_nodes(args.model,
							args.dataset,
							args.node_explainers,
							args.hops,
							args.num_samples,
							args.test_samples, 
							args.K,
							args.num_noise_nodes,
							args.p,
							args.binary,
							args.connectedness,
							node_indices,
							args.multiclass,
							info=True)
	else: 
		filter_useless_features1(args.model,
						args.dataset,
						args.explainers,
						args.hops,
						args.num_samples,
						args.test_samples,
						args.K,
						args.num_noise_feat,
						args.p,
						args.binary,
						node_indices,
						info=True)

		# Neighbours
		filter_useless_nodes1(args.model,
							args.dataset,
							args.node_explainers,
							args.hops,
							args.num_samples,
							args.test_samples, 
							args.K,
							args.num_noise_nodes,
							args.p,
							args.binary,
							args.connectedness,
							node_indices,
							info=True)
if __name__=="__main__":
	main()
