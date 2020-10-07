# Use files in src folder
import argparse
import warnings
import random
import numpy as np

warnings.filterwarnings("ignore")

import torch

from src.data import prepare_data
from src.explainers import (LIME, SHAP, GNNExplainer, GraphLIME, GraphSHAP,
                            Greedy)

def build_arguments():

	# Argument parser
	# use if __name__=='main': and place the rest in a function that you call 
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default= 'GCN', 
								help= "Name of the GNN: GCN or GAT")
	parser.add_argument("--dataset", type=str, default= 'Cora',
								help= "Name of the dataset among Cora, PubMed, Amazon, PPI, Reddit")
	parser.add_argument("--explainer", type=str, default= 'GraphSHAP',
								help= "Name of the explainer among Greedy, GraphLIME, Random, SHAP, LIME")
	parser.add_argument("--seed", type=int, default=10)
	parser.add_argument("--node_index", type=int, default=0,
								help="index of the node whose prediction is explained")
	parser.add_argument("--hops", type=int, default=2, 
								help="number k for k-hops neighbours considered in an explanation")
	parser.add_argument("--num_samples", type=int, default=100,
								help="number of coalitions sampled and used to approx shapley values")

	parser.set_defaults(
            	model='GCN',
          		dataset='Cora',
          		seed=10,
          		explainer='GraphSHAP',
				node_index=0,
          		hops=2,
          		num_samples=100,
	)

	return parser.parse_args()


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():

	args = build_arguments()
	fix_seed(args.seed)

	# Load the dataset
	data = prepare_data(args.dataset, args.seed)

	# Load the model 
	model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)
	model = torch.load(model_path)
	# model.eval()

	# Explain it with GraphSHAP
	explainer = eval(args.explainer)(data, model)
	explanations = explainer.explain(node_index=args.node_index, 
										hops=args.hops, 
										num_samples=args.num_samples,
										info=True)

	print(explanations.shape)
	print(explanations[1].max())

if __name__=="__main__":
	main()
