# Use files in src folder
from src.explainers import (LIME, SHAP, GNNExplainer, GraphLIME, GraphSHAP,
							Greedy)
from src.data import prepare_data
from src.train import accuracy
import torch
import argparse
import random
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def build_arguments():

	# Argument parser
	# use if __name__=='main': and place the rest in a function that you call
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str,
						help="Name of the GNN: GCN or GAT")
	parser.add_argument("--dataset", type=str,
						help="Name of the dataset among Cora, PubMed, Amazon, PPI, Reddit")
	parser.add_argument("--explainer", type=str,
						help="Name of the explainer among Greedy, GraphLIME, Random, SHAP, LIME")
	parser.add_argument("--seed", type=int)
	parser.add_argument("--node_index", type=int, default=0,
						help="index of the node whose prediction is explained")
	parser.add_argument("--hops", type=int,
						help="number k for k-hops neighbours considered in an explanation")
	parser.add_argument("--num_samples", type=int,
						help="number of coalitions sampled and used to approx shapley values")

	parser.set_defaults(
		model='GCN',
		dataset='Cora',
		seed=10,
		explainer='GraphSHAP',
		node_index=0,
		hops=2,
		num_samples=500,
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

	# Evaluate the model - test set
	model.eval()
	with torch.no_grad():
		log_logits = model(x=data.x, edge_index=data.edge_index)  # [2708, 7]
	test_acc = accuracy(log_logits[data.test_mask], data.y[data.test_mask])
	print('Test accuracy is {:.4f}'.format(test_acc))
	del log_logits

	# Explain it with GraphSHAP
	explainer = eval(args.explainer)(data, model)
	explanations = explainer.explain(node_index=args.node_index,
									 hops=args.hops,
									 num_samples=args.num_samples,
									 info=True)

	print(explanations.shape)
	print('g(x_): ', sum(explanations[:,3]))


if __name__ == "__main__":
	main()
