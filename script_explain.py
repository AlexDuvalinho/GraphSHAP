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
	parser.add_argument("--node_indexes", type=list, default=[0],
						help="indexes of the nodes whose prediction are explained")
	parser.add_argument("--hops", type=int,
						help="number k for k-hops neighbours considered in an explanation")
	parser.add_argument("--num_samples", type=int,
						help="number of coalitions sampled and used to approx shapley values")
	parser.add_argument("--hv", type=str,
                     help="way simplified input is translated to the original input space")
	parser.add_argument("--feat", type=str,
                     help="node features considered for hv above")
	parser.add_argument("--coal", type=str,
                     help="type of coalition sampler")
	parser.add_argument("--g", type=str,
                     help="method used to train g on derived dataset")
	parser.add_argument("--multiclass", type=bool,
                     help='False if we consider explanations for the predicted class only')

	parser.set_defaults(
		model='GCN',
		dataset='Cora',
		seed=10,
		explainer='GraphSHAP',
		node_indexes=[0,10],
		hops=2,
		num_samples=1000,
		hv='compute_pred',
		feat='Expectation',
		coal='SmarterPlus',
		g='WLS',
		multiclass=False,
	)

	return parser.parse_args()

# args_hv: compute_pred', 'node_specific', 'basic_default', 'basic_default_2hop', 'neutral'
# args_feat: 'All', 'Expectation', 'Null', 'Random'
# args_coal: 'Smarter', 'Smart', 'Random', 'SmarterPlus'
# args_g: 'WLR', WLS, 'WLR_sklearn'

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
	explanations = explainer.explain(node_indexes=args.node_indexes,
									 hops=args.hops,
									 num_samples=args.num_samples,
									 info=True,
									 args_hv=args.hv,
									 args_feat=args.feat,
									 args_coal=args.coal,
									 args_g=args.g,
									 multiclass=args.multiclass)

	print('number samples: ' ,len(explanations))
	print('dim expl:' , explanations[0].shape)

	for expl in explanations:
		if args.multiclass:
			print('g(x_): ', sum(expl[:,3]))
		else:
			print('g(x_): ', sum(expl))

if __name__ == "__main__":
	main()
