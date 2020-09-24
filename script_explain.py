# Use files in src folder
import torch
from src.data import prepare_data
from src.explainers import GraphSHAP, Greedy, GraphLIME, LIME, SHAP, GNNExplainer
import argparse


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
args = parser.parse_args()


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
									num_samples=args.num_samples)

print(explanations.shape)
print(explanations[1].max())



data = prepare_data('syn1', 10)

data = torch.load(data_path)