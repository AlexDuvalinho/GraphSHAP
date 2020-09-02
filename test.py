# Use files in src folder
from src.data import prepare_data
from src.models import GCN, GAT, Net
from src.train import *
from src.explainer import GraphSHAP
from src.utils import *
import argparse


# Argument parser
# use if __name__=='main': and place the rest in a function that you call 
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default= 'GCN', 
								help= "Name of the GNN: GCN or GAT")
parser.add_argument("--dataset", type=str, default= 'Cora',
								help= "Name of the dataset among Cora, PubMed, Amazon, PPI, Reddit")
parser.add_argument("--seed", type=int, default='10')
args = parser.parse_args()


# Load the dataset
data = prepare_data(args.dataset, args.seed)

# Retrieve the model and training hyperparameters depending the data/model given as input
hyperparam = ''.join(['hparams_',args.dataset,'_', args.model])
param = ''.join(['params_',args.dataset,'_', args.model])

# Define the model
if args.model == 'GCN':
	model = GCN(input_dim=data.x.size(1), output_dim= max(data.y).item()+1, **eval(hyperparam) )
else:
	model = GAT(input_dim=data.x.size(1), output_dim= max(data.y).item()+1,  **eval(hyperparam) )

# Train the model
train_and_val(model, data, **eval(param))

# Compute predictions
log_logits = model(x=data.x, edge_index=data.edge_index) # [2708, 7]
probas = log_logits.exp()  # combine in 1 line + change accuracy

# Evaluate the model - test set
test_acc = accuracy(log_logits[data.test_mask], data.y[data.test_mask])
print('Test accuracy is {:.4f}'.format(test_acc))

# Explain it with GraphSHAP
#graphshap = GraphSHAP(data, model)
#explanations = graphshap.explainer(node_index=10, hops=2, num_samples=100)