""" script_train.py

	Train, evaluate and save the desired GNN model 
	on this given dataset. 
"""

import argparse
import random
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from src.data import prepare_data
from src.models import GAT, GCN
from src.train import *
from src.train_ppi import main_ppi
from src.utils import *

	
def build_arguments():

	# Argument parser
	# use if __name__=='main': and place the rest in a function that you call 
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default= 'GAT', 
								help= "Name of the GNN: GCN or GAT")
	parser.add_argument("--dataset", type=str, default= 'Cora',
								help= "Name of the dataset among Cora, PubMed, Amazon, PPI, Reddit")
	parser.add_argument("--seed", type=int, default=10)
	parser.add_argument("--save", type=str, default=False,
								help= "True to save the trained model obtained")
	
	parser.set_defaults(
		model='GAT',
		dataset='Cora', 
		seed=10, 
		save=False
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

	# Train the model - specific case for PPI dataset
	if args.dataset == "PPI":
		model = main_ppi(type=args.model)
		# test_ppi shows how to compute predictions (model(), then positive values => predict this class)

	else: 
		# Retrieve the model and training hyperparameters depending the data/model given as input
		hyperparam = ''.join(['hparams_',args.dataset,'_', args.model])
		param = ''.join(['params_',args.dataset,'_', args.model])

		# Define the model
		if args.model == 'GCN':
			model = GCN(input_dim=data.x.size(1), output_dim= data.num_classes, **eval(hyperparam) )
		else:
			model = GAT(input_dim=data.x.size(1), output_dim= data.num_classes,  **eval(hyperparam) )

		# Train the model 
		train_and_val(model, data, **eval(param))

		# Compute predictions
		log_logits = model(x=data.x, edge_index=data.edge_index) # [2708, 7]
		probas = log_logits.exp()  # combine in 1 line + change accuracy

		# Evaluate the model - test set
		test_acc = accuracy(log_logits[data.test_mask], data.y[data.test_mask])
		print('Test accuracy is {:.4f}'.format(test_acc))

	# Save model
	if args.save:
		model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)	
		torch.save(model, model_path)


if __name__ == "__main__":
	main()
