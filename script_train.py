""" script_train.py

	Train, evaluate and save the desired GNN model 
	on the given dataset. 
"""

from src.utils import *
from src.train import train_and_val, accuracy, train_syn
from src.models import GAT, GCN, GCNNet
from src.data import prepare_data
import argparse
import random
import torch
import configs

import numpy as np
import os 
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
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save", type=str,
                        help="True to save the trained model obtained")

    parser.set_defaults(
        model='GCN',
        dataset='syn1',
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
    prog_args = configs.arg_parse()
    fix_seed(args.seed)

    # Load the dataset
    data = prepare_data(args.dataset, args.seed)

    # Retrieve the model and training hyperparameters depending the data/model given as input
    hyperparam = ''.join(['hparams_', args.dataset, '_', args.model])
    param = ''.join(['params_', args.dataset, '_', args.model])

    # Define and train the model
    if args.dataset in ['Cora', 'PubMed']:
        model = eval(args.model)(input_dim=data.x.size(
            1), output_dim=data.num_classes, **eval(hyperparam))
        train_and_val(model, data, **eval(param))
    else: 
        model = GCNNet(prog_args.input_dim, prog_args.hidden_dim,
               data.num_classes, prog_args.num_gc_layers, args=prog_args)
        train_syn(data, model, prog_args)
    
    # Compute predictions
    model.eval()
    with torch.no_grad():
        log_logits = model(x=data.x, edge_index=data.edge_index)  # [2708, 7]
    probas = log_logits.exp()  # combine in 1 line + change accuracy

    # Evaluate the model - test set
    test_acc = accuracy(log_logits[data.test_mask], data.y[data.test_mask])
    print('Test accuracy is {:.4f}'.format(test_acc))

    # Save model
    model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)
    if not os.path.exists(model_path):
        torch.save(model, model_path)

if __name__ == "__main__":
    main()
