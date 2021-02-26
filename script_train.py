""" script_train.py

	Train, evaluate and save the desired GNN model 
	on the given dataset. 
"""

from src.utils import *
from src.train import train_and_val, evaluate, train_syn, train_gc
from src.models import GAT, GCN, GCNNet, GcnEncoderGraph, GcnEncoderNode
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
        dataset='Cora',
        seed=10,
        save=True,
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

    # Define and train the model
    if args.dataset in ['Cora', 'PubMed']:
        # Retrieve the model and training hyperparameters depending the data/model given as input
        hyperparam = ''.join(['hparams_', args.dataset, '_', args.model])
        param = ''.join(['params_', args.dataset, '_', args.model])
        model = eval(args.model)(input_dim=data.x.size(
            1), output_dim=data.num_classes, **eval(hyperparam))
        train_and_val(model, data, **eval(param))
        _, test_acc = evaluate(data, model, data.test_mask)
        print('Test accuracy is {:.4f}'.format(test_acc))

    elif args.dataset in ['syn6', 'Mutagenicity']:
        input_dims = data.x.shape[-1]
        model = GcnEncoderGraph(input_dims,
                            prog_args.hidden_dim,
                            prog_args.output_dim,
                            prog_args.num_classes,
                            prog_args.num_gc_layers,
                            bn=prog_args.bn,
                            dropout=prog_args.dropout,
                            args=prog_args)
        train_gc(data, model, prog_args)

    else: 
        # For pytorch geometric model 
        #model = GCNNet(prog_args.input_dim, prog_args.hidden_dim,
        #       data.num_classes, prog_args.num_gc_layers, args=prog_args)
        input_dims = data.x.shape[-1]
        model = GcnEncoderNode(input_dims,
                                prog_args.hidden_dim,
                                prog_args.output_dim,
                                data.num_classes,
                                prog_args.num_gc_layers,
                                bn=prog_args.bn,
                                dropout=prog_args.dropout,
                                args=prog_args)
        train_syn(data, model, prog_args)
    
    # Save model 
    model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)
    if not os.path.exists(model_path) or args.save==True:
        torch.save(model, model_path)

if __name__ == "__main__":
    main()
