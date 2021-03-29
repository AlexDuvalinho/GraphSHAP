""" script_train.py

	Train, evaluate and save the desired GNN model 
	on the given dataset. 
"""

from src.utils import *
from src.train import train_and_val, evaluate, train_syn, train_gc
from src.models import GAT, GCN, GCNNet, GcnEncoderGraph, GcnEncoderNode
from src.data import prepare_data
from utils.io_utils import fix_seed
import argparse
import random
import torch
import configs

import numpy as np
import os 
import warnings

warnings.filterwarnings("ignore")


def main():

    args = configs.arg_parse()
    fix_seed(args.seed)

    # Load the dataset
    data = prepare_data(args.dataset, args.seed)

    # Define and train the model
    if args.dataset in ['Cora', 'PubMed']:
        # Retrieve the model and training hyperparameters depending the data/model given as input
        hyperparam = ''.join(['hparams_', args.dataset, '_', args.model])
        param = ''.join(['params_', args.dataset, '_', args.model])
        model = eval(args.model)(input_dim=data.num_features, output_dim=data.num_classes, **eval(hyperparam))
        train_and_val(model, data, **eval(param))
        _, test_acc = evaluate(data, model, data.test_mask)
        print('Test accuracy is {:.4f}'.format(test_acc))

    elif args.dataset in ['syn6', 'Mutagenicity']:
        input_dims = data.x.shape[-1]
        model = GcnEncoderGraph(input_dims,
                            args.hidden_dim,
                            args.output_dim,
                            data.num_classes,
                            args.num_gc_layers,
                            bn=args.bn,
                            dropout=args.dropout,
                            args=args)
        train_gc(data, model, args)
        _, test_acc = evaluate(data, model, data.test_mask)
        print('Test accuracy is {:.4f}'.format(test_acc))

    else: 
        # For pytorch geometric model 
        #model = GCNNet(args.input_dim, args.hidden_dim,
        #       data.num_classes, args.num_gc_layers, args=args)
        input_dims = data.x.shape[-1]
        model = GcnEncoderNode(data.num_features,
                                args.hidden_dim,
                                args.output_dim,
                                data.num_classes,
                                args.num_gc_layers,
                                bn=args.bn,
                                dropout=args.dropout,
                                args=args)
        train_syn(data, model, args)
        _, test_acc = evaluate(data, model, data.test_mask)
        print('Test accuracy is {:.4f}'.format(test_acc))
    
    # Save model 
    model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)
    if not os.path.exists(model_path) or args.save==True:
        torch.save(model, model_path)


if __name__ == "__main__":
    main()
