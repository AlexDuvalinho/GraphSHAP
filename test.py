from src.train import test_gc, train_gc
from src.models import GcnEncoderGraph
from src.data import prepare_data, selected_data
from src.explainers import GraphSVX
from src.eval import eval_Mutagenicity, eval_syn6
import configs
import warnings
import argparse
import torch
import os
import numpy as np

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
    parser.add_argument("--regu", type=int,
                        help='None if we do not apply regularisation, 1 if only feat')
    parser.add_argument("--info", type=bool,
                        help='True if want to print info')
    parser.add_argument("--gpu", type=bool,
                        help='True if want to use gpu')
    parser.add_argument("--fullempty", type=str,
                        help='True if want to discard full and empty coalitions')
    parser.add_argument("--S", type=int,
                        help='Max size of coalitions sampled in priority and treated specifically')

    parser.set_defaults(
        model='GCN',
        dataset='syn6',
        seed=10,
        explainer='GraphSVX',
        node_indexes=[500],
        hops=3,
        num_samples=200,
        fullempty=None,
        S=4,
        hv='compute_pred',
        feat='Expectation',
        coal='Smarter',
        g='WLR_sklearn',
        multiclass=False,
        regu=None,
        info=True,
        gpu=False
    )

    return parser.parse_args()


def main():

    # Argument parser
    args = build_arguments()
    prog_args = configs.arg_parse()

    # Load the dataset
    data = prepare_data(args.dataset, args.seed)

    # Def model  
    input_dims = data.x.shape[-1]
    model = GcnEncoderGraph(input_dims,
                            prog_args.hidden_dim,
                            prog_args.output_dim,
                            prog_args.num_classes,
                            prog_args.num_gc_layers,
                            bn=prog_args.bn,
                            dropout=prog_args.dropout,
                            args=prog_args)

    # Train model 
    model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)
    if not os.path.exists(model_path):
        train_gc(data, model, prog_args)
        torch.save(model, model_path)
    else: 
        model = torch.load(model_path)
        res = test_gc(data, model, prog_args, data.test_mask)
        print('Test accuracy: ', res)


    # Eval on Mutagenicity or BA-2motifs
    if args.dataset == 'Mutagenicity':
        # Study only mutagen graphs with NO2 and NH2 components
        data = selected_data(data, args.dataset)
        eval_Mutagenicity(data, model, args)
    else: 
        eval_syn6(data, model, args)


if __name__ == '__main__':
    main()
