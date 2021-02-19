# The whole frame from GNN Explainer to get data and model
""" explainer_main.py

     Main user interface for the explainer module.
"""
from src.explainers import GraphSVX
from src.data import prepare_data

import utils.train_utils as train_utils
import utils.parser_utils as parser_utils
import utils.math_utils as math_utils
import utils.io_utils as io_utils
import utils.graph_utils as graph_utils
import utils.featgen as featgen

import src.models
import src.gengraph
import configs
from tensorboardX import SummaryWriter
import torch
import sklearn.metrics as metrics
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import shutil
import warnings
from types import SimpleNamespace
from copy import deepcopy
import time

warnings.filterwarnings("ignore")


def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname",
                           help="Name of the pkl data file")

    parser_utils.parse_optimizer(parser)

    parser.add_argument("--clean-log", action="store_true",
                        help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir",
                        help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir",
                        help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout",
                        type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act",
                        type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--explain-node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=False,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )
    parser.add_argument(
        "--explainer-suffix",
        dest="explainer_suffix",
        help="suffix added to the explainer log",
    )
    parser.add_argument(
        "--hops",
        dest="hops",
        type=int,
        help="k-hop subgraph considered for GraphSHAP",
    )
    parser.add_argument(
        "--num_samples",
        dest="num_samples",
        type=int,
        help="number of samples used to train GraphSHAP",
    )
    parser.add_argument("--seed", type=int,
                        help='Seed number')
    parser.add_argument("--multiclass", type=bool,
                        help='False if we consider explanations for the predicted class only')
    parser.add_argument("--hv", type=str,
                        help="way simplified input is translated to the original input space")
    parser.add_argument("--feat", type=str,
                        help="node features considered for hv above")
    parser.add_argument("--coal", type=str,
                        help="type of coalition sampler")
    parser.add_argument("--g", type=str,
                        help="method used to train g on derived dataset")
    parser.add_argument("--regu", type=int,
                        help='None if we do not apply regularisation, 1 if only feat')
    parser.add_argument("--info", type=bool,
                        help='True if we want to see info about the explainer')
    parser.add_argument("--fullempty", type=str, default=None,
                        help='True if want to discard full and empty coalitions')
    parser.add_argument("--S", type=int, default=3,
                        help='Max size of coalitions sampled in priority and treated specifically')

    # TODO: Check argument usage
    parser.set_defaults(
        logdir="log",
        ckptdir="models",
        dataset="syn1",
        #bmname='Mutagenicity',
        graph_mode=False,
        opt="adam",
        opt_scheduler="none",
        gpu=False,
        cuda="0",
        lr=0.1,
        clip=2.0,
        batch_size=20,
        num_epochs=150,
        hidden_dim=20,
        output_dim=20,
        num_gc_layers=3,
        dropout=0.0,
        method="base",
        name_suffix="",
        explainer_suffix="",
        align_steps=100,
        explain_node=None,
        graph_idx=-1,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
        hops=2,
        num_samples=65,
        multiclass=False,
        fullempty=None,
        S=3,
        hv='compute_pred_subgraph',
        feat='Expectation',
        coal='SmarterSeparate',
        g='WLR_sklearn',
        regu=None,
        info=True,
    )
    return parser.parse_args()


def main():
    # Load a configuration
    args = arg_parse()
    seed = 10

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        print("CUDA", args.cuda)
    else:
        print("Using CPU")

    # Load the dataset
    data = prepare_data(args.dataset, seed)

    # Load the model
    model_path = 'models/GCN_model_{}.pth'.format(args.dataset)
    model = torch.load(model_path)

    if args.gpu:
        model = model.cuda()
    
    if args.gpu:
        y_pred = model(data.x.cuda(), data.edge_index.cuda())
    else:
        y_pred = model(data.x, data.edge_index)

    # Generate test nodes
    # Use only these specific nodes as they are the ones added manually, part of the defined shapes
    # node_indices = extract_test_nodes(data, num_samples=10, cg_dict['train_idx'])
    k = 4  # number of nodes for the shape introduced (house, cycle)
    K = 0
    if args.dataset == 'syn1':
        node_indices = list(range(400, 450, 5))
    elif args.dataset == 'syn2':
        node_indices = list(range(400, 425, 5)) + list(range(1100, 1125, 5))
    elif args.dataset == 'syn4':
        node_indices = list(range(511, 571, 6))
        if args.hops == 3:
            k = 5
        else:
            K = 5
    elif args.dataset == 'syn5':
        node_indices = list(range(511, 601, 9))
        if args.hops == 3:
            k = 8
        else:
            k = 5
            K = 8

    # GraphSHAP explainer
    graphshap = GraphSVX(data, model, args.gpu)
    
    # Condition for graph classification (Mutag) else explain node
    """
    graph_indices = [10,22]
    for graph_idx in graph_indices: 
        graphshap_explanations = graphshap.graph_classif([graph_idx],
                                                        args.hops,
                                                        args.num_samples,
                                                        args.info,
                                                        args.multiclass,
                                                        args.fullempty,
                                                        args.S,
                                                        'graph_classification',
                                                        args.feat,
                                                        args.coal,
                                                        args.g,
                                                        args.regu,
                                                        )[0]

        # Def k = len(ground_truth)
        # Derive ground truth from graph structure 
        # ground_truth 

        # Predicted class
        pred_val, predicted_class = y_pred[graph_idx, :].max(dim=0)

        # Keep only node explanations 
        graphshap_node_explanations = graphshap_explanations[graphshap.F:] #,predicted_class]
        
        # Retrieve top k elements indices form graphshap_node_explanations
        if graphshap.neighbours.shape[0] > k: 
            i = 0
            val, indices = torch.topk(torch.tensor(
                graphshap_node_explanations.T), k+1)
            # could weight importance based on val 
            for node in torch.tensor(graphshap.neighbours)[indices]: 
                if node.item() in ground_truth:
                    i += 1
            # Sort of accruacy metric
            accuracy.append(i / k) 
    """
                                                    
    # GraphSHAP - assess accuracy of explanations
    # Loop over test nodes
    accuracy = []
    diff_in_pred = []
    percentage_fidelity = []
    fct = torch.nn.Softmax(dim=1)
    feat_accuracy = []
    for node_idx in node_indices:
        graphshap_explanations = graphshap.explain([node_idx],
                                                   args.hops,
                                                   args.num_samples,
                                                   args.info,
                                                   args.multiclass,
                                                   args.fullempty,
                                                   args.S,
                                                   args.hv,
                                                   args.feat,
                                                   args.coal,
                                                   args.g,
                                                   args.regu,
                                                   )[0]

        # Predicted class
        pred_val, predicted_class = y_pred[node_idx, :].max(dim=0)

        # Keep only node explanations
        # ,predicted_class]
        graphshap_node_explanations = graphshap_explanations[graphshap.F:]

        # Derive ground truth from graph structure
        ground_truth = list(range(node_idx+1, node_idx+max(k, K)+1))

        # Retrieve top k elements indices form graphshap_node_explanations
        l = list(graphshap.neighbours).index(ground_truth[0])
        print('Importance:', np.sum(graphshap_explanations[l:l+5]))
        #print('Importance:', np.sum(
        #    graphshap_explanations[l:l+4]) / (np.sum(graphshap_explanations)-0.01652819)) # base value

        if graphshap.neighbours.shape[0] > k:
            i = 0
            val, indices = torch.topk(torch.tensor(
                graphshap_node_explanations.T), k+1)
            # could weight importance based on val
            for node in graphshap.neighbours[indices]:
                if node.item() in ground_truth:
                    i += 1
            # Sort of accruacy metric
            accuracy.append(i / k)

            print('There are {} from targeted shape among most imp. nodes'.format(i))

        # Look at importance distribution among features
        # Identify most important features and check if it corresponds to truly imp ones
        if args.dataset == 'syn2':
            # ,predicted_class]
            graphshap_feat_explanations = graphshap_explanations[:graphshap.F]
            print('Feature importance graphshap',
                  graphshap_feat_explanations.T)
            feat_accuracy.append(len(set(np.argsort(
                graphshap_feat_explanations)[-2:]).intersection([0, 1])) / 2)

        # Fidelity
        """
        # Compute original prediction
        original_prediction = fct(y_pred[:, node_idx])
        original_prediction, pred_idx = torch.topk(original_prediction, 1)
        # Isolate nodes in graphshap neighbours, in adj
        adj_bis = deepcopy(adj)
        for item in graphshap.neighbours[indices]:
            adj_bis[0, item, :] = torch.zeros(adj.shape[0])
        # Compute new prediction
        new_pred, _ = model(data.x, adj_bis)
        new_prediction = fct(new_pred[:, node_idx])[0][pred_idx]
        # Difference
        diff_in_pred.append((original_prediction - new_prediction).item())
        percentage_fidelity.append(
            (original_prediction - new_prediction).item() / original_prediction.item())
        """
    # Metric for graphshap
    final_accuracy = sum(accuracy)/len(accuracy)

    
    print(np.mean(final_accuracy), np.mean(
        percentage_fidelity), np.mean(feat_accuracy))


if __name__ == "__main__":
    main()
