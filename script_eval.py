""" script_eval.py

    Evaluation and benchmark of GraphSHAP explainer
"""

import argparse
import warnings
import numpy as np
import random
import time

import torch

from src.eval_multiclass import filter_useless_features_multiclass, filter_useless_nodes_multiclass
from src.eval import filter_useless_features, filter_useless_nodes

warnings.filterwarnings("ignore")


def build_arguments():
    """ Build argument parser  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="Name of the GNN: GCN or GAT")
    parser.add_argument("--dataset", type=str,
                        help="Name of the dataset among Cora, PubMed, Amazon, PPI")
    parser.add_argument("--seed", type=int,
                        help="fix random state")
    parser.add_argument("--explainers", type=list, default=['GraphSHAP', 'GNNExplainer', 'GraphLIME',
                                                            'LIME', 'SHAP', 'Greedy'],
                        help="Name of the benchmarked explainers among GraphSHAP, SHAP, LIME, GraphLIME, Greedy and GNNExplainer")
    parser.add_argument("--node_explainers", type=list, default=['GraphSHAP', 'Greedy', 'GNNExplainer'],
                        help="Name of the benchmarked explainers among Greedy, GNNExplainer and GraphSHAP")
    parser.add_argument("--hops", type=int,
                        help='k of k-hops neighbours considered for the node of interest')
    parser.add_argument("--num_samples", type=int,
                        help='number of coalitions sampled')
    parser.add_argument("--test_samples", type=int,
                        help='number of test samples for evaluation')
    parser.add_argument("--K", type=float,
                        help='proportion of most important features considered, among non zero ones')
    parser.add_argument("--prop_noise_feat", type=float,
                        help='proportion of noisy features')
    parser.add_argument("--prop_noise_nodes", type=float,
                        help='proportion of noisy nodes')
    parser.add_argument("--connectedness", type=str,
                        help='how connected are the noisy nodes we define: low, high or medium')
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
    parser.add_argument("--gpu", type=bool,
                     help='True if want to use gpu')

    parser.set_defaults(
        model='GAT',
        dataset='Cora',
        seed=10,
        explainers=['GraphSHAP', 'GNNExplainer'],
        node_explainers=['GraphSHAP','GNNExplainer'],
        hops=2,
        num_samples=1000,
        test_samples=10,
        K=0.2,
        prop_noise_feat=0.10,
        prop_noise_nodes=0.10,
        connectedness='medium',
        multiclass=False,
        hv='compute_pred',
        feat='Expectation',
        coal='Smarter',
        g='WLR_sklearn',
        regu=None,
        info=False,
        gpu=True
    )
    # args_hv: 'compute_pred', 'node_specific', 'basic_default', 'basic_default_2hop', 'neutral', 'compute_pred_regu'
    # args_feat: 'All', 'Expectation', 'Null', 'Random'
    # args_coal: 'Smarter', 'Smart', 'Random', 'SmarterPlus', 'SmarterRegu'
    # args_g: 'WLR', WLS, 'WLR_sklearn'
    # args_regu: 'None', integer in [0,1] (1 for feat only)

    args = parser.parse_args()
    return args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():

    args = build_arguments()
    fix_seed(args.seed)
    node_indices = None
    
    start_time = time.time()

    if args.multiclass == False:

        # Only study neighbours 
        if args.coal == 'SmarterRegu':
            args.regu = 0

        # Neighbours
        filter_useless_nodes(args.model,
                             args.dataset,
                             args.node_explainers,
                             args.hops,
                             args.num_samples,
                             args.test_samples,
                             args.K,
                             args.prop_noise_nodes,
                             args.connectedness,
                             node_indices,
                             args.info,
                             args.hv, #node_specific
                             args.feat,
                             args.coal,
                             args.g,
                             args.multiclass,
                             args.regu,
                             args.gpu)
        
        # Only study features
        if args.coal == 'SmarterRegu':
            args.regu = 1

        # Features
        filter_useless_features(args.model,
                                args.dataset,
                                args.explainers,
                                args.hops,
                                args.num_samples,
                                args.test_samples,
                                args.K,
                                args.prop_noise_feat,
                                node_indices,
                                args.info,
                                args.hv,
                                args.feat,
                                args.coal,
                                args.g,
                                args.multiclass,
                                args.regu,
                                args.gpu)
    else:
        # Neighbours
        filter_useless_nodes_multiclass(args.model,
                                        args.dataset,
                                        args.node_explainers,
                                        args.hops,
                                        args.num_samples,
                                        args.test_samples,
                                        args.prop_noise_nodes,
                                        args.connectedness,
                                        node_indices,
                                        5,
                                        args.info,
                                        args.hv,
                                        args.feat,
                                        args.coal,
                                        args.g,
                                        args.multiclass,
                                        args.regu,
                                          args.gpu)

        # Node features
        filter_useless_features_multiclass(args.model,
                                           args.dataset,
                                           args.explainers,
                                           args.hops,
                                           args.num_samples,
                                           args.test_samples,
                                           args.prop_noise_feat,
                                           node_indices,
                                           5,
                                           args.info,
                                           args.hv,
                                           args.feat,
                                           args.coal,
                                           args.g,
                                           args.multiclass,
                                           args.regu,
                                           args.gpu)

    end_time = time.time()
    print('Time: ', end_time - start_time)

if __name__ == "__main__":
    main()
