""" script_eval.py

    Evaluation and benchmark of GraphSVX explainer
"""

import argparse
import warnings
import numpy as np
import random
import time
from itertools import product

import torch

from src.eval_multiclass import filter_useless_features_multiclass, filter_useless_nodes_multiclass
from src.eval import filter_useless_features, filter_useless_nodes, eval_shap

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
    parser.add_argument("--explainers", type=list, default=['GraphSVX', 'GNNExplainer', 'GraphLIME',
                                                            'LIME', 'SHAP', 'Greedy'],
                        help="Name of the benchmarked explainers among GraphSVX, SHAP, LIME, GraphLIME, Greedy and GNNExplainer")
    parser.add_argument("--node_explainers", type=list, default=['GraphSVX', 'Greedy', 'GNNExplainer'],
                        help="Name of the benchmarked explainers among Greedy, GNNExplainer and GraphSVX")
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
    parser.add_argument("--evalshap", type=bool,
                        help='True if want to compare GS with SHAP')
    parser.add_argument("--fullempty", type=str, default=None,
                        help='True if want to discard full and empty coalitions')
    parser.add_argument("--S", type=int, default=3,
                        help='Max size of coalitions sampled in priority and treated specifically')


    parser.set_defaults(
        model='GCN',
        dataset='Cora',
        seed=10,
        explainers=['GraphSVX','GNNExplainer', 'GraphLIME', 'LIME', 'SHAP'],
        node_explainers=['GraphSVX', 'GNNExplainer', 'Greedy'],
        hops=2,
        num_samples=3000,
        test_samples=50,
        K=0.10,
        prop_noise_feat=0.20,
        prop_noise_nodes=0.20,
        connectedness='medium',
        multiclass=False,
        fullempty=None,
        S=4,
        hv='compute_pred',
        feat='Expectation',
        coal='SmarterSeparate',
        g='WLR_sklearn',
        regu=None,
        info=False,
        gpu=False,
        evalshap=False
    )
    # args_hv: compute_pred', 'basic_default', 'neutral', 'compute_pred_subgraph', 'graph_classification'
    # args_feat: 'All', 'Expectation', 'Null'
    # args_coal: 'SmarterSeparate', 'Smarter', 'Smart', 'Random', 'All'
    # args_g: WLS, 'WLR_sklearn', 'WLR_Lasso'

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
        
        # Noisy nodes
        filter_useless_nodes(args.seed,
                            args.model,
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
                            args.hv,
                            args.feat,
                            args.coal,
                            args.g,
                            args.multiclass,
                            args.regu,
                            args.gpu,
                            args.fullempty,
                            args.S)
        
        # Noisy features
        filter_useless_features(args.seed,
                                args.model,
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
                                args.gpu,
                                args.fullempty,
                                args.S)
        
    else:
        # Neighbours
        filter_useless_nodes_multiclass(args.seed,
                                        args.model,
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
                                        args.gpu,
                                        args.fullempty,
                                        args.S)

        # Node features
        filter_useless_features_multiclass(args.seed,
                                        args.model,
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
                                        args.gpu,
                                        args.fullempty,
                                        args.S)

    
    if args.evalshap:
        eval_shap(args.seed,
                    args.dataset,
                    args.model,
                    args.test_samples,
                    args.hops,
                    args.K,
                    args.num_samples,
                    node_indices,
                    args.info,
                    args.hv,
                    args.feat,
                    args.coal,
                    args.g,
                    args.multiclass,
                    args.regu,
                    args.gpu,
                    args.fullempty,
                    args.S)

    end_time = time.time()
    print('Time: ', end_time - start_time)

if __name__ == "__main__":
    main()
