""" script_eval.py

    Evaluation and benchmark of GraphSHAP explainer
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
    parser.add_argument("--evalshap", type=bool,
                        help='True if want to compare GS with SHAP')
    parser.add_argument("--fullempty", type=str, default=None,
                        help='True if want to discard full and empty coalitions')
    parser.add_argument("--S", type=int, default=3,
                        help='Max size of coalitions sampled in priority and treated specifically')


    parser.set_defaults(
        model='GAT',
        dataset='Cora',
        seed=10,
        explainers=['GraphSHAP'],
        node_explainers=['GraphSHAP'],
        hops=2,
        num_samples=200,
        test_samples=10,
        K=0.10,
        prop_noise_feat=0.10,
        prop_noise_nodes=0.10,
        connectedness='medium',
        multiclass=False,
        fullempty=None,
        S=3,
        hv='compute_pred',
        feat='Null',
        coal='SmarterSeparate',
        g='WLS',
        regu=None,
        info=False,
        gpu=True,
        evalshap=False
    )
    # args_hv: 'compute_pred', 'compute_pred_subgraph', 'basic_default', 'neutral', 'compute_pred_regu'
    # args_feat: 'All', 'Expectation', 'Null', 'Random'
    # args_coal: 'Smarter', 'Smart', 'Random', 'All', 'SmarterSeparate', 'SmarterSoftRegu
    # args_g: 'WLR', WLS, 'WLR_sklearn', 'WLR_Lasso'
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

    ## List specific sets of hyperparameters values
    # node_indices = [2549,2664,2250,1881,2467,2663,1830,1938,1719,1828] #2367,2127,1899,2652,2100,2125
    # args_num_samples = [1000, 3000, 5000]
    # args_model = ['GCN', 'GAT']
    # args_seed = [0, 10, 100]
    # args_dataset = ['Cora', 'PubMed', 'Amazon']
    # args_hv = ['compute_pred'] #, 'basic_default', 'basic_default_2hop', 'neutral']
    # args_feat = ['All', 'Expectation', 'Null']
    # args_coal = ['Smarter', 'Smart', 'All', 'SmarterSeparate']
    # args_g = ['WLR', 'WLS', 'WLR_sklearn']
    # args_regu= ['None', 0, 1]
    # args_K = [0.1,0.2,0.05]
    # args_prop_noise_nodes = [0.1, 0.2, 0.05]
    # args_prop_noise_feat = [0.1, 0.2, 0.05]

    # Create combinations of above hyperparameters 
    # l = [args_K, args_prop_noise_nodes]
    # flat_list = list(product(*l))
    
    # for (args.K, args.prop_noise_nodes) in flat_list:
    #     args.prop_noise_feat = args_prop_noise_nodes
    
    for _ in [1]: 

        if args.multiclass == False:

            # Only study neighbours 
            #if args.coal == 'SmarterSoftRegu':
                #args.regu = 0

            # Neighbours
            
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
                                args.hv, #node_specific
                                args.feat,
                                args.coal,
                                args.g,
                                args.multiclass,
                                args.regu,
                                args.gpu,
                                args.fullempty,
                                args.S)
            
            # Only study features
            #if args.coal == 'SmarterRegu' or args.coal == 'SmarterSoftRegu':
                #args.regu = 1

            # Features
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
