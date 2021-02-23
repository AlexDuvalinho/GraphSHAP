from src.utils import *
from src.train import train_syn, test_gc, train_gc
from src.models import GcnEncoderGraph
from src.data import prepare_data, selected_data
from src.explainers import GraphSVX
import configs
import warnings
import argparse
import torch
import os
import numpy as np
import pickle as pkl

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
        dataset='Mutagenicity',
        seed=10,
        explainer='GraphSVX',
        node_indexes=[500],
        hops=2,
        num_samples=300,
        fullempty=None,
        S=3,
        hv='compute_pred_subgraph',
        feat='Expectation',
        coal='Smarter',
        g='WLR_sklearn',
        multiclass=False,
        regu=None,
        info=True,
        gpu=False
    )

    return parser.parse_args()

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


#ckpt = io_utils.load_ckpt(prog_args)
#model.load_state_dict(ckpt["model_state"])

# Train model 
model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)
if not os.path.exists(model_path):
    train_gc(data, model, prog_args)
    torch.save(model, model_path)
else: 
    model = torch.load(model_path)

# ckpt = io_utils.load_ckpt(prog_args)
# model.load_state_dict(ckpt["model_state"])

data = selected_data(data, args.dataset)

accuracy = []
for graph_idx in list(range(len(data.selected)))[100:120]:
    #graph_idx = 1
    graphshap = GraphSVX(data, model, args.gpu)
    graphshap_explanations = graphshap.explain_graphs([graph_idx],
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
                                                      regu=0,
                                                      vizu=False)[0]

    # Find gt
    idexs = np.nonzero(data.edge_label_lists[graph_idx])[
        0].tolist()  # indexes of edge gt
    inter = []  # retrieve edge gt from index
    for i in idexs:
        inter.append(data.edge_lists[graph_idx][i])
    ground_truth = [item for sublist in inter for item in sublist]
    ground_truth = list(set(ground_truth))  # node gt

    # Find ground truth (nodes) for each graph
    # Length gt
    k = len(ground_truth)

    # Keep only node explanations
    graphshap_node_explanations = graphshap_explanations  # [graphshap.F:]

    # Retrieve top k elements indices form graphshap_node_explanations
    if len(graphshap.neighbours) >= k:
        i = 0
        val, indices = torch.topk(torch.tensor(
            graphshap_node_explanations.T), k)
        # could weight importance based on val
        for node in torch.tensor(graphshap.neighbours)[indices]:
            if node.item() in ground_truth:
                i += 1
        # Sort of accruacy metric
        accuracy.append(i / k)
        print('acc:', i/k)
        #print('node', graphshap_node_explanations)
        print('indexes', indices)
        print('gt', ground_truth)

print('accuracy', accuracy)



# Explain and Eval 
def eval_syn6(data, model, args): 
    """Explain and evaluate syn6 dataset
    """
    allgraphs = [i for i in range(0, 100)]
    allgraphs.extend([i for i in range(500, 600)])
    accuracy = []
    for graph_idx in allgraphs[-20:]:
        graphshap = GraphSVX(data, model, args.gpu)
        graphshap_explanations = graphshap.explain_graphs([graph_idx],
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
                                                        regu=0,
                                                        vizu=False)[0]

        # Find gt
        preds = []
        reals = []
        ground_truth = list(range(20, 25))

        # Length gt
        k = len(ground_truth)

        # Keep only node explanations
        graphshap_node_explanations = graphshap_explanations  # [graphshap.F:]

        # Retrieve top k elements indices form graphshap_node_explanations
        if len(graphshap.neighbours) >= k:
            i = 0
            val, indices = torch.topk(torch.tensor(
                graphshap_node_explanations.T), k)
            # could weight importance based on val
            for node in torch.tensor(graphshap.neighbours)[indices]:
                if node.item() in ground_truth:
                    i += 1
            # Sort of accruacy metric
            accuracy.append(i / k)
            print('acc:', i/k)

    print('accuracy', accuracy)
    print('mean', np.mean(accuracy))





