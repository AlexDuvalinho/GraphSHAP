from src.explainers import (LIME, SHAP, GNNExplainer, GraphLIME, GraphSVX,
                            Greedy)
from src.data import prepare_data
from src.train import accuracy
import torch
import argparse
import random
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def build_arguments():

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="Name of the GNN: GCN or GAT")
    parser.add_argument("--dataset", type=str,
                        help="Name of the dataset among Cora or PubMed")
    parser.add_argument("--explainer", type=str,
                        help="Name of the explainer among Greedy, GraphLIME, Random, SHAP, LIME")
    parser.add_argument("--seed", type=int)
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
        model='GAT',
        dataset='Cora',
        seed=10,
        explainer='GraphSVX',
        node_indexes=[90],
        hops=2,
        num_samples=300,
        fullempty=None, 
        S=3,
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

# args_hv: compute_pred', 'basic_default', 'neutral', 'graph_classification', 'compute_pred_subgraph'
# args_feat: 'All', 'Expectation', 'Null'
# args_coal: 'SmarterSeparate', 'Smarter', 'Smart', 'Random', 'All'
# args_g: WLS, 'WLR_sklearn', 'WLR_Lasso'

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

    # Load the model
    model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)
    model = torch.load(model_path)

    # Evaluate the model - test set
    model.eval()
    with torch.no_grad():
        log_logits = model(x=data.x, edge_index=data.edge_index)  # [2708, 7]
    test_acc = accuracy(log_logits[data.test_mask], data.y[data.test_mask])
    print('Test accuracy is {:.4f}'.format(test_acc))
    del log_logits, model_path, test_acc

    # Explain it with GraphSVX
    explainer = eval(args.explainer)(data, model, args.gpu)
    explanations = explainer.explain(args.node_indexes,
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
                                     True)

if __name__ == "__main__":
    main()
