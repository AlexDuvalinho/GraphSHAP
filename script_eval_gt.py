# The whole frame from GNN Explainer to get data and model
""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os
import warnings
import torch

# import configs
import utils.parser_utils as parser_utils
from src.data import prepare_data, selected_data
from src.eval import eval_Mutagenicity, eval_syn, eval_syn6

warnings.filterwarnings("ignore")


def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    #io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
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
        help="k-hop subgraph considered for GraphSVX",
    )
    parser.add_argument(
        "--num_samples",
        dest="num_samples",
        type=int,
        help="number of samples used to train GraphSVX",
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
    parser.add_argument("--model", type=str,
                        help="Name of the GNN: GCN or GAT")
    parser.add_argument("--dataset", type=str,
                        help="Name of the dataset among Cora, PubMed, Amazon, PPI, Reddit")
    parser.add_argument("--save", type=str,
                        help="True to save the trained model obtained")
    parser.add_argument("--node_indexes", type=list, default=[0],
                        help="indexes of the nodes whose prediction are explained")
    parser.add_argument("--gpu", type=bool, default=False, 
                        help='Whether we want to use GPU')


    # TODO: Check argument usage
    parser.set_defaults(
        logdir="log",
        ckptdir="models",
        dataset="syn6",
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
        hv='compute_pred',
        feat='Expectation',
        coal='SmarterSeparate',
        g='WLR_sklearn',
        regu=None,
        info=True,
        model='GCN',
        seed=10,
        explainer='GraphSVX',
        node_indexes=[500],
    )
    return parser.parse_args()


def main():

    # Load a configuration
    args = arg_parse() # configs.arg_parse()

    # GPU or CPU
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        print("CUDA", args.cuda)
    else:
        print("Using CPU")

    # Load dataset
    data = prepare_data(args.dataset, args.seed)

    # Load model
    model_path = 'models/GCN_model_{}.pth'.format(args.dataset)
    model = torch.load(model_path)

    # Evaluate GraphSVX 
    if args.dataset == 'Mutagenicity':
        data = selected_data(data, args.dataset)
        eval_Mutagenicity(data, model, args)
    elif args.dataset == 'syn6': 
        eval_syn6(data, model, args)
    else: 
        eval_syn(data, model, args)

if __name__ == "__main__":
    main()
