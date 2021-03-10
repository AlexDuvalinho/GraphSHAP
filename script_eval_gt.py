# The whole frame from GNN Explainer to get data and model
""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os
import warnings
import torch

import configs 
from src.data import prepare_data, selected_data
from src.eval import eval_Mutagenicity, eval_syn, eval_syn6

import utils.io_utils as io_utils
import src.models as models
from types import SimpleNamespace
import torch_geometric

warnings.filterwarnings("ignore")

def main():

    # Load a configuration
    args = configs.arg_parse()

    # GPU or CPU
    if args.gpu:
        print("CUDA")
    else:
        print("Using CPU")


    # Load dataset
    data = prepare_data(args.dataset, args.seed)
    
    args.ckptdir = 'ckpt'
    args.bmname = None
    args.name_suffix = ''
    ckpt = io_utils.load_ckpt(args)

    cg_dict = ckpt["cg"]
    adj = torch.tensor(cg_dict["adj"], dtype=torch.float)
    x = torch.tensor(cg_dict["feat"], requires_grad=True, dtype=torch.float)
    data = SimpleNamespace()
    data.x = torch.tensor(x.squeeze())
    data.edge_index = torch_geometric.utils.dense_to_sparse(adj.squeeze(0))[0]
    data.y = cg_dict["label"][0].tolist()
    data.num_classes = max(data.y) + 1
    data.num_features = data.x.size(1)
    data.num_nodes = data.x.size(0)

    model = models.GcnEncoderNode(
       input_dim=data.num_features,
       hidden_dim=args.hidden_dim,
       embedding_dim=args.output_dim,
       label_dim=data.num_classes,
       num_layers=args.num_gc_layers,
       bn=args.bn,
       args=args,
    )
    model.load_state_dict(ckpt["model_state"])
    del ckpt
    
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

"""
    args.ckptdir = 'ckpt'
    args.bmname = None
    args.name_suffix = ''
    ckpt = io_utils.load_ckpt(args)

    cg_dict = ckpt["cg"]
    adj = torch.tensor(cg_dict["adj"], dtype=torch.float)
    x = torch.tensor(cg_dict["feat"], requires_grad=True, dtype=torch.float)
    data = SimpleNamespace()
    data.x = torch.tensor(x.squeeze())
    data.edge_index = torch_geometric.utils.dense_to_sparse(adj.squeeze(0))[0]
    data.y = cg_dict["label"][0].tolist()
    data.num_classes = max(data.y) + 1
    data.num_features = data.x.size(1)
    data.num_nodes = data.x.size(0)
"""
