import random
import torch_geometric
from utils import featgen
import numpy as np
import utils.io_utils as io_utils
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from torch_geometric.utils import from_networkx
from tensorboardX import SummaryWriter
import argparse
from types import SimpleNamespace
import torch.nn.functional as F
from torch.nn import init
import configs
import src.gengraph as gengraph
import utils.featgen as featgen
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from src.data import prepare_data, split_function
from src.explainers import GraphSVX
from src.models import GCNNet
from src.train import train_and_val


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


def test(loader, model, args, labels, test_mask):
    model.eval()

    train_ratio = args.train_ratio
    correct = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data)
            # print ('pred:', pred)
            pred = pred.argmax(dim=1)
            # print ('pred:', pred)

        # node classification: only evaluate on nodes in test set
        pred = pred[test_mask]
        # print ('pred:', pred)
        label = labels[test_mask]
        # print ('label:', label)

        correct += pred.eq(label).sum().item()

    total = len(test_mask)
    # print ('correct:', correct)
    return correct / total


def syn_task1(args, writer=None):
    # data
    print('Generating graph.')
    G, labels, name = gengraph.gen_syn1(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)))
    # print ('G.node[0]:', G.node[0]['feat'].dtype)
    # print ('Original labels:', labels)
    pyg_G = from_networkx(G)
    num_classes = max(labels)+1
    labels = torch.LongTensor(labels)
    print('Done generating graph.')

    model = GCNNet(args.input_dim, args.hidden_dim,
                   num_classes, args.num_gc_layers, args=args)

    if args.gpu:
        model = model.cuda()

    train_ratio = args.train_ratio
    num_train = int(train_ratio * G.number_of_nodes())
    num_test = G.number_of_nodes() - num_train
    shuffle_indices = list(range(G.number_of_nodes()))
    shuffle_indices = np.random.permutation(shuffle_indices)

    train_mask = num_train * [True] + num_test * [False]
    train_mask = torch.BoolTensor([train_mask[i] for i in shuffle_indices])
    test_mask = num_train * [False] + num_test * [True]
    test_mask = torch.BoolTensor([test_mask[i] for i in shuffle_indices])

    loader = torch_geometric.data.DataLoader([pyg_G], batch_size=1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    for epoch in range(args.num_epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            # print ('batch:', batch.feat)
            opt.zero_grad()
            pred = model(batch)

            pred = pred[train_mask]
            # print ('pred:', pred)
            label = labels[train_mask]
            # print ('label:', label)
            loss = model.loss(pred, label)
            print('loss:', loss)
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            loss.backward()
            opt.step()
            total_loss += loss.item() * 1
        total_loss /= num_train
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 10 == 0:
            test_acc = test(loader, model, args, labels, test_mask)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)



# Arguments
my_args = build_arguments()
prog_args = configs.arg_parse()
path = os.path.join(prog_args.logdir, io_utils.gen_prefix(prog_args))
syn_task1(prog_args, writer=SummaryWriter(path))


##############################################################################



# Construct graph
#G, labels, name = gengraph.gen_syn1( feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim)) )
#G, labels, name = gengraph.gen_syn4( feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)) )
#G, labels, name = gengraph.gen_syn5( feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)) )
#G, labels, name = gengraph.gen_syn2()
#args.input_dim = len(G.nodes[0]["feat"])

# Create
#data = prepare_data('syn1', 10)

#data = SimpleNamespace()
#data.x, data.edge_index, data.y = gengraph.preprocess_input_graph(G, labels)
#data.num_classes = max(labels) + 1
#data.num_features = data.x.shape[1]
#data.num_nodes = G.number_of_nodes()

# Train/test split only for nodes
# data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())

# Pickle adjacency and feature vector for each dataset

# Train model
# hyperparam = ''.join(['hparams_', 'syn1', '_', 'GAT'])
# param = ''.join(['params_', 'syn1', '_', 'GAT'])
# model = GAT(input_dim=data.x.size(
#     1), output_dim=data.num_classes, **eval(hyperparam))
# train_and_val(model, data, **eval(param))

# # Explain
# explainer = GraphSHAP(data, model)
# explanations = explainer.explain(node_index=0,
#                                  hops=2,
#                                  num_samples=100)


