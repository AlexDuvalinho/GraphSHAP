from src.explainers import GraphLIME, SHAP, GraphSHAP
from src.models import GCN, GAT
import torch
from src.utils import *
from src.data import prepare_data


#dataset = prepare_data('Cora', 10)
#hyperparam = ''.join(['hparams_','Cora','_', 'GCN'])
#param = ''.join(['params_','Cora','_', 'GCN'])
#model = GCN(input_dim=data.x.size(1), output_dim= data.num_classes, **eval(hyperparam) )

########################################################
# GNNE evaluation

from src.train import train_and_val
import configs
import src.gengraph as gengraph
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import utils.featgen as featgen
from types import SimpleNamespace
import numpy as np
from src.data import split_function

#prog_args = configs.arg_parse()

# Arguments
args_train_ratio = 0.6 
args_input_dim = 10

# Construct graph
#G, labels, name = gengraph.gen_syn1( feature_generator=featgen.ConstFeatureGen(np.ones(args_input_dim)) )
#G, labels, name = gengraph.gen_syn4( feature_generator=featgen.ConstFeatureGen(np.ones(args_input_dim, dtype=float)) )
#G, labels, name = gengraph.gen_syn5( feature_generator=featgen.ConstFeatureGen(np.ones(args_input_dim, dtype=float)) )
#G, labels, name = gengraph.gen_syn2()
#args_input_dim = len(G.nodes[0]["feat"])

# Create 
#data = prepare_data('syn4', 10)
data = SimpleNamespace()
data.x, data.edge_index, data.y = gengraph.preprocess_input_graph(G, labels)
data.num_classes = max(labels) + 1
data.num_features = data.x.shape[1]
data.num_nodes = G.number_of_nodes()
 	
# Train/test split only for nodes
data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())

# Pickle adjacency and feature vector for each dataset 

# Train model 
hyperparam = ''.join(['hparams_','syn4','_', 'GAT'])
param = ''.join(['params_','syn1','_', 'GAT'])
model = GCN(input_dim=data.x.size(1), output_dim= data.num_classes, **eval(hyperparam) )
train_and_val(model, data, **eval(param))

# Explain
explainer = GraphSHAP(data, model)
explanations = explainer.explain(node_index=0, 
									hops=2, 
									num_samples=100)

#

