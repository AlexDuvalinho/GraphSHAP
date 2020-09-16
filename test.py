from src.explainers import GraphLIME
from src.models import GCN, GAT
import torch
from src.utils import *
from src.data import prepare_data

data = prepare_data('Cora', 10)
hyperparam = ''.join(['hparams_','Cora','_', 'GCN'])
param = ''.join(['params_','Cora','_', 'GCN'])
model = GCN(input_dim=data.x.size(1), output_dim= data.num_classes, **eval(hyperparam) )
explainer = GraphLIME(model, hop=2, rho=0.1)
args_K = 5
args_test_samples= 5

# Should loop over test samples (and classes)
node_indices = [1, 10, 20, 30, 40, 50]
num_noise_feats = []
#pred_class_num_noise_feats = []
#true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[node_idx].max(dim=0)
node_idx = 1

coefs = explainer.explain(node_idx, data.x, data.edge_index) 

feat_indices = coefs.argsort()[-args_K:]
feat_indices = [idx for idx in feat_indices if coefs[idx] > 0.0]

num_noise_feat = sum(idx < args_test_samples for idx in feat_indices)
num_noise_feats.append(num_noise_feat)

#if i==predicted_class:
#pred_class_num_noise_feats.append(num_noise_feat)