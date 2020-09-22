from src.explainers import GraphLIME, SHAP
from src.models import GCN, GAT
import torch
from src.utils import *
from src.data import prepare_data


data = prepare_data('Cora', 10)
hyperparam = ''.join(['hparams_','Cora','_', 'GCN'])
param = ''.join(['params_','Cora','_', 'GCN'])
model = GCN(input_dim=data.x.size(1), output_dim= data.num_classes, **eval(hyperparam) )

args_K = 5
args_test_samples= 5

# Should loop over test samples (and classes)
node_indices = [1, 10, 20, 30, 40, 50]
num_noise_feats = []
#pred_class_num_noise_feats = []
true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[node_idx].max(dim=0)
node_idx = 1


explainer = SHAP(data, model)
coefs = explainer.explain(node_idx, data.x, data.edge_index)

explainer = GraphLIME(model, hop=2, rho=0.1)
coefs = explainer.explain(node_idx, data.x, data.edge_index) 

feat_indices = coefs.argsort()[-args_K:]
feat_indices = [idx for idx in feat_indices if coefs[idx] > 0.0]

num_noise_feat = sum(idx < args_test_samples for idx in feat_indices)
num_noise_feats.append(num_noise_feat)

#if i==predicted_class:
#pred_class_num_noise_feats.append(num_noise_feat)


from torch_geometric.nn import GNNExplainer
explainer = GNNExplainer(model, epochs=200)
node_idx = 10
node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)

# For nodes - use in GraphSHAP class to visualise important neighbours
ax, G = explainer.visualize_subgraph(node_idx, data.edge_index, edge_mask, y=data.y)
plt.show()




########################################################
# GNNE evaluation

from src.train import train_and_val
import configs
import src.gengraph as gengraph
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import utils.featgen as featgen
from types import SimpleNamespace

#prog_args = configs.arg_parse()

# Arguments
args_train_ratio = 0.6 
args_input_dim = 10

# Construct graph
data = SimpleNamespace()
G, labels, name = gengraph.gen_syn1( feature_generator=featgen.ConstFeatureGen(np.ones(10, dtype=float)) )
G, labels, name = gengraph.gen_syn4( feature_generator=featgen.ConstFeatureGen(np.ones(10, dtype=float)) )
G, labels, name = gengraph.gen_syn5( feature_generator=featgen.ConstFeatureGen(np.ones(10, dtype=float)) )
G, labels, name = gengraph.gen_syn2()
args_input_dim = len(G.nodes[0]["feat"])
df = gengraph.preprocess_input_graph(G, labels)
data.num_classes = max(labels) + 1

# Train/test split only for nodes
num_nodes = G.number_of_nodes()
num_train = int(num_nodes * args_train_ratio) 
idx = [i for i in range(num_nodes)]
np.random.shuffle(idx)
train_idx = idx[:num_train]
test_idx = idx[num_train:]

# Separate different components of data variable
data.x = torch.tensor(df["feat"], requires_grad=True, dtype=torch.float)[0]
edge_index = torch.tensor(df["adj"], dtype=torch.float)[0]
data.y = torch.tensor(df["labels"][:, train_idx], dtype=torch.long)[0]
 	
# Convert adjacency matrix to desired format
data.edge_index = torch.tensor([[],[]])
for i, row in enumerate(adj):
	for j, entry in enumerate(row): 
		if entry != 0:
			data.edge_index = torch.cat((data.edge_index,torch.tensor([[i],[j]])),dim=1)

# Pickle adjacency and feature vector for each dataset 

# Train model 
hyperparam = ''.join(['hparams_','syn','_', 'GCN'])
param = ''.join(['params_','Cora','_', 'GCN'])
model = GCN(input_dim=data.x.size(1), output_dim= data.num_classes, **eval(hyperparam) )

train_and_val(model, data, **eval(param))