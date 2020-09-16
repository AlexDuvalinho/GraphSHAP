COLOURS = ['g', 'b', 'r', 'p']


INPUT_DIM = {'Cora': 1433,
			'PubMed': 500,
			'Amazon':745,
			'PPI': 50,
			'Reddit': 602}

EVAL1_CORA = {'args_p':0.013,
			'args_binary':True,
			'args_num_noise_feat':200,
			'args_num_noise_nei':10}

EVAL1_PUBMED = {'args_p':0.1,
			'args_binary':False,
			'args_num_noise_feat':75,
			'args_num_noise_nei':10}

DIM_FEAT_P = {'Cora': 0.013,
			'PubMed':0.1}

# Model structure hyperparameters for Cora dataset, GCN model
hparams_Cora_GCN = {
		'hidden_dim': [16],
		'dropout': 0.5
		}

# Training hyperparameters for Cora dataset, GCN model 
params_Cora_GCN = {
	'num_epochs':50,
	'lr':0.01, 
	'wd':5e-4
	}

# Cora - GAT
hparams_Cora_GAT = {
		'hidden_dim': [8],
		'dropout': 0.6,
		'n_heads': [8,1]
		}

params_Cora_GAT = {
	'num_epochs':100,
	'lr':0.005, 
	'wd':5e-4
	}


# PubMed - GCN
hparams_PubMed_GCN = hparams_Cora_GCN 
params_PubMed_GCN = {
	'num_epochs':150,
	'lr':0.01, 
	'wd':5e-4
	}

# PubMed - GAT 
hparams_PubMed_GAT = {
		'hidden_dim': [8],
		'dropout': 0.6,
		'n_heads': [8,8]
		}

params_PubMed_GAT = {
	'num_epochs':120,
	'lr':0.005, 
	'wd':5e-4
	}
# suggested n_heads = [8,8] with more epochs, but not necessary and better in this case


# Amazon - GCN 
hparams_Amazon_GCN = {
		'hidden_dim': [32],
		'dropout': 0.5
		}

params_Amazon_GCN = params_PubMed_GCN

# Amazon - GAT
hparams_Amazon_GAT = hparams_PubMed_GAT 
params_Amazon_GAT = params_PubMed_GAT


# PPI - GCN
hparams_PPI_GCN = {
		'hidden_dim': [32,16],
		'dropout': 0.1
		}

params_PPI_GCN = {
	'num_epochs':20,
	'lr':0.01, 
	'wd':5e-4
	}

# PPI - GAT
# Change loss function as well to BCEWithLogits
hparams_PPI_GAT = {
		'hidden_dim': [256,256],
		'dropout': 0,
		'n_heads': [4,4,6]
		}

params_PPI_GAT = {
	'num_epochs':2000,
	'lr':0.005, 
	'wd':0
	}


import torch

def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
				   num_nodes=None, flow='source_to_target'):
	r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
	:attr:`node_idx`.
	It returns (1) the nodes involved in the subgraph, (2) the filtered
	:obj:`edge_index` connectivity, (3) the mapping from node indices in
	:obj:`node_idx` to their new location, and (4) the edge mask indicating
	which edges were preserved.
	Args:
		node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
			node(s).
		num_hops: (int): The number of hops :math:`k`.
		edge_index (LongTensor): The edge indices.
		relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
			:obj:`edge_index` will be relabeled to hold consecutive indices
			starting from zero. (default: :obj:`False`)
		num_nodes (int, optional): The number of nodes, *i.e.*
			:obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
		flow (string, optional): The flow direction of :math:`k`-hop
			aggregation (:obj:`"source_to_target"` or
			:obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
	:rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
			 :class:`BoolTensor`)
	"""

	def maybe_num_nodes(index, num_nodes=None):
		return index.max().item() + 1 if num_nodes is None else num_nodes

	num_nodes = maybe_num_nodes(edge_index, num_nodes)

	assert flow in ['source_to_target', 'target_to_source']
	if flow == 'target_to_source':
		row, col = edge_index
	else:
		col, row = edge_index

	node_mask = row.new_empty(num_nodes, dtype=torch.bool)
	edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

	if isinstance(node_idx, (int, list, tuple)):
		node_idx = torch.tensor([node_idx], device=row.device).flatten()
	else:
		node_idx = node_idx.to(row.device)

	subsets = [node_idx]

	for _ in range(num_hops):
		node_mask.fill_(False)
		node_mask[subsets[-1]] = True
		torch.index_select(node_mask, 0, row, out=edge_mask)
		subsets.append(col[edge_mask])

	subset, inv = torch.cat(subsets).unique(return_inverse=True)
	inv = inv[:node_idx.numel()]

	node_mask.fill_(False)
	node_mask[subset] = True
	edge_mask = node_mask[row] & node_mask[col]

	edge_index = edge_index[:, edge_mask]

	if relabel_nodes:
		node_idx = row.new_full((num_nodes, ), -1)
		node_idx[subset] = torch.arange(subset.size(0), device=row.device)
		edge_index = node_idx[edge_index]

	return subset, edge_index, inv, edge_mask
