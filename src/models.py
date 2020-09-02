import torch.nn as nn
import torch.nn.functional as F

# Use already implemented GCN
from torch_geometric.nn import GCNConv, GATConv


class GCN(nn.Module):
	"""
	Construct a GNN with 2 GCN blocks
	"""
	def __init__(self, input_dim, hidden_dim, output_dim, dropout):
		super(GCN, self).__init__()

		self.dropout = dropout
		self.conv_in = GCNConv(input_dim, hidden_dim[0])
		self.conv = [GCNConv(hidden_dim[i-1], hidden_dim[i]) for i in range(1,len(hidden_dim))]
		self.conv_out = GCNConv(hidden_dim[-1], output_dim)

	def forward(self, x, edge_index):
		x = F.relu(self.conv_in(x, edge_index))
		x = F.dropout(x, p=self.dropout, training=self.training)
		
		for block in self.conv:
			x = F.relu(block(x, edge_index))
			x = F.dropout(x, p=self.dropout, training=self.training)

		x = self.conv_out(x,edge_index)

		return F.log_softmax(x, dim=1)


class GAT(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, dropout, n_heads):
		super(GAT, self).__init__()
		self.dropout = dropout

		self.conv_in = GATConv(input_dim, hidden_dim[0], heads=n_heads[0], dropout=self.dropout)
		self.conv = [GATConv(hidden_dim[i-1] * n_heads[i-1], hidden_dim[i], heads=n_heads[i], dropout=self.dropout) for i in range(1,len(n_heads)-1)]
		self.conv_out = GATConv(hidden_dim[-1] * n_heads[-2], output_dim, heads=n_heads[-1], dropout=self.dropout)

	def forward(self, x, edge_index):
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = F.elu(self.conv_in(x, edge_index))

		for attention in self.conv:
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = F.elu(attention(x, edge_index))
		
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.conv_out(x, edge_index)
	
		return F.log_softmax(x, dim=1)


class Net(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, dropout, n_heads):
		super(Net, self).__init__()
		self.conv1 = GATConv(input_dim, hidden_dim, heads=n_heads[0])
		self.lin1 = torch.nn.Linear(train_dataset.num_features, n_heads[0] * hidden_dim)
		self.conv2 = GATConv(n_heads[0] * hidden_dim, hidden_dim, heads=n_heads[1])
		self.lin2 = torch.nn.Linear(n_heads[1]* hidden_dim, n_heads[1] * hidden_dim)
		self.conv3 = GATConv(
			n_heads[1] * hidden_dim, train_dataset.num_classes, heads=n_heads[2], concat=False)
		self.lin3 = torch.nn.Linear(n_heads[2] * hidden_dim, train_dataset.num_classes)

	def forward(self, x, edge_index):
		ipdb.set_trace()

		x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
		x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
		x = self.conv3(x, edge_index) + self.lin3(x)
		return x









"""
class GraphSAGE(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels):
		super(SAGE, self).__init__()

		self.num_layers = 2

		self.convs = torch.nn.ModuleList()
		self.convs.append(SAGEConv(in_channels, hidden_channels))
		self.convs.append(SAGEConv(hidden_channels, out_channels))

	def forward(self, x, adjs):
		# `train_loader` computes the k-hop neighborhood of a batch of nodes,
		# and returns, for each layer, a bipartite graph object, holding the
		# bipartite edges `edge_index`, the index `e_id` of the original edges,
		# and the size/shape `size` of the bipartite graph.
		# Target nodes are also included in the source nodes so that one can
		# easily apply skip-connections or add self-loops.
		for i, (edge_index, _, size) in enumerate(adjs):
			x_target = x[:size[1]]  # Target nodes are always placed first.
			x = self.convs[i]((x, x_target), edge_index)
			if i != self.num_layers - 1:
				x = F.relu(x)
				x = F.dropout(x, p=0.5, training=self.training)
		return x.log_softmax(dim=-1)

	def inference(self, x_all):
		pbar = tqdm(total=x_all.size(0) * self.num_layers)
		pbar.set_description('Evaluating')

		# Compute representations of nodes layer by layer, using *all*
		# available edges. This leads to faster computation in contrast to
		# immediately computing the final representations of each batch.
		for i in range(self.num_layers):
			xs = []
			for batch_size, n_id, adj in subgraph_loader:
				edge_index, _, size = adj.to(device)
				x = x_all[n_id].to(device)
				x_target = x[:size[1]]
				x = self.convs[i]((x, x_target), edge_index)
				if i != self.num_layers - 1:
					x = F.relu(x)
				xs.append(x.cpu())

				pbar.update(batch_size)

			x_all = torch.cat(xs, dim=0)

		pbar.close()

		return x_all
"""