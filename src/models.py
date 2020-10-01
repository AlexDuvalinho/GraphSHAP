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


"""
class GAT(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, dropout, n_heads):
		super(GAT, self).__init__()
		self.dropout = dropout

		self.conv_in = GATConv(input_dim, hidden_dim[0], heads=n_heads[0], dropout=self.dropout)
		self.conv = [GATConv(hidden_dim[i-1] * n_heads[i-1], hidden_dim[i], heads=n_heads[i], dropout=self.dropout) for i in range(1,len(n_heads)-1)]
		self.conv_out = GATConv(hidden_dim[-1] * n_heads[-2], output_dim, heads=n_heads[-1], dropout=self.dropout, concat=False)

	def forward(self, x, edge_index, att=None):
		x = F.dropout(x, p=self.dropout, training=self.training)
		
		if att: # if we want to see attention weights
			x, alpha = self.conv_in(x, edge_index, return_attention_weights=att)
			x = F.elu(x)

			for attention in self.conv:
				x = F.dropout(x, p=self.dropout, training=self.training)
				x = F.elu(attention(x, edge_index))
			
			x = F.dropout(x, p=self.dropout, training=self.training)
			x, alpha2 = self.conv_out(x, edge_index, return_attention_weights=att)
		
			return F.log_softmax(x, dim=1), alpha, alpha2
		
		else: 
			x = self.conv_in(x, edge_index)
			x = F.elu(x)

			for attention in self.conv:
				x = F.dropout(x, p=self.dropout, training=self.training)
				x = F.elu(attention(x, edge_index))
			
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = self.conv_out(x, edge_index)
		
			return F.log_softmax(x, dim=1)
"""

class GAT(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, dropout, n_heads):
		super(GAT, self).__init__()
		self.dropout = dropout

		self.conv_in = GATConv(
			input_dim, hidden_dim[0], heads=n_heads[0], dropout=self.dropout)


		self.linear = nn.Linear(60, output_dim)

	def forward(self, x, edge_index, att=None):
		x = F.dropout(x, p=self.dropout, training=self.training)

		if att:  # if we want to see attention weights

			x, alpha = self.conv_in(x, edge_index, return_attention_weights=att)
			x = F.elu(x)

			x = self.linear(x)

			return F.log_softmax(x, dim=1), alpha, alpha

		else:
			x = self.conv_in(x, edge_index)
			x = F.elu(x)
			x = self.linear(x)

			return F.log_softmax(x, dim=1)

