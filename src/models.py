""" models.py

	Define 2 GNN models made of GCN and GAT layers respectively
"""

from torch_geometric.nn import GCNConv, GATConv
import torch_geometric.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    """
    Construct a GNN with several Graph Convolution blocks
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()

        self.dropout = dropout
        self.conv_in = GCNConv(input_dim, hidden_dim[0])
        self.conv = [GCNConv(hidden_dim[i-1], hidden_dim[i])
                     for i in range(1, len(hidden_dim))]
        self.conv_out = GCNConv(hidden_dim[-1], output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv_in(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for block in self.conv:
            x = F.relu(block(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv_out(x, edge_index)

        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    """
    Contruct a GNN with several Graph Attention layers 
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, n_heads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.conv_in = GATConv(
            input_dim, hidden_dim[0], heads=n_heads[0], dropout=self.dropout)
        self.conv = [GATConv(hidden_dim[i-1] * n_heads[i-1], hidden_dim[i],
                             heads=n_heads[i], dropout=self.dropout) for i in range(1, len(n_heads)-1)]
        self.conv_out = GATConv(
            hidden_dim[-1] * n_heads[-2], output_dim, heads=n_heads[-1], dropout=self.dropout, concat=False)

    def forward(self, x, edge_index, att=None):
        x = F.dropout(x, p=self.dropout, training=self.training)

        if att:  # if we want to see attention weights
            x, alpha = self.conv_in(
                x, edge_index, return_attention_weights=att)
            x = F.elu(x)

            for attention in self.conv:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.elu(attention(x, edge_index))

            x = F.dropout(x, p=self.dropout, training=self.training)
            x, alpha2 = self.conv_out(
                x, edge_index, return_attention_weights=att)

            return F.log_softmax(x, dim=1), alpha, alpha2

        else:  # we don't consider attention weights
            x = self.conv_in(x, edge_index)
            x = F.elu(x)

            for attention in self.conv:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.elu(attention(x, edge_index))

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv_out(x, edge_index)

            return F.log_softmax(x, dim=1)


class LinearRegressionModel(nn.Module):
    """Construct a simple linear regression

    """

    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred =self.linear1(x)
        return y_pred



class GCNNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, add_self=False, args=None):
        super(GCNNet, self).__init__()
        self.input_dim = input_dim
        print('GCNNet input_dim:', self.input_dim)
        self.hidden_dim = hidden_dim
        print('GCNNet hidden_dim:', self.hidden_dim)
        self.label_dim = label_dim
        print('GCNNet label_dim:', self.label_dim)
        self.num_layers = num_layers
        print('GCNNet num_layers:', self.num_layers)

        # self.concat = concat
        # self.bn = bn
        # self.add_self = add_self
        self.args = args
        self.dropout = dropout
        self.act = F.relu
        self.celloss = torch.nn.CrossEntropyLoss()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(self.input_dim, self.hidden_dim))
        for layer in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        self.linear = torch.nn.Linear(
            len(self.convs) * self.hidden_dim, self.label_dim)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        print('len(self.convs):', len(self.convs))

        # Init weights
        for conv in self.convs:
            torch.nn.init.xavier_uniform_(conv.weight.data)  # .data

    def forward(self, x, edge_index):
        x_all = []

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_all.append(x)
        x = torch.cat(x_all, dim=1)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return self.celloss(pred, label)


