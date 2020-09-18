import seaborn as sns
import matplotlib.pyplot as plt

from copy import copy
from math import sqrt
import torch
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx


def plot_dist(noise_feats, label=None, ymax=1.1, color=None, title=None, save_path=None):
	"""
	Kernel density plot of the number of noisy features included in explanations, 
	for a certain number of test samples
	"""
	if not any(noise_feats): # handle special case where noise_feats=0 
		noise_feats[0]=1

	sns.set_style('darkgrid')
	ax = sns.distplot(noise_feats, hist=False, kde=True, kde_kws={'label': label}, color=color)
	plt.xlim(-3, 11)
	plt.ylim(ymin=0.0, ymax=ymax)

	if title:
		plt.title(title)
		
	if save_path:
		plt.savefig(save_path)

	return ax

def __flow__(model):
	for module in model.modules():
		if isinstance(module, MessagePassing):
			return module.flow
	return 'source_to_target'

def visualize_subgraph(model, node_idx, edge_index, edge_mask, num_hops, y=None,
						   threshold=None, **kwargs):
		"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
		:attr:`edge_mask`.

		Args:
			node_idx (int): The node id to explain.
			edge_index (LongTensor): The edge indices.
			edge_mask (Tensor): The edge mask.
			y (Tensor, optional): The ground-truth node-prediction labels used
				as node colorings. (default: :obj:`None`)
			threshold (float, optional): Sets a threshold for visualizing
				important edges. If set to :obj:`None`, will visualize all
				edges with transparancy indicating the importance of edges.
				(default: :obj:`None`)
			**kwargs (optional): Additional arguments passed to
				:func:`nx.draw`.

		:rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
		"""

		assert edge_mask.size(0) == edge_index.size(1)

		# Only operate on a k-hop subgraph around `node_idx`.
		subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
			node_idx, num_hops, edge_index, relabel_nodes=True,
			num_nodes=None, flow=__flow__(model))

		edge_mask = edge_mask[hard_edge_mask]

		if threshold is not None:
			edge_mask = (edge_mask >= threshold).to(torch.float)

		if y is None:
			y = torch.zeros(edge_index.max().item() + 1,
							device=edge_index.device)
		else:
			y = y[subset].to(torch.float) / y.max().item()

		data = Data(edge_index=edge_index, att=edge_mask, y=y,
					num_nodes=y.size(0)).to('cpu')
		G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
		mapping = {k: i for k, i in enumerate(subset.tolist())}
		G = nx.relabel_nodes(G, mapping)

		node_kwargs = copy(kwargs)
		node_kwargs['node_size'] = kwargs.get('node_size') or 800
		node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

		label_kwargs = copy(kwargs)
		label_kwargs['font_size'] = kwargs.get('font_size') or 10

		pos = nx.spring_layout(G)
		ax = plt.gca()
		for source, target, data in G.edges(data=True):
			ax.annotate(
				'', xy=pos[target], xycoords='data', xytext=pos[source],
				textcoords='data', arrowprops=dict(
					arrowstyle="->",
					alpha=max(data['att'], 0.1),
					shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
					shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
					connectionstyle="arc3,rad=0.1",
				))
		nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **node_kwargs)
		nx.draw_networkx_labels(G, pos, **label_kwargs)

		return ax, G