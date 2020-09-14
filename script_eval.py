from src.eval import filter_useless_features, filter_useless_nodes
from src.utils import DIM_FEAT_P
import torch


# Argument parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default= 'GAT', 
							help= "Name of the GNN: GCN or GAT")
parser.add_argument("--dataset", type=str, default= 'PubMed',
							help= "Name of the dataset among Cora, PubMed, Amazon, PPI, Reddit")
parser.add_argument("--hops", type=int, default=2,
							help= 'k of k-hops neighbours considered for the node of interest')
parser.add_argument("--num_samples", type=int, default=100,
							help= 'number of coalitions sampled')
parser.add_argument("--test_samples", type=int, default=20,
							help='number of test samples for evaluation')
parser.add_argument("--K", type=int, default=5,
							help= 'number of most important features considered')
parser.add_argument("--num_noise_feat", type=int, default=10,
							help='number of noisy features')
parser.add_argument("--num_noise_nei", type=int, default=0,
							help= 'number of noisy nodes')
parser.add_argument("--p", type=float, default=0.5,
							help= 'proba of existance for each feature, if binary')
parser.add_argument("--binary", type=bool, default=True,
							help= 'if noisy features are binary or not')
parser.add_argument("--connectedness", type=str, default='low',
							help= 'how connected are the noisy nodes we define')
args = parser.parse_args()



#node_indices= [2420,2455,1783,2165,2628,1822,2682,2261,1896,1880,2137,2237,2313,2218,1822,1719,1763,2263,2020,1988]
#node_indices = [10, 18, 89, 178, 333, 356, 378, 456, 500, 2222, 1220, 1900, 1328, 189, 1111]
#node_indices = [1834,2512,2591,2101,1848,1853,2326,1987,2359,2453,2230,2267,2399, 2150,2400]
node_indices = None
torch.manual_seed(10)

# Node features
noisy_feat_included = filter_useless_features(args.model,
											args.dataset,
											args.hops,
											args.num_samples,
											args.test_samples, 
											args.K,
											args.num_noise_feat,
											args.p,
											args.binary,
											node_indices,
											info=True)

# Neighbours
noisy_nei_included = filter_useless_nodes(args.model,
										args.dataset,
										args.hops,
										args.num_samples,
										args.test_samples, 
										args.K,
										args.num_noise_nei,
										args.p,
										args.binary,
										args.connectedness,
										node_indices,
										info=True)
