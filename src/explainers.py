""" explainers.py

    Define the different explainers: GraphSHAP + benchmarks
"""
from sklearn.linear_model import LinearRegression

from src.train import accuracy
from src.models import LinearRegressionModel
from sklearn.metrics import r2_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import warnings
import time
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import networkx as nx
import torch
import torch_geometric
from sklearn.linear_model import LassoLarsCV, LassoLars, Lasso, Ridge
from itertools import combinations

# GraphLIME
from src.plots import visualize_subgraph, k_hop_subgraph, denoise_graph, log_graph

# GNNExplainer
from torch_geometric.nn import GNNExplainer as GNNE
from torch_geometric.nn import MessagePassing

warnings.filterwarnings("ignore")



class GraphSHAP():

    def __init__(self, data, model, gpu=False):
        self.model = model
        self.data = data
        self.gpu = gpu
        self.F = None  # number of non zero node features
        self.neighbours = None  # neighbours considered
        self.M = None  # number of nonzero features - for each node index

        self.model.eval()

    def explain(self, 
                node_indexes=[0], 
                hops=2, 
                num_samples=10,
                info=True,
                 multiclass=False,
                args_hv='compute_pred',
                args_feat='Expectation', 
                args_coal='Smarter', 
                args_g = 'WLS',
                regu=None,
                vizu=False):
        """ Explain prediction for a particular node - GraphSHAP method

        Args:
            node_index (int, optional): index of the node of interest. Defaults to 0.
            hops (int, optional): number k of k-hop neighbours to consider in the subgraph 
                                                    around node_index. Defaults to 2.
            num_samples (int, optional): number of samples we want to form GraphSHAP's new dataset. 
                                                    Defaults to 10.
            info (bool, optional): Print information about explainer's inner workings. 
                                                    And include vizualisation. Defaults to True.
            args_hv (str, optional): strategy used to convert simplified input z' to original
                                                    input space z
            args_feat (str, optional): way to switch off and discard node features (0 or expectation)
            args_coal (str, optional): how we sample coalitions z'
            args_g (str, optional): method used to train model g on (z', f(z))
            multiclass (bool, optional): extension - consider predicted class only or all classes
            regu (int, optional): extension - apply regularisation to balance importance granted
                                                    to nodes vs features

        Returns:
                [type]: shapley values for features/neighbours that influence node v's pred
        """	
        # Time
        start = time.time()

        # Accept a subset of nodes for explanations
        phi_list = []
        for node_index in node_indexes: 

            # Compute true prediction of model, for original instance
            if self.gpu: 
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
                with torch.no_grad():
                    true_conf, true_pred = self.model(
                        x=self.data.x.cuda(), 
                        edge_index=self.data.edge_index.cuda()).exp()[node_index].max(dim=0)
            else: 
                with torch.no_grad():
                    true_conf, true_pred = self.model(
                        x=self.data.x, 
                        edge_index=self.data.edge_index).exp()[node_index].max(dim=0)

            # Construct the k-hop subgraph of the node of interest (v)
            self.neighbours, _, _, edge_mask =\
                torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
                                                    num_hops=hops,
                                                    edge_index=self.data.edge_index)
            # Stores the indexes of the neighbours of v (+ index of v itself)

            # Retrieve 1-hop neighbours of v
            one_hop_neighbours, _, _, _ =\
                        torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
                                                            num_hops=1,
                                                            edge_index=self.data.edge_index)

            # Determine z': features and neighbours whose importance is investigated
            discarded_feat_idx = []
            # Consider only non-zero entries in the subgraph of v
            if args_feat == 'Null':
                feat_idx = self.data.x[self.neighbours, :].mean(axis=0).nonzero()
                self.F = feat_idx.size()[0]

            # Consider all features (+ use expectation like below)
            elif args_feat == 'All':
                self.F = self.data.x[node_index, :].shape[0]
                feat_idx = torch.unsqueeze(torch.arange(self.data.x.size(0)), 1)

            # Consider only features whose aggregated value is different from expected one
            else: 
                # Stats dataset
                std = self.data.x.std(axis=0)
                mean = self.data.x.mean(axis=0)
                # Feature intermediate rep
                mean_subgraph = self.data.x[self.neighbours,:].mean(axis=0)
                # Select relevant features only - (E-e,E+e)
                mean_subgraph = torch.where(mean_subgraph > mean - 0.25*std, mean_subgraph,
                            torch.ones_like(mean_subgraph)*100)
                mean_subgraph = torch.where(mean_subgraph < mean + 0.25*std, mean_subgraph,
                            torch.ones_like(mean_subgraph)*100)
                feat_idx = (mean_subgraph == 100).nonzero()
                discarded_feat_idx = (mean_subgraph != 100).nonzero()
                self.F = feat_idx.shape[0]
                del mean, mean_subgraph, std
            
            # Potentially do a feature selection with Lasso (or otherwise)
            # Long process

            # Remove node v index from neighbours and store their number in D
            self.neighbours = self.neighbours[self.neighbours != node_index]
            D = self.neighbours.shape[0]

            # Total number of features + neighbours considered for node v
            self.M = self.F+D
            
            # Def range of endcases considered 
            args_K = 5

            ### COALITIONS: sample z' - binary vector of dimension (num_samples, M)
            z_ = eval('self.' + args_coal)(num_samples, args_K, regu)
            
            # Compute |z'| for each sample z': number of non-zero entries
            s = (z_ != 0).sum(dim=1)

            ### GRAPHSHAP KERNEL: define weights associated with each sample 
            weights = self.shapley_kernel(s)
            # TODO: remove when tests are finished
            if max(weights) > 9 and info:
                print('!! Empty or/and full coalition is included !!')
                
            ### H_V: Create dataset (z', f(hv(z'))=(z', f(z)), stored as (z_, fz)
            # Retrive z from z' and x_v, then compute f(z)
            fz = eval('self.' + args_hv)(node_index, num_samples, D, z_,
                                feat_idx, one_hop_neighbours, args_K, args_feat, 
                                discarded_feat_idx, multiclass, true_pred)
            
            ### g: Weighted Linear Regression to learn shapley values
            phi, base_value = eval('self.' + args_g)(z_, weights, fz, multiclass, info)		
            if info:	
                print('Base value', base_value, 'for class ', true_pred.item())

            ### REGU
            if type(regu)==int and not multiclass: 
                expl = np.array(true_conf.cpu() - base_value)
                phi[:self.F] = (regu * expl / sum(phi[:self.F])) * phi[:self.F]
                phi[self.F:] = ((1-regu) * expl / sum(phi[self.F:]) ) * phi[self.F:]

            ### PRINT some information
            if info:
                self.print_info(D, node_index, phi, feat_idx, true_pred, true_conf, multiclass)

            ### VISUALISATION
            if vizu:
                self.vizu(edge_mask, node_index, phi, true_pred, hops, multiclass)
            
            # Time
            # TODO: remove after tests
            end = time.time()
            if info:
                print('Time: ', end - start)

            # Append explanations for this node to list of expl.
            phi_list.append(phi)

        return phi_list

    ################################  
    # Coalition sampler
    ################################
    def SmarterRegu(self, num_samples, args_K, regu):
        """ Coalition sampling that favour neighbours or features 

        """
        if regu=='None':
            z_ = self.Smarter(num_samples, args_K, regu) 
            return z_

        # Favour features - special coalitions don't study node's effect
        elif regu > 0.5:
            # Define empty and full coalitions
            self.M = self.F
            z_ = torch.ones(num_samples, self.M)
            z_[1::2] = torch.zeros(num_samples//2, self.M)
            # z_[1, :] = torch.empty(1, self.M).random_(2)
            i = 2 
            k = 1
            # Loop until all samples are created
            while i < num_samples:
                # Look at each feat/nei individually if have enough sample
                # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
                if i + 2 * self.F < num_samples and k == 1:
                    z_[i:i+self.F, :] = torch.ones(self.F, self.M)
                    z_[i:i+self.F, :].fill_diagonal_(0)
                    z_[i+self.F:i+2*self.F, :] = torch.zeros(self.F, self.M)
                    z_[i+self.F:i+2*self.F, :].fill_diagonal_(1)
                    i += 2 * self.F
                    k += 1

                else:
                    # Split in two number of remaining samples
                    # Half for specific coalitions with low k and rest random samples
                    samp = i + 2*(num_samples - i)//3
                    while i<samp and k<=min(args_K, self.F):
                        # Sample coalitions of k1 neighbours or k1 features without repet and order. 
                        L = list( combinations(range(self.F),k) )
                        random.shuffle(L)
                        L = L[:samp+1]

                        for j in range(len(L)):
                            # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                            z_[i, L[j]] = torch.zeros(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                                return z_
                            # Coalitions (No nei, k feat) or (No feat, k nei)
                            z_[i, L[j]] = torch.ones(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                                return z_
                        k += 1

                    # Sample random coalitions 
                    z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                    return z_
            return z_
        
        # Favour neighbour
        else: 
            # Define empty and full coalitions
            D = len(self.neighbours)
            self.M = D
            self.F = 0 
            z_ = torch.ones(num_samples, self.M)
            z_[1::2] = torch.zeros(num_samples//2, self.M)
            i = 2 
            k = 1
            # Loop until all samples are created
            while i < num_samples:
                # Look at each feat/nei individually if have enough sample
                # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
                if i + 2 * D < num_samples and k == 1:
                    z_[i:i+D, :] = torch.ones(D, self.M)
                    z_[i:i+D, :].fill_diagonal_(0)
                    z_[i+D:i+2*D, :] = torch.zeros(D, self.M)
                    z_[i+D:i+2*D, :].fill_diagonal_(1)
                    i += 2 * D
                    k += 1

                else:
                    # Split in two number of remaining samples
                    # Half for specific coalitions with low k and rest random samples
                    samp = i + 2*(num_samples - i)//3
                    while i<samp and k<=min(args_K, D):
                        # Sample coalitions of k1 neighbours or k1 features without repet and order. 
                        L = list( combinations(range(self.F,self.M), k) )
                        random.shuffle(L)
                        L = L[:samp+1]

                        for j in range(len(L)):
                            # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                            z_[i, L[j]] = torch.zeros(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                                return z_
                            # Coalitions (No nei, k feat) or (No feat, k nei)
                            z_[i, L[j]] = torch.ones(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                                return z_
                        k += 1

                    # Sample random coalitions 
                    z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                    return z_
            return z_

    def SmarterPlus(self, num_samples, args_K, *unused):
        """ Sample coalitions cleverly given shapley kernel def
        Consider nodes and features separately to better capture their effect

        Args:
            num_samples ([int]): total number of coalitions z_
            args_K: max size of coalitions favoured in sampling 

        Returns:
            [tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
        """
        # Define empty and full coalitions
        z_ = torch.ones(num_samples, self.M)
        z_[1::2] = torch.zeros(num_samples//2, self.M)
        # No coalitions
        i = 0 
        k = 1
        # Loop until all samples are created
        while i < num_samples:
            # Look at each feat/nei individually if have enough sample
            # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
            if i + 2 * self.M < num_samples and k == 1:
                z_[i:i+self.M, :] = torch.ones(self.M, self.M)
                z_[i:i+self.M, :].fill_diagonal_(0)
                z_[i+self.M:i+2*self.M, :] = torch.zeros(self.M, self.M)
                z_[i+self.M:i+2*self.M, :].fill_diagonal_(1)
                i += 2 * self.M
                k += 1

            else:
                # Split in two number of remaining samples
                # Half for specific coalitions with low k and rest random samples
                samp = i + 2*(num_samples - i)//3
                while i<samp and k<=min(args_K, self.F, self.M-self.F):
                    # Sample coalitions of k1 neighbours or k1 features without repet and order. 
                    L = list( combinations(range(self.F),k) ) + list( combinations(range(self.F,self.M), k) )
                    random.shuffle(L)
                    L = L[:samp+1]

                    for j in range(len(L)):
                        # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                        z_[i, L[j]] = torch.zeros(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                            return z_
                        # Coalitions (No nei, k feat) or (No feat, k nei)
                        z_[i, L[j]] = torch.ones(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                            return z_
                    k += 1

                # Sample random coalitions 
                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
        
        return z_

    def Smarter(self, num_samples, args_K, *unused):
        """ Sample coalitions cleverly given shapley kernel def
        Consider nodes and features separately to better capture their effect

        Args:
            num_samples ([int]): total number of coalitions z_
            args_K: max size of coalitions favoured in sampling 

        Returns:
            [tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
        """
        # Define empty and full coalitions
        z_ = torch.ones(num_samples, self.M)
        z_[1::2] = torch.zeros(num_samples//2, self.M)
        # z_[1, :] = torch.empty(1, self.M).random_(2)
        i = 2 
        k = 1
        # Loop until all samples are created
        while i < num_samples:
            # Look at each feat/nei individually if have enough sample
            # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
            if i + 2 * self.M < num_samples and k == 1:
                z_[i:i+self.M, :] = torch.ones(self.M, self.M)
                z_[i:i+self.M, :].fill_diagonal_(0)
                z_[i+self.M:i+2*self.M, :] = torch.zeros(self.M, self.M)
                z_[i+self.M:i+2*self.M, :].fill_diagonal_(1)
                i += 2 * self.M
                k += 1

            else:
                # Split in two number of remaining samples
                # Half for specific coalitions with low k and rest random samples
                samp = i + 2*(num_samples - i)//3
                while i<samp and k<=min(args_K, self.F, self.M-self.F):
                    # Sample coalitions of k1 neighbours or k1 features without repet and order. 
                    L = list( combinations(range(self.F),k) ) + list( combinations(range(self.F,self.M), k) )
                    random.shuffle(L)
                    L = L[:samp+1]

                    for j in range(len(L)):
                        # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                        z_[i, L[j]] = torch.zeros(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                            return z_
                        # Coalitions (No nei, k feat) or (No feat, k nei)
                        z_[i, L[j]] = torch.ones(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                            return z_
                    k += 1

                # Sample random coalitions 
                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                return z_
        return z_

    def Smart(self, num_samples, *unused):
        """ Sample coalitions cleverly given shapley kernel def

        Args:
            num_samples ([int]): total number of coalitions z_

        Returns:
            [tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
        """
        z_ = torch.ones(num_samples, self.M)
        z_[1::2] = torch.zeros(num_samples//2, self.M)
        k = 1
        i = 2
        while i < num_samples:
            if i + 2 * self.M < num_samples and k == 1:
                z_[i:i+self.M, :] = torch.ones(self.M, self.M)
                z_[i:i+self.M, :].fill_diagonal_(0)
                z_[i+self.M:i+2*self.M, :] = torch.zeros(self.M, self.M)
                z_[i+self.M:i+2*self.M, :].fill_diagonal_(1)
                i += 2 * self.M
                k += 1
            elif k == 1:
                M = list(range(self.M))
                random.shuffle(M)
                for j in range(self.M):
                    z_[i, M[j]] = torch.zeros(1)
                    i += 1
                    if i == num_samples:
                        return z_
                    z_[i, M[j]] = torch.ones(1)
                    i += 1
                    if i == num_samples:
                        return z_
                k += 1
            elif k == 2:
                M = list(combinations(range(self.M), 2))[:num_samples-i+1]
                random.shuffle(M)
                for j in range(len(M)):
                    z_[i, M[j][0]] = torch.tensor(0)
                    z_[i, M[j][1]] = torch.tensor(0)
                    i += 1
                    if i == num_samples:
                        return z_
                    z_[i, M[j][0]] = torch.tensor(1)
                    z_[i, M[j][1]] = torch.tensor(1)
                    i += 1
                    if i == num_samples:
                        return z_
                k += 1
            else:
                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                return z_

        return z_
    
    def Random(self, num_samples, *unused):
        z_ = torch.empty(num_samples, self.M).random_(2)
        # z_[0, :] = torch.ones(self.M)
        # z_[1, :] = torch.zeros(self.M)
        return z_
    
    ################################
    # GraphSHAP kernel
    ################################
    def shapley_kernel(self, s):
        """ Computes a weight for each newly created sample 

        Args:
            s (tensor): contains dimension of z' for all instances
                (number of features + neighbours included)

        Returns:
                [tensor]: shapley kernel value for each sample
        """
        shap_kernel = []
        # Loop around elements of s in order to specify a special case
        # Otherwise could have procedeed with tensor s direclty
        for i in range(s.shape[0]):
            a = s[i].item()
            # Put an emphasis on samples where all or none features are included
            if a == 0 or a == self.M:
                shap_kernel.append(1000)
            elif scipy.special.binom(self.M, a) == float('+inf'):
                shap_kernel.append(1)
            else:
                shap_kernel.append(
                    (self.M-1)/(scipy.special.binom(self.M, a)*a*(self.M-a)))
        return torch.tensor(shap_kernel)

    ################################
    # COMPUTE PREDICTIONS f(z)
    ################################
    def compute_pred(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z from z' and compute prediction f(z) for each sample z'
            In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

        Args: 
                Variables are defined exactly as defined in explainer function 

        Returns: 
                (tensor): f(z) - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses. 
        """
        # To networkx 
        G = torch_geometric.utils.to_networkx(self.data)

        # We need to recover z from z' - wrt sampled neighbours and node features
        # Initialise new node feature vectors and neighbours to disregard
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x.size(1))
        else: 
            av_feat_values = self.data.x.mean(dim=0)
            # 'All' and 'Expectation'
        
        # Store nodes and features not sampled
        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z'
        for i in range(num_samples):

            # Define new node features dataset (we only modify x_v for now)
            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Define new neighbourhood
            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z) for graphshap dataset - consider all classes
        if multiclass: 
            fz = torch.zeros((num_samples, self.data.num_classes))
        else: 
            fz = torch.zeros(num_samples)
        # classes_labels = torch.zeros(num_samples)
        # pred_confidence = torch.zeros(num_samples)

        # Create new matrix A and X - for each sample ≈ reform z from z'
        for (key, ex_nei), (_, ex_feat)  in tqdm(zip(excluded_nei.items(), excluded_feat.items())):

            positions = []
            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            for val in ex_nei:
                pos = (self.data.edge_index == val).nonzero()[:, 1].tolist()
                positions += pos
            # Create new adjacency matrix for that sample
            positions = list(set(positions))
            A = np.array(self.data.edge_index)
            # Special case - consider only feat. influence if too few nei included
            if self.M - self.F - len(ex_nei) >= min(self.F - len(ex_feat), args_K):
                A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)

            # Change feature vector for node of interest - excluded and discarded features
            X = deepcopy(self.data.x)
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            if discarded_feat_idx!=[] and len(self.neighbours) - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]
                # May delete - should be an approximation
                #if args_feat == 'Expectation':
                #    for val in discarded_feat_idx:
                #        X[self.neighbours, val] = av_feat_values[val].repeat(D)


            # Special case - consider only nei. influence if too few feat included
            if self.F - len(ex_feat) < min(self.M - self.F - len(ex_nei), args_K):
                
                # Look at the 2-hop neighbours included
                # Make sure that they are connected to v (with current nodes sampled nodes)
                included_nei = set(
                    self.neighbours.detach().numpy()).difference(ex_nei)
                included_nei = included_nei.difference(
                    one_hop_neighbours.detach().numpy())
                
                for incl_nei in included_nei:
                    l = nx.shortest_path(G, source=node_index, target=incl_nei)
                    if set(l[1:-1]).isdisjoint(ex_nei):
                        pass
                    else:
                        for n in range(1, len(l)-1):
                            A = torch.cat((A, torch.tensor(
                                [[l[n-1]], [l[n]]])), dim=-1)
                            X[l[n], :] = av_feat_values

            # Usual case - exclude features for the whole subgraph
            else:
                for val in ex_feat:
                    X[self.neighbours, val] = av_feat_values[val].repeat(D)  # 0
                
            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model(x=X.cuda(), edge_index=A.cuda()).exp()[node_index]
            else: 
                with torch.no_grad():
                    proba = self.model(x=X, edge_index=A).exp()[node_index]

            # Store final class prediction and confience level
            # pred_confidence[key], classes_labels[key] = torch.topk(proba, k=1) 

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else: 
                fz[key] = proba[true_pred]

        return fz

    def basic_default_2hop(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z from z' and compute prediction f(z) for each sample z'
            In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

        Args:
                Variables are defined exactly as defined in explainer function

        Returns:
                (tensor): f(z) - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses.
        """
        G = torch_geometric.utils.to_networkx(self.data)

        # We need to recover z from z' - wrt sampled neighbours and node features
        # Initialise new node feature vectors and neighbours to disregard
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x.size(1))
        else:
            av_feat_values = self.data.x.mean(dim=0)

        # or random feature vector made of random value across each col of X
        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z'
        for i in tqdm(range(num_samples)):

            # Define new node features dataset (we only modify x_v for now)
            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Define new neighbourhood
            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z) for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Create new matrix A and X - for each sample ≈ reform z from z'
        for (key, ex_nei), (_, ex_feat) in zip(excluded_nei.items(), excluded_feat.items()):

            positions = []
            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            for val in ex_nei:
                pos = (self.data.edge_index == val).nonzero()[:, 1].tolist()
                positions += pos
            # Create new adjacency matrix for that sample
            positions = list(set(positions))
            A = np.array(self.data.edge_index)
            A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)

            # Change feature vector for node of interest
            # NOTE: maybe change values of all nodes for features not inlcuded, not just x_v
            X = deepcopy(self.data.x)

            # Set discarded features to an average value when few neighbours
            if discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            for val in ex_feat:
                X[self.neighbours, val] = av_feat_values[val].repeat(D)  # 0

            # Special case - consider only nei. influence if too few feat included
            if self.F - len(ex_feat) < min(self.M - self.F - len(ex_nei), args_K):
                # Look at the 2-hop neighbours included
                # Make sure that they are connected to v (with current nodes sampled nodes)
                included_nei = set(
                    self.neighbours.detach().numpy()).difference(ex_nei)
                included_nei = included_nei.difference(
                    one_hop_neighbours.detach().numpy())
                #if len(self.neighbours) - len(ex_nei) < args_K:
                for incl_nei in included_nei:
                    l = nx.shortest_path(G, source=node_index, target=incl_nei)
                    if set(l[1:-1]).isdisjoint(ex_nei):
                        pass
                    else:
                        for n in range(1, len(l)-1):
                            A = torch.cat((A, torch.tensor(
                                [[l[n-1]], [l[n]]])), dim=-1)
                            X[l[n], :] = av_feat_values
                
            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model(x=X.cuda(), edge_index=A.cuda()).exp()[node_index]
            else:
                with torch.no_grad():
                    proba = self.model(x=X, edge_index=A).exp()[node_index]

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else:
                fz[key] = proba[true_pred]

        return fz

    def basic_default(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z from z' and compute prediction f(z) for each sample z'
            In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

        Args:
                Variables are defined exactly as defined in explainer function

        Returns:
                (tensor): f(z) - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses.
        """
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x.size(1))
        else:
            av_feat_values = self.data.x.mean(dim=0)

        # or random feature vector made of random value across each col of X
        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z'
        for i in tqdm(range(num_samples)):

            # Define new node features dataset (we only modify x_v for now)
            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Define new neighbourhood
            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z) for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Create new matrix A and X - for each sample ≈ reform z from z'
        for (key, ex_nei), (_, ex_feat) in zip(excluded_nei.items(), excluded_feat.items()):

            positions = []
            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            for val in ex_nei:
                pos = (self.data.edge_index == val).nonzero()[:, 1].tolist()
                positions += pos
            # Create new adjacency matrix for that sample
            positions = list(set(positions))
            A = np.array(self.data.edge_index)
            A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)

            # Change feature vector for node of interest and the whole subgraph
            # NOTE: maybe change values of all nodes for features not inlcuded, not just x_v
            X = deepcopy(self.data.x)
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            if discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]
            
            for val in ex_feat:
                X[self.neighbours, val] = av_feat_values[val].repeat(D)  # 0

            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model(x=X.cuda(), edge_index=A.cuda()).exp()[node_index]
            else:
                with torch.no_grad():
                    proba = self.model(x=X, edge_index=A).exp()[node_index]

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else: 
                fz[key] = proba[true_pred]

        return fz

    def node_specific(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z from z' and compute prediction f(z) for each sample z'
            In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

        Args: 
                Variables are defined exactly as defined in explainer function 

        Returns: 
                (tensor): f(z) - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses. 
        """
        G = torch_geometric.utils.to_networkx(self.data)

        # We need to recover z from z' - wrt sampled neighbours and node features
        # Initialise new node feature vectors and neighbours to disregard
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x.size(1))
        else:
            av_feat_values = self.data.x.mean(dim=0)
        # or random feature vector made of random value across each col of X
        
        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z'
        for i in range(num_samples):

            # Define new node features dataset (we only modify x_v for now)
            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Define new neighbourhood
            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z) for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Create new matrix A and X - for each sample ≈ reform z from z'
        for (key, ex_nei), (_, ex_feat)  in tqdm(zip(excluded_nei.items(), excluded_feat.items())):

            positions = []
            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            for val in ex_nei:
                pos = (self.data.edge_index == val).nonzero()[:, 1].tolist()
                positions += pos
            # Create new adjacency matrix for that sample
            positions = list(set(positions))
            A = np.array(self.data.edge_index)
            A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)

            # Change feature vector for node of interest
            # NOTE: maybe change values of all nodes for features not inlcuded, not just x_v
            X = deepcopy(self.data.x)
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            if discarded_feat_idx!=[] and len(self.neighbours) - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

            # Special case - consider only nei. influence if too few feat included
            if self.F - len(ex_feat) < min(self.M - self.F - len(ex_nei), args_K):	
                # Look at the 2-hop neighbours included
                # Make sure that they are connected to v (with current nodes sampled nodes)
                included_nei = set(
                    self.neighbours.detach().numpy()).difference(ex_nei)
                included_nei = included_nei.difference(
                    one_hop_neighbours.detach().numpy())
                #if len(self.neighbours) - len(ex_nei) < args_K:
                for incl_nei in included_nei:
                    l = nx.shortest_path(G, source=node_index, target=incl_nei)
                    if set(l[1:-1]).isdisjoint(ex_nei):
                        pass
                    else:
                        for n in range(1, len(l)-1):
                            A = torch.cat((A, torch.tensor(
                                [[l[n-1]], [l[n]]])), dim=-1)
                            X[l[n], :] = av_feat_values

            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model(x=X.cuda(), edge_index=A.cuda()).exp()[node_index]
            else:
                with torch.no_grad():
                    proba = self.model(x=X, edge_index=A).exp()[node_index]

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else:
                fz[key] = proba[true_pred]

        return fz

    def neutral(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z from z' and compute prediction f(z) for each sample z'
            In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

        Args:
                Variables are defined exactly as defined in explainer function

        Returns:
                (tensor): f(z) - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses.
        """
        # Initialise new node feature vectors and neighbours to disregard
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x.size(1))
        else: 
            av_feat_values = self.data.x.mean(dim=0)
        
        # or random feature vector made of random value across each col of X 

        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z'
        for i in tqdm(range(num_samples)):

            # Define new node features dataset (we only modify x_v for now)
            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Define new neighbourhood
            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z) for graphshap dataset - consider all classes
        if multiclass: 
            fz = torch.zeros((num_samples, self.data.num_classes))
        else: 
            fz = torch.zeros(num_samples)

        # Create new matrix A and X - for each sample ≈ reform z from z'
        for (key, ex_nei), (_, ex_feat)  in zip(excluded_nei.items(), excluded_feat.items()):

            # Change feature vector for node of interest
            X = deepcopy(self.data.x)

            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            A = self.data.edge_index
            X[ex_nei,:]=av_feat_values.repeat(len(ex_nei),1)
            # Only for node index
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            if discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]
                
            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model(x=X.cuda(), edge_index=A.cuda()).exp()[node_index]
            else: 
                with torch.no_grad():
                    proba = self.model(x=X, edge_index=A).exp()[node_index]

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else: 
                fz[key] = proba[true_pred]

        return fz
    
        ################################
    
    def compute_pred_regu(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z from z' and compute prediction f(z) for each sample z'
            In fact, we build the dataset (z', f(z)), required to train the weighted linear model.

        Args: 
                Variables are defined exactly as defined in explainer function 

        Returns: 
                (tensor): f(z) - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses. 
        """
        # To networkx
        G = torch_geometric.utils.to_networkx(self.data)

        # We need to recover z from z' - wrt sampled neighbours and node features
        # Initialise new node feature vectors and neighbours to disregard
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x.size(1))
        else:
            av_feat_values = self.data.x.mean(dim=0)

        # Init label f(z) for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        ### Look only at nodes
        if self.M == self.F: 
            excluded_feat = {}

            for i in range(num_samples):
                feats_id = []
                for j in range(self.F):
                    if z_[i, j].item() == 0:
                        feats_id.append(feat_idx[j].item())
                excluded_feat[i] = feats_id

            for key, ex_feat in tqdm(excluded_feat.items()):
                # Change feature vector for node of interest - excluded and discarded features
                X = deepcopy(self.data.x) 
                X[node_index, ex_feat] = av_feat_values[ex_feat]
                for val in ex_feat:
                    X[self.neighbours, val] = av_feat_values[val].repeat(D)
                
                # Apply model on (X,A) as input.
                if self.gpu:
                    with torch.no_grad():
                        proba = self.model(x=X.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                            node_index]
                else:
                    with torch.no_grad():
                        proba = self.model(x=X, edge_index=self.data.edge_index).exp()[node_index]

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else: 
                fz[key] = proba[true_pred]
            
        ### Look only at neighbours
        elif self.M == len(self.neighbours):
            excluded_nei = {}

            for i in range(num_samples):
                nodes_id = []
                for j in range(D):
                    if z_[i, j] == 0:
                        nodes_id.append(self.neighbours[j].item())
                # Dico with key = num_sample id, value = excluded neighbour index
                excluded_nei[i] = nodes_id

            for key, ex_nei in tqdm(excluded_nei.items()):
                positions = []
                for val in ex_nei:
                    pos = (self.data.edge_index == val).nonzero()[:, 1].tolist()
                    positions += pos
                # Create new adjacency matrix for that sample
                positions = list(set(positions))
                A = np.array(self.data.edge_index)
                A = np.delete(A, positions, axis=1)
                A = torch.tensor(A)
                X = deepcopy(self.data.x)

                # Look at the 2-hop neighbours included
                # Make sure that they are connected to v (with current nodes sampled nodes)
                included_nei = set(
                    self.neighbours.detach().numpy()).difference(ex_nei)
                included_nei = included_nei.difference(
                    one_hop_neighbours.detach().numpy())
                #if len(self.neighbours) - len(ex_nei) < args_K:
                for incl_nei in included_nei:
                    l = nx.shortest_path(G, source=node_index, target=incl_nei)
                    if set(l[1:-1]).isdisjoint(ex_nei):
                        pass
                    else: 
                        for n in range(1, len(l)-1):
                            A = torch.cat((A, torch.tensor(
                                [[l[n-1]], [l[n]]])), dim=-1)
                            X[l[n], :] = av_feat_values

                # Apply model on (X,A) as input.
                if self.gpu:
                    with torch.no_grad():
                        proba = self.model(x=X.cuda(), edge_index=A.cuda()).exp()[
                            node_index]
                else:
                    with torch.no_grad():
                        proba = self.model(x=X, edge_index=A).exp()[node_index]

                # Store predicted class label in fz
                if multiclass:
                    fz[key] = proba
                else: 
                    fz[key] = proba[true_pred]
        
        else: 
            fz = self.compute_pred(node_index, num_samples, D, z_, feat_idx,
                         one_hop_neighbours, args_K, args_feat, discarded_feat_idx)
    
        return fz


    ################################
    # LEARN MODEL G
    ################################
    def WLR(self, z_, weights, fz, multiclass, info):
        """Train a weighted linear regression

        Args:
            z_ (torch.tensor): data
            weights (torch.tensor): weights of each sample
            fz (torch.tensor): y data 
        """
        # Define model 
        if multiclass: 
            our_model = LinearRegressionModel(z_.shape[1], self.data.num_classes)
        else:
            our_model = LinearRegressionModel(z_.shape[1], 1)

        # Define optimizer and loss function
        def weighted_mse_loss(input, target, weight):
            return (weight * (input - target) ** 2).mean()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(our_model.parameters(), lr=0.4)

        # Dataloader 
        train = torch.utils.data.TensorDataset(z_, fz)
        train_loader = torch.utils.data.DataLoader(train, batch_size=1)
        
        # Repeat for several epochs
        for epoch in range(50):

            av_loss = []
            #for x,y,w in zip(z_,fz, weights):
            for batch_idx, (dat, target) in enumerate(train_loader):
                x, y = Variable(dat), Variable(target)
            
                # Forward pass: Compute predicted y by passing x to the model 
                pred_y = our_model(x)

                # Compute loss
                loss = weighted_mse_loss(pred_y, y, weights[batch_idx])
                # loss = criterion(pred_y,y)
                
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Store batch loss
                av_loss.append(loss.item())
            if info:
                print('av loss epoch: ', np.mean(av_loss))

        # Evaluate model
        our_model.eval()
        with torch.no_grad():
            pred = our_model(z_)  
        if info:
            print('weighted r2 score: ', r2_score(pred, fz, multioutput='variance_weighted'))
            if multiclass: 
                print(r2_score(pred, fz, multioutput='raw_values'))
            print('r2 score: ', r2_score(pred, fz, weights))

        phi, base_value = [param.T for _,param in our_model.named_parameters()]
        phi = np.squeeze(phi, axis=1)
        return phi.detach().numpy().astype('float64'), base_value
    
    def WLR_sklearn(self, z_, weights, fz, multiclass, info):
        """Train a weighted linear regression

        Args:
            z_ (torch.tensor): data
            weights (torch.tensor): weights of each sample
            fz (torch.tensor): y data 
        """
        # Convert to numpy
        weights = weights.detach().numpy()
        z_ = z_.detach().numpy()
        fz = fz.detach().numpy()
        # Fit weighted linear regression
        reg = LinearRegression()
        reg.fit(z_, fz, weights)
        y_pred = reg.predict(z_)
        # Assess perf
        if info:
            print('weighted r2: ', reg.score(z_, fz, sample_weight=weights))
            print('r2: ', r2_score(fz, y_pred))
        # Coefficients
        phi = reg.coef_
        base_value = reg.intercept_
    
        return phi, base_value

    def WLS(self, z_, weights, fz, multiclass, info):
        """ Ordinary Least Squares Method, weighted
            Estimates shapely value coefficients

        Args:
            z_ (tensor): binary vector representing the new instance
            weights ([type]): shapley kernel weights for z'
            fz ([type]): prediction f(z) where z is a new instance - formed from z' and x

        Returns:
            [tensor]: estimated coefficients of our weighted linear regression - on (z', f(z))
            Dimension (M * num_classes)
        """
        # Add constant term 
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)

        # WLS to estimate parameters 
        try:
            tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
        except np.linalg.LinAlgError:  # matrix not invertible
            if info: 
                print('WLS: Matrix not invertible')
            tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
            tmp = np.linalg.inv(tmp + np.diag(0.01 * np.random.randn(tmp.shape[1])))
        phi = np.dot(tmp, np.dot(
                        np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))

        # Test accuracy 
        y_pred=z_.detach().numpy() @ phi
        if info:
            print('r2: ', r2_score(fz, y_pred))
            print('weighted r2: ', r2_score(fz, y_pred, weights))

        return phi[:-1], phi[-1]

        ################################
    
    ################################
    # INFO ON EXPLANATIONS
    ################################
    def print_info(self, D, node_index, phi, feat_idx, true_pred, true_conf, multiclass):
        """
        Displays some information about explanations - for a better comprehension and audit
        """

        # Print some information
        print('Explanations include {} node features and {} neighbours for this node\
        for {} classes'.format(self.F, D, self.data.num_classes))

        # Compare with true prediction of the model - see what class should truly be explained
        print('Prediction of orignal model is class {} with confidence {}, while label is {}'
              .format(true_pred, true_conf, self.data.y[node_index]))

        # Isolate explanations for predicted class - explain model choices
        if multiclass: 
            pred_explanation = phi[true_pred, :]
        else: 
            pred_explanation = phi
        
        # print('Explanation for the class predicted by the model:', pred_explanation)

        # Look at repartition of weights among neighbours and node features
        # Motivation for regularisation		
        print('Weights for node features: ', sum(pred_explanation[:self.F]),
         'and neighbours: ', sum(pred_explanation[self.F:]))
        print('Total Weights (abs val) for node features: ', sum(np.abs(pred_explanation[:self.F])),
         'and neighbours: ', sum(np.abs(pred_explanation[self.F:])))

        # Note we focus on explanation for class predicted by the model here, so there is a bias towards
        # positive weights in our explanations (proba is close to 1 everytime).
        # Alternative is to view a class at random or the second best class

        # Select most influential neighbours and/or features (+ or -)
        if self.F + D > 10:
            _, idxs = torch.topk(torch.from_numpy(np.abs(pred_explanation)), 6)
            vals = [pred_explanation[idx] for idx in idxs]
            influential_feat = {}
            influential_nei = {}
            for idx, val in zip(idxs, vals):
                if idx.item() < self.F:
                    influential_feat[feat_idx[idx]] = val
                else:
                    influential_nei[self.neighbours[idx-self.F]] = val
            print( 'Most influential features: ', len([(item[0].item(), item[1].item()) for item in list(influential_feat.items())]),
                  'and neighbours', len([(item[0].item(), item[1].item()) for item in list(influential_nei.items())]) )

        # Most influential features splitted bewteen neighbours and features
        if self.F > 5:
            _, idxs = torch.topk(torch.from_numpy(
                np.abs(pred_explanation[:self.F])), 3)
            vals = [pred_explanation[idx] for idx in idxs]
            influential_feat = {}
            for idx, val in zip(idxs, vals):
                influential_feat[feat_idx[idx]] = val
            print('Most influential features: ', [
                  (item[0].item(), item[1].item()) for item in list(influential_feat.items())])

        # Most influential features splitted bewteen neighbours and features
        if D > 5 and self.M!=self.F:
            _, idxs = torch.topk(torch.from_numpy(
                np.abs(pred_explanation[self.F:])), 3)
            vals = [pred_explanation[self.F + idx] for idx in idxs]
            influential_nei = {}
            for idx, val in zip(idxs, vals):
                influential_nei[self.neighbours[idx]] = val
            print('Most influential neighbours: ', [
                  (item[0].item(), item[1].item()) for item in list(influential_nei.items())])

    def vizu(self, edge_mask, node_index, phi, predicted_class, hops, multiclass):
        """ Vizu of important nodes in subgraph around node_index

        Args:
            edge_mask ([type]): vector of size data.edge_index with False 
                                            if edge is not included in subgraph around node_index
            node_index ([type]): node of interest index
            phi ([type]): explanations for node of interest
            predicted_class ([type]): class predicted by model for node of interest 
            hops ([type]):  number of hops considered for subgraph around node of interest 
            multiclass: if we look at explanations for all classes or only for the predicted one
        """
        if multiclass: 
            phi = torch.tensor(phi[predicted_class,:])
        else:
            phi = torch.from_numpy(phi).float()

        # Replace False by 0, True by 1 in edge_mask
        mask = torch.zeros(self.data.edge_index.shape[1])
        for i, val in enumerate(edge_mask):
            if val.item() == True:
                mask[i] = 1

        # Identify one-hop neighbour
        one_hop_nei, _, _, _ = k_hop_subgraph(
                        node_index, 1, self.data.edge_index, relabel_nodes=True,
                        num_nodes=None)

        # Attribute phi to edges in subgraph bsed on the incident node phi value
        for i, nei in enumerate(self.neighbours):
            list_indexes = (self.data.edge_index[0, :] == nei).nonzero()
            for idx in list_indexes:
                # Remove importance of 1-hop neighbours to 2-hop nei.
                if nei in one_hop_nei:
                    if self.data.edge_index[1, idx] in one_hop_nei:
                        mask[idx] = phi[self.F + i]
                    else:
                        pass
                elif mask[idx] == 1:
                    mask[idx] = phi[self.F + i]
            #mask[mask.nonzero()[i].item()]=phi[i, predicted_class]

        # Set to 0 importance of edges related to 0
        mask[mask == 1] = 0

        # Increase coef for visibility and consider absolute contribution
        mask = torch.abs(mask)

        # Vizu nodes 
        ax, G = visualize_subgraph(self.model,
                                   node_index,
                                   self.data.edge_index,
                                   mask,
                                   hops,
                                   y=self.data.y,
                                   threshold=None)
        
        plt.savefig('results/GS1_{}_{}_{}'.format(self.data.name,
                                           self.model.__class__.__name__,
                                           node_index),
                    bbox_inches='tight')

        # Other visualisation 
        G = denoise_graph(self.data, mask, phi[self.F:], self.neighbours, node_index, feat=None, label=self.data.y, threshold_num=10)
        
        log_graph(G,
                    identify_self=True,
                    nodecolor="label",
                    epoch=0,
                    fig_size=(4, 3),
                    dpi=300,
                    label_node_feat=False,
                    edge_vmax=None,
                    args=None)

        plt.savefig('results/GS_{}_{}_{}'.format(self.data.name,
                                                 self.model.__class__.__name__,
                                                  node_index),
                                                  bbox_inches='tight')
        

        #plt.show()




class Greedy:

    def __init__(self, data, model, gpu=False):
        self.model = model
        self.data = data
        self.neighbours = None
        self.gpu = gpu
        self.M = self.data.x.size(1)
        self.F = self.M

        self.model.eval()


    def explain(self, node_index=0, hops=2, num_samples=0, info=False, multiclass=False, *unused):
        """
        Greedy explainer - only considers node features for explanations
        Computes the prediction proba with and without the targeted feature (repeat for all feat)
        This feature's importance is set as the normalised absolute difference in predictions above
        :param num_samples, info: useless here (simply to match GraphSHAP structure)
        """
        # Create a variable to store node features
        x = self.data.x[node_index, :]

        # Store indexes of these non zero feature values
        feat_idx = torch.arange(self.F)

        # Compute predictions
        if self.gpu: 
            device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            with torch.no_grad():
                probas = self.model(x=self.data.x.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                    node_index]
        else: 
            with torch.no_grad():
                probas = self.model(x=self.data.x, edge_index=self.data.edge_index).exp()[
                    node_index]
        probas, label = torch.topk(probas, k=1)

        # Init explanations vector
        
        if multiclass: 
            coefs = np.zeros([self.M, self.data.num_classes])  # (m, #feats)
            
            # Loop on all features - consider all classes
            for i, idx in enumerate(feat_idx):
                idx = idx.item()
                x_ = deepcopy(self.data.x)
                x_[:, idx] = 0.0  # set feat of interest to 0
                if self.gpu: 
                    with torch.no_grad():
                        probas_ = self.model(x=x_.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                            node_index]  # [label].item()
                else:
                    with torch.no_grad():
                        probas_ = self.model(x=x_, edge_index=self.data.edge_index).exp()[
                            node_index]  # [label].item()
                # Compute explanations with the following formula
                coefs[i] = (torch.abs(probas-probas_)/probas).detach().numpy()
        else: 
            probas = probas[label.item()]
            coefs = np.zeros([self.M])  # (m, #feats)
            # Loop on all features - consider all classes
            for i, idx in enumerate(feat_idx):
                idx = idx.item()
                x_ = deepcopy(self.data.x)
                x_[:, idx] = 0.0  # set feat of interest to 0
                if self.gpu: 
                    with torch.no_grad():
                        probas_ = self.model(x=x_.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                            node_index, label.item()]  # [label].item()
                else:
                    with torch.no_grad():
                        probas_ = self.model(x=x_, edge_index=self.data.edge_index).exp()[
                            node_index, label.item()]  # [label].item()
                # Compute explanations with the following formula
                coefs[i] = (torch.abs(probas-probas_)/probas).cpu().detach().numpy()

        return coefs 

    def explain_nei(self, node_index=0, hops=2, num_samples=0, info=False, multiclass=False):
        
        # Create a variable to store node features
        x = self.data.x[node_index, :]

        # Construct k hop subgraph of node of interest (denoted v)
        neighbours, _, _, edge_mask =\
            torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
                                                 num_hops=hops,
                                                 edge_index=self.data.edge_index)
        # Store the indexes of the neighbours of v (+ index of v itself)

        # Remove node v index from neighbours and store their number in D
        neighbours = neighbours[neighbours != node_index]
        self.neighbours = neighbours
        self.M = neighbours.shape[0]

        # Compute predictions
        if self.gpu: 
            device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            with torch.no_grad():
                probas = self.model(x=self.data.x.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                    node_index]
        else: 
            with torch.no_grad():
                probas = self.model(x=self.data.x, edge_index=self.data.edge_index).exp()[
                    node_index]
        pred_confidence, label = torch.topk(probas, k=1)

        if multiclass:
            # Init explanations vector
            coefs = np.zeros([self.M, self.data.num_classes])  # (m, #feats)

            # Loop on all neighbours - consider all classes
            for i, nei_idx in enumerate(self.neighbours):
                nei_idx = nei_idx.item()
                A_ = deepcopy(self.data.edge_index)

                # Find all edges incident to the isolated neighbour
                pos = (self.data.edge_index == nei_idx).nonzero()[:, 1].tolist()

                # Create new adjacency matrix where this neighbour is isolated
                A_ = np.array(self.data.edge_index)
                A_ = np.delete(A_, pos, axis=1)
                A_ = torch.tensor(A_)

                # Compute new prediction with updated adj matrix (without this neighbour)
                if self.gpu: 
                    with torch.no_grad():
                        probas_ = self.model(x=self.data.x.cuda(), edge_index=A_.cuda()).exp()[
                            node_index]
                else: 
                    with torch.no_grad():
                        probas_ = self.model(x=self.data.x, edge_index=A_).exp()[
                            node_index]
                # Compute explanations with the following formula
                coefs[i] = (torch.abs(probas-probas_)/probas).cpu().detach().numpy()

        else: 
            probas = probas[label.item()]
            coefs = np.zeros(self.M)
            for i, nei_idx in enumerate(self.neighbours):
                nei_idx = nei_idx.item()
                A_ = deepcopy(self.data.edge_index)

                # Find all edges incident to the isolated neighbour
                pos = (self.data.edge_index == nei_idx).nonzero()[:, 1].tolist()

                # Create new adjacency matrix where this neighbour is isolated
                A_ = np.array(self.data.edge_index)
                A_ = np.delete(A_, pos, axis=1)
                A_ = torch.tensor(A_)

                # Compute new prediction with updated adj matrix (without this neighbour)
                if self.gpu:
                    with torch.no_grad():
                        probas_ = self.model(x=self.data.x.cuda(), edge_index=A_.cuda()).exp()[
                            node_index, label.item()]
                else:
                    with torch.no_grad():
                        probas_ = self.model(x=self.data.x, edge_index=A_).exp()[
                            node_index, label.item()]
                # Compute explanations with the following formula
                coefs[i] = (torch.abs(probas-probas_)/probas).cpu().detach().numpy()

        return coefs


class Random:

    def __init__(self, num_feats, K):
        self.num_feats = num_feats
        self.K = K 

    def explain(self):
        return np.random.choice(self.num_feats, self.K)


class GraphLIME:

    def __init__(self, data, model, gpu=False, hop=2, rho=0.1, cached=True):
        self.data = data
        self.model = model
        self.hop = hop
        self.rho = rho
        self.cached = cached
        self.cached_result = None
        self.M = data.x.size(1)
        self.F = data.x.size(1)
        self.gpu = gpu

        self.model.eval()

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, y, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.hop, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        y = y[subset]

        for key, item in kwargs:
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, y, edge_index, mapping, edge_mask, kwargs

    def __init_predict__(self, x, edge_index, **kwargs):
        if self.cached and self.cached_result is not None:
            if x.size(0) != self.cached_result.size(0):
                raise RuntimeError(
                    'Cached {} number of nodes, but found {}.'.format(
                        x.size(0), self.cached_result.size(0)))

        if not self.cached or self.cached_result is None:
            # Get the initial prediction.
            with torch.no_grad():
                if self.gpu: 
                    device = torch.device(
                            'cuda' if torch.cuda.is_available() else 'cpu')
                    self.model = self.model.to(device)
                    log_logits = self.model(x=x.cuda(), edge_index=edge_index.cuda(), **kwargs)
                else: 
                    log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
                probas = log_logits.exp()

            self.cached_result = probas

        return self.cached_result

    def __compute_kernel__(self, x, reduce):
        assert x.ndim == 2, x.shape

        n, d = x.shape

        dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
        dist = dist ** 2

        if reduce:
            dist = np.sum(dist, axis=-1, keepdims=True)  # (n, n, 1)

        std = np.sqrt(d)

        # (n, n, 1) or (n, n, d)
        K = np.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))

        return K

    def __compute_gram_matrix__(self, x):
        # unstable implementation due to matrix product (HxH)
        # n = x.shape[0]
        # H = np.eye(n, dtype=np.float) - 1.0 / n * np.ones(n, dtype=np.float)
        # G = np.dot(np.dot(H, x), H)

        # more stable and accurate implementation
        G = x - np.mean(x, axis=0, keepdims=True)
        G = G - np.mean(G, axis=1, keepdims=True)

        G = G / (np.linalg.norm(G, ord='fro', axis=(0, 1), keepdims=True) + 1e-10)

        return G

    def explain(self, node_index, hops, num_samples, info=False, multiclass=False, *unused, **kwargs):
        # hops, num_samples, info are useless: just to copy graphshap pipeline
        x = self.data.x
        edge_index = self.data.edge_index

        probas = self.__init_predict__(x, edge_index, **kwargs)

        x, probas, _, _, _, _ = self.__subgraph__(
            node_index, x, probas, edge_index, **kwargs)

        x = x.cpu().detach().numpy()  # (n, d)
        y = probas.cpu().detach().numpy()  # (n, classes)

        n, d = x.shape

        if multiclass:
            K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
            L = self.__compute_kernel__(y, reduce=False)  # (n, n, 1)

            K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
            L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

            K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
            L_bar = L_bar.reshape(n ** 2, self.data.num_classes)  # (n ** 2,) 

            solver = LassoLars(self.rho, fit_intercept=False,
                            normalize=False, positive=True)
            solver.fit(K_bar * n, L_bar * n)

            return solver.coef_.T

        else: 
            K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
            L = self.__compute_kernel__(y, reduce=True)  # (n, n, 1)

            K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
            L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

            K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
            L_bar = L_bar.reshape(n ** 2,)  # (n ** 2,) 

            solver = LassoLars(self.rho, fit_intercept=False,
                normalize=False, positive=True)
            solver.fit(K_bar * n, L_bar * n)

            return solver.coef_


class LIME:

    def __init__(self, data, model, gpu=False, cached=True):
        self.data = data
        self.model = model
        self.gpu = gpu
        self.M = data.x.size(1)
        self.F = data.x.size(1)

        self.model.eval()

    def __init_predict__(self, x, edge_index, *unused,**kwargs):
        
        # Get the initial prediction.
        with torch.no_grad():
            if self.gpu: 
                device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
                log_logits = self.model(x=x.cuda(), edge_index=edge_index.cuda(), **kwargs)
            else:
                log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
            probas = log_logits.exp()

        return probas

    def explain(self, node_index, hops, num_samples, info=False, multiclass=False, **kwargs):
        num_samples = num_samples//3
        x = self.data.x
        edge_index = self.data.edge_index

        probas = self.__init_predict__(x, edge_index, **kwargs)
        proba, label = probas[node_index, :].max(dim=0)

        x_ = deepcopy(x)
        original_feats = x[node_index, :]

        if multiclass: 
            sample_x = [original_feats.detach().numpy()]
            #sample_y = [proba.item()]
            sample_y = [probas[node_index, :].detach().numpy()]

            for _ in range(num_samples):
                x_[node_index, :] = original_feats + \
                    torch.randn_like(original_feats)

                with torch.no_grad():
                    if self.gpu:
                        log_logits = self.model(x=x_.cuda(), edge_index=edge_index.cuda(), **kwargs)
                    else: 
                        log_logits = self.model(x=x_, edge_index=edge_index, **kwargs)
                    probas_ = log_logits.exp()

                #proba_ = probas_[node_index, label]
                proba_ = probas_[node_index]

                sample_x.append(x_[node_index, :].detach().numpy())
                # sample_y.append(proba_.item())
                sample_y.append(proba_.detach().numpy())

        else: 
            sample_x = [original_feats.detach().numpy()]
            sample_y = [proba.item()]
            # sample_y = [probas[node_index, :].detach().numpy()]

            for _ in range(num_samples):
                x_[node_index, :] = original_feats + \
                    torch.randn_like(original_feats)

                with torch.no_grad():
                    if self.gpu:
                        log_logits = self.model(x=x_.cuda(), edge_index=edge_index.cuda(), **kwargs)
                    else:
                        log_logits = self.model(x=x_, edge_index=edge_index, **kwargs)
                    probas_ = log_logits.exp()

                proba_ = probas_[node_index, label]
                # proba_ = probas_[node_index]

                sample_x.append(x_[node_index, :].detach().numpy())
                sample_y.append(proba_.item())
                # sample_y.append(proba_.detach().numpy())

        sample_x = np.array(sample_x)
        sample_y = np.array(sample_y)

        solver = Ridge(alpha=0.1)
        solver.fit(sample_x, sample_y)

        return solver.coef_.T


class SHAP():

    def __init__(self, data, model, gpu=False):
        self.model = model
        self.data = data
        self.gpu = gpu
        self.M = data.x.size(1)  # number of nonzero features - for each node index
        self.neighbours = None
        self.F = self.M

        self.model.eval()

    def explain(self, node_index=0, hops=2, num_samples=10, info=True, multiclass=False, *unused):
        """
        :param node_index: index of the node of interest
        :param hops: number k of k-hop neighbours to consider in the subgraph around node_index
        :param num_samples: number of samples we want to form GraphSHAP's new dataset 
        :return: shapley values for features that influence node v's pred
        """

        # Determine z' => features and neighbours whose importance is investigated

        # Consider all features (+ use expectation like below)
        feat_idx = torch.unsqueeze(torch.arange(self.data.x.size(0)), 1)

        # Sample z' - binary vector of dimension (num_samples, M)
        # F node features first, then D neighbours
        z_ = torch.empty(num_samples, self.M).random_(2)
        # Compute |z'| for each sample z'
        s = (z_ != 0).sum(dim=1)

        # Define weights associated with each sample using shapley kernel formula
        weights = self.shapley_kernel(s)

        # Create dataset (z', f(z)), stored as (z_, fz)
        # Retrive z from z' and x_v, then compute f(z)
        fz = self.compute_pred(node_index, num_samples, self.F, z_, feat_idx)

        # Compute true prediction of model, for original instance
        with torch.no_grad():
            if self.gpu: 
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
                true_conf, true_pred = self.model(
                    x=self.data.x.cuda(), edge_index=self.data.edge_index.cuda()).exp()[node_index].max(dim=0)
            else:
                true_conf, true_pred = self.model(
                    x=self.data.x, edge_index=self.data.edge_index).exp()[node_index].max(dim=0)

        # Multiclass 
        if not multiclass: 
            fz = fz[:, true_pred]

        # OLS estimator for weighted linear regression
        phi, base_value = self.OLS(z_, weights, fz)  # dim (M*num_classes)

        # Visualisation
        # Call visu function
        # Pass it true_pred

        return phi

    def shapley_kernel(self, s):
        """
        :param s: dimension of z' (number of features + neighbours included)
        :return: [scalar] value of shapley value 
        """
        shap_kernel = []
        # Loop around elements of s in order to specify a special case
        # Otherwise could have procedeed with tensor s direclty
        for i in range(s.shape[0]):
            a = s[i].item()
            # Put an emphasis on samples where all or none features are included
            if a == 0 or a == self.M:
                shap_kernel.append(1000)
            elif scipy.special.binom(self.M, a) == float('+inf'):
                shap_kernel.append(1)
            else:
                shap_kernel.append(
                    (self.M-1)/(scipy.special.binom(self.M, a)*a*(self.M-a)))
        return torch.tensor(shap_kernel)

    def compute_pred(self, node_index, num_samples, F, z_, feat_idx):
        """
        Variables are exactly as defined in explainer function, where compute_pred is used
        This function aims to construct z (from z' and x_v) and then to compute f(z), 
        meaning the prediction of the new instances with our original model. 
        In fact, it builds the dataset (z', f(z)), required to train the weighted linear model.
        :return fz: probability of belonging to each target classes, for all samples z
        fz is of dimension N*C where N is num_samples and C num_classses. 
        """
        # This implies retrieving z from z' - wrt sampled neighbours and node features
        # We start this process here by storing new node features for v and neigbours to
        # isolate
        X_v = torch.zeros([num_samples, self.data.num_features])

        # Feature matrix
        A = np.array(self.data.edge_index)
        A = torch.tensor(A)

        # Init label f(z) for graphshap dataset - consider all classes
        fz = torch.zeros((num_samples, self.data.num_classes))
        # Init final predicted class for each sample (informative)
        classes_labels = torch.zeros(num_samples)
        pred_confidence = torch.zeros(num_samples)

        # Do it for each sample
        for i in range(num_samples):

            # Define new node features dataset (we only modify x_v for now)
            # Features where z_j == 1 are kept, others are set to 0
            for j in range(F):
                if z_[i, j].item() == 1:
                    X_v[i, feat_idx[j].item()] = 1

            # Change feature vector for node of interest
            # NOTE: maybe change values of all nodes for features not inlcuded, not just x_v
            X = deepcopy(self.data.x)
            X[node_index, :] = X_v[i, :]

            # Apply model on (X,A) as input.
            with torch.no_grad():
                if self.gpu:
                    proba = self.model(x=X.cuda(), edge_index=A.cuda()).exp()[node_index]
                else:
                    proba = self.model(x=X, edge_index=A).exp()[node_index]
                    

            # Store final class prediction and confience level
            # pred_confidence[i], classes_labels[i] = torch.topk(proba, k=1) # optional
            # NOTE: maybe only consider predicted class for explanations

            # Store predicted class label in fz
            fz[i] = proba

        return fz

    def OLS(self, z_, weights, fz):
        """
        :param z_: z' - binary vector  
        :param weights: shapley kernel weights for z'
        :param fz: f(z) where z is a new instance - formed from z' and x
        :return: estimated coefficients of our weighted linear regression - on (z', f(z))
        phi is of dimension (M * num_classes)
        """
        # Add constant term 
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)

        # WLS to estimate parameters 
        try:
            tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
        except np.linalg.LinAlgError:  # matrix not invertible
            tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
            tmp = np.linalg.inv(tmp + np.diag(0.01 * np.random.randn(tmp.shape[1])))
        phi = np.dot(tmp, np.dot(
                        np.dot(z_.T, np.diag(weights)), fz.cpu().detach().numpy()))

        # Test accuracy 
        # y_pred=z_.detach().numpy() @ phi
        #	print('r2: ', r2_score(fz, y_pred))
        #	print('weighted r2: ', r2_score(fz, y_pred, weights))

        return phi[:-1], phi[-1]

class GNNExplainer():

    def __init__(self, data, model, gpu=False):
        self.data = data
        self.model = model
        self.M = data.x.size(0) + data.x.size(1)
        self.gpu = gpu
        # self.coefs = torch.zeros(data.x.size(0), self.data.num_classes)
        self.coefs = None  # node importance derived from edge importance
        self.edge_mask = None
        self.neighbours = None
        self.F = data.x.size(1)

        self.model.eval()

    def explain(self, node_index, hops, num_samples, info=False, multiclass=False, *unused):
        num_samples = num_samples//3
        # Use GNNE open source implem - outputs features's and edges importance
        if self.gpu: 
            device = torch.device('cpu')
            self.model = self.model.to(device)
        explainer = GNNE(self.model, epochs=num_samples)
        node_feat_mask, self.edge_mask = explainer.explain_node(
            node_index, self.data.x, self.data.edge_index)
            
        # Transfer edge importance to node importance
        dico = {}
        for idx in torch.nonzero(self.edge_mask):
            node = self.data.edge_index[0, idx].item()
            if not node in dico.keys():
                dico[node] = [self.edge_mask[idx]]
            else:
                dico[node].append(self.edge_mask[idx])
        # Count neighbours in the subgraph
        self.neighbours = torch.tensor([index for index in dico.keys()])
            
        if multiclass:	
            # Attribute an importance measure to each node = sum of incident edges' importance
            self.coefs = torch.zeros(
                self.neighbours.shape[0], self.data.num_classes)
            # for key, val in dico.items():
            for i, val in enumerate(dico.values()):
                #self.coefs[key,:] = sum(val)
                self.coefs[i, :] = sum(val)

            # Eliminate node_index from neighbourhood
            self.neighbours = self.neighbours[self.neighbours != node_index]
            self.coefs = self.coefs[1:]

            if info == True:
                self.vizu(self.edge_mask, node_index, self.coefs[0], hops)

            return torch.stack([node_feat_mask]*self.data.num_classes, 1)
        
        else: 
            # Attribute an importance measure to each node = sum of incident edges' importance
            self.coefs = torch.zeros(
                self.neighbours.shape[0])
            for i, val in enumerate(dico.values()):
                self.coefs[i] = sum(val)

            # Eliminate node_index from neighbourhood
            self.neighbours = self.neighbours[self.neighbours != node_index]
            self.coefs = self.coefs[1:]

            if info == True:
                self.vizu(self.edge_mask, node_index, self.coefs[0], hops)
            del explainer

            return node_feat_mask

    def vizu(self, edge_mask, node_index, phi, hops):
        """
        Visualize the importance of neighbours in the subgraph of node_index
        """
        # Replace False by 0, True by 1
        mask = torch.zeros(self.data.edge_index.shape[1])
        for i, val in enumerate(edge_mask):
            if val.item() != 0:
                mask[i] = 1

        # Attribute phi to edges in subgraph bsed on the incident node phi value
        for i, nei in enumerate(self.neighbours):
            list_indexes = (self.data.edge_index[0, :] == nei).nonzero()
            for idx in list_indexes:
                if self.data.edge_index[1, idx] == node_index:
                    mask[idx] = edge_mask[idx]
                    break
                elif mask[idx] != 0:
                    mask[idx] = edge_mask[idx]

        # Set to 0 importance of edges related to 0
        mask[mask == 1] = 0

        # Vizu nodes and
        ax, G = visualize_subgraph(self.model,
                                   node_index,
                                   self.data.edge_index,
                                   mask,
                                   hops,
                                   y=self.data.y,
                                   threshold=None)

        plt.savefig('results/GS1_{}_{}_{}'.format(self.data.name,
                                            self.model.__class__.__name__,
                                            node_index),
                    bbox_inches='tight')
        #plt.show()
