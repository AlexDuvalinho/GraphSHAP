from src.eval import filter_useless_feature, filter_useless_nodes

# Add argument parser of at least pass all tunable arguments of filter_useless_features
noisy_feat_included = filter_useless_feature(info=True)

noisy_nei_included = filter_useless_nodes(info=True)
