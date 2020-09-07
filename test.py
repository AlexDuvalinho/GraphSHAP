


"""
import torch
from src.data import prepare_data
from src.explainer import GraphSHAP
import argparse

data = prepare_data('PPI', 10)

model_path = 'models/{}_model_{}.pth'.format('GCN', 'PPI')
model = torch.load(model_path)
model.eval()

for df in data.graphs:
	graphshap = GraphSHAP(df, model)
	explanations = graphshap.explainer(10, 2, 10)
"""