# Use files in src folder
from src.data import prepare_data
from src.models import GCN
from src.train import *
from src.explainer import GraphSHAP

# Load the dataset
data = prepare_data(dataset='Cora')

# Define the model
hparams = {
		'input_dim': data.x.size(1),
		'hidden_dim': 16,
		'output_dim': max(data.y).item() + 1
		}
model = GCN(**hparams)

# Train the model
train_and_val(model, data, num_epochs=40)

# Compute predictions
log_logits = model(x=data.x, edge_index=data.edge_index) # [2708, 7]
probas = log_logits.exp()  # combine in 1 line + change accuracy

# Evaluate the model - test set
test_acc = accuracy(log_logits[data.test_mask], data.y[data.test_mask])
print('Test accuracy is {:.4f}'.format(test_acc))

# Explain it with GraphSHAP
graphshap = GraphSHAP(data, model)
explanations = graphshap.explainer(node_index=10, hops=2, num_samples=100)