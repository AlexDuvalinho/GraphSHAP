"""train.py

Trains our model 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm

def train_and_val(model, data, num_epochs, lr, wd, verbose=True):
	""" Model training

	Args:
		model (pyg): model trained, previously defined
		data (torch_geometric.Data): dataset the model is trained on
		num_epochs (int): number of epochs 
		lr (float): learning rate
		wd (float): weight decay
		verbose (bool, optional): print information. Defaults to True.
	
	"""

	# Define the optimizer for the learning process
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

	# Training and eval modes
	train_loss_values, train_acc_values = [], []
	val_loss_values, val_acc_values = [], []

	best = np.inf
	bad_counter = 0

	for epoch in tqdm(range(num_epochs), desc='Training', leave=False):
		if epoch == 0:
			print('       |     Trainging     |     Validation     |')
			print('       |-------------------|--------------------|')
			print(' Epoch |  loss    accuracy |  loss    accuracy  |')
			print('-------|-------------------|--------------------|')
		
		# Training 
		train_loss, train_acc = train_on_epoch(model, data, optimizer)
		train_loss_values.append(train_loss.item())
		train_acc_values.append(train_acc.item())

		# Eval
		val_loss, val_acc = evaluate(model, data, data.val_mask)
		val_loss_values.append(val_loss.item())
		val_acc_values.append(val_acc.item())

		if val_loss_values[-1] < best:
			bad_counter = 0
			log = '  {:3d}  | {:.4f}    {:.4f}  | {:.4f}    {:.4f}   |'.format(epoch + 1,
																			train_loss.item(),
																			train_acc.item(),
																			val_loss.item(),
																			val_acc.item())

			#model_path = 'model.pth'
			#torch.save(model.state_dict(), model_path)

			if verbose:
				tqdm.write(log)
			
			best = val_loss_values[-1]
		else:
			bad_counter += 1

	print('-------------------------------------------------')

	#model.load_state_dict(torch.load(model_path))


def train_on_epoch(model, data, optimizer):
	""" Pytorch core training scheme for one epoch

	Args: 
		optimizer: optimizer used to teach the model. Here Adam. 

	Returns:
		[torch.Tensor]: loss function's value
		[torch.Tensor]: accuracy of model on training data
	"""
	model.train()
	optimizer.zero_grad()
	output = model(data.x, data.edge_index)
	train_loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
	train_acc = accuracy(output[data.train_mask], data.y[data.train_mask])
	train_loss.backward()
	optimizer.step()

	return train_loss, train_acc


def evaluate(model, data, mask):
	""" Model evaluation on validation data

	Args: 
		mask (torch.tensor): validation mask

	Returns:
		[torch.Tensor]: loss function's value on validation data
		[torch.Tensor]: accuracy of model on validation data
	"""
	model.eval()
	with torch.no_grad():
		output = model(data.x, data.edge_index)
		loss = F.nll_loss(output[mask], data.y[mask])
		acc = accuracy(output[mask], data.y[mask])

	return loss, acc


def accuracy(output, labels):
	""" Computes accuracy metric for the model on test set

	Args:
		output (tensor): class predictions for each node, computed with our model
		labels (tensor): true label of each node
	
	Returns:
		[tensor]: accuracy metric
	
	"""
	# Find predicted label from predicted probabilities
	_, pred = output.max(dim=1)
	# Derive number of correct predicted labels
	correct = pred.eq(labels).double()
	# Sum over all nodes
	correct = correct.sum()

	# Return accuracy metric
	return correct / len(labels)


























"""
def train(model, data, num_epochs, verbose=True):
	'''
	:return: training error of trained GNN model on Cora dataset
	'''
	optimizer = torch.optim.Adam(model.parameters(), lr= 0.05)

	# Train mode
	train_error = []
	model.train()

	# For each epoch, compute predictions and backpropagate the loss
	for epoch in range(num_epochs):

		# Compute pred 
		y_pred = model(data.x, data.edge_index)

		# Learning phase via backpropagation
		loss = F.nll_loss(y_pred[data.train_mask], data.y[data.train_mask])
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_error.append(loss)
		train_acc = accuracy(y_pred[data.train_mask], data.y[data.train_mask])
		print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch+1, num_epochs, loss, train_acc))
	# return train_error
"""



"""
### PPI dataset 

# Use train_ppi in train_and_val above
def ppi_overall(model, data): 
	#def train_ppi(model, data, num_epochs, lr, wd, verbose=True):
	loss_op = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
		
	for epoch in range(1, 10):
		loss = train_ppi(model, data)
		val_f1 = test_ppi([df for df in data.graphs if df['split']=='val'])
		test_f1 = test_ppi([df for df in data.graphs if df['split']=='test'])
		print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
			epoch, loss, val_f1, test_f1))


def train_ppi(model, data):
	model.train()

	total_loss = 0
	len_training_data = 0
	for df in data.graphs:
		if df['split']=='train':
			optimizer.zero_grad()
			prediction = model(df.x, df.edge_index)
			loss = loss_op(prediction, df.y) # multilabel 
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			len_training_data += 1 
	return (total_loss / len_training_data)


def test_ppi(loader):
	model.eval()

	ys, preds = [], []
	for df in loader:
		ys.append(df.y)
		with torch.no_grad():
			out = model(df.x, df.edge_index)
		preds.append((out>0).float())

	y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
	return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
"""
