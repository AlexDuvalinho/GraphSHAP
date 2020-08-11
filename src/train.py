import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# from torch_geometric.data import DataLoader



def train(model, data, num_epochs, verbose=True):
	"""
	:return: trained GNN model on Cora dataset 
	"""
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
	return train_error


def train_and_val(model, data, num_epochs, verbose=True):
	"""
	Similar function as train above except that it combines training of the model
	with its evaluation (epoch by epoch) on validation data
	"""

	# Define the optimizer for the learning process
	optimizer = torch.optim.Adam(model.parameters(), lr= 0.01, weight_decay=5e-4)

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

			# model_path = 'model.pth'	
			#torch.save(model.state_dict(), model_path)
			#log += ' save model to {}'.format(model_path)

			if verbose:
				tqdm.write(log)
			
			best = val_loss_values[-1]
		else:
			bad_counter += 1

	print('-------------------------------------------------')

	#model.load_state_dict(torch.load(model_path))


def train_on_epoch(model, data, optimizer):
	"""
	:return: loss and accuracy of model on training data
	Core of the training process
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
	"""
	:return: loss and accuracy of model on validation data 
	Core of the evaluation phase for model on data
	"""
	model.eval()

	with torch.no_grad():
		output = model(data.x, data.edge_index)
		loss = F.nll_loss(output[mask], data.y[mask])
		acc = accuracy(output[mask], data.y[mask])

	return loss, acc


def accuracy(output, labels):
	"""
	:param output: class predictions for each node, computed with our model
	:param labels: true label of each node
	Compute accuracy metric for the model
	"""
	# Find predicted label from predicted probabilities
	_, pred = output.max(dim=1)
	# Derive number of correct predicted labels
	correct = pred.eq(labels).double()
	# Sum over all nodes
	correct = correct.sum()

	# Return accuracy metric
	return correct / len(labels)
