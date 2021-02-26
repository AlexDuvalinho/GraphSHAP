""" train.py

	Trains our model 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
from utils.graph_utils import GraphSampler
from torch.autograd import Variable


##################################################################
# Node Classification on real world datasets
##################################################################

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
        
            if verbose:
                tqdm.write(log)

            best = val_loss_values[-1]
        else:
            bad_counter += 1

    print('-------------------------------------------------')



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


##################################################################
# Node Classification on synthetic datasets
##################################################################

def train_syn(data, model, args):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr) #weight_decay=args.weight_decay)
    for epoch in range(args.num_epochs):
        total_loss = 0
        model.train()
       
        opt.zero_grad()
        pred = model(data.x, data.edge_index)

        pred = pred[data.train_mask]
        label = data.y[data.train_mask]

        loss = model.loss(pred, label)
        loss.backward()
        opt.step()
        total_loss += loss.item() * 1
        
        if epoch % 10 == 0:
            test_acc = test_syn(data, model, args, data.y, data.val_mask)
            print("Epoch {}. Loss: {:.4f}. Val accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
    total_loss = total_loss / data.x.shape[0]

def test_syn(data, model, args, labels, test_mask):
    model.eval()

    train_ratio = args.train_ratio
    correct = 0
    
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        pred = pred.argmax(dim=1)

        # node classification: only evaluate on nodes in test set
        pred = pred[data.test_mask]
        label = data.y[data.test_mask]

        correct += pred.eq(label).sum().item()

    total = (data.test_mask == True).nonzero().shape[0]
    return correct / total


##################################################################
# Graph Classification 
##################################################################


def train_gc(data, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    dataset_sampler = GraphSampler(data)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True
        )
    
    for epoch in range(args.num_epochs):
        avg_loss = 0.0
        model.train()
        for batch_idx, df in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            y_pred, _ = model(df["feats"], df["adj"])
            loss = model.loss(y_pred, df['label'])
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            avg_loss += loss

        # Eval
        avg_loss /= batch_idx + 1
        if epoch % 10 == 0:
            test_acc = test_gc(data, model, args, data.test_mask)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, avg_loss, test_acc))


def test_gc(data, model, args, mask):
    model.eval()

    train_ratio = args.train_ratio
    correct = 0

    with torch.no_grad():
        pred, _ = model(data.x, data.edge_index)
        pred = pred.argmax(dim=1)

        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = data.y[mask]

        correct += pred.eq(label).sum().item()

    total = (mask == True).nonzero().shape[0]
    return correct / total


