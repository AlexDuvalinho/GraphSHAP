INPUT_DIM = {'Cora': 1433,
			'PubMed': 500,
			'Amazon':745,
			'PPI': 50,
			'Reddit': 602}

DIM_FEAT_P = {'Cora': 0.013,
			'PubMed':0.1}

# Model structure hyperparameters for Cora dataset, GCN model
hparams_Cora_GCN = {
		'hidden_dim': [16],
		'dropout': 0.5
		}

# Training hyperparameters for Cora dataset, GCN model 
params_Cora_GCN = {
	'num_epochs':50,
	'lr':0.01, 
	'wd':5e-4
	}

# Cora - GAT
hparams_Cora_GAT = {
		'hidden_dim': [8],
		'dropout': 0.6,
		'n_heads': [8,1]
		}

params_Cora_GAT = {
	'num_epochs':100,
	'lr':0.005, 
	'wd':5e-4
	}


# PubMed - GCN
hparams_PubMed_GCN = hparams_Cora_GCN 
params_PubMed_GCN = {
	'num_epochs':150,
	'lr':0.01, 
	'wd':5e-4
	}

# PubMed - GAT 
hparams_PubMed_GAT = {
		'hidden_dim': [8],
		'dropout': 0.6,
		'n_heads': [8,8]
		}

params_PubMed_GAT = {
	'num_epochs':120,
	'lr':0.005, 
	'wd':5e-4
	}
# suggested n_heads = [8,8] with more epochs, but not necessary and better in this case


# Amazon - GCN 
hparams_Amazon_GCN = {
		'hidden_dim': [32],
		'dropout': 0.5
		}

params_Amazon_GCN = params_PubMed_GCN

# Amazon - GAT
hparams_Amazon_GAT = hparams_PubMed_GAT 
params_Amazon_GAT = params_PubMed_GAT


# PPI - GCN
hparams_PPI_GCN = {
		'hidden_dim': [32,16],
		'dropout': 0.1
		}

params_PPI_GCN = {
	'num_epochs':20,
	'lr':0.01, 
	'wd':5e-4
	}

# PPI - GAT
# Change loss function as well to BCEWithLogits
hparams_PPI_GAT = {
		'hidden_dim': [256,256],
		'dropout': 0,
		'n_heads': [4,4,6]
		}

params_PPI_GAT = {
	'num_epochs':2000,
	'lr':0.005, 
	'wd':0
	}

