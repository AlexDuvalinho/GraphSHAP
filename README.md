
GraphSVX: Shapley Value Explanations for Graph Neural Networks 

In this folder, we provide the code of GraphSVX, as well as one of the evaluation
pipeline defined in the paper, which we use to show the functioning of GraphSVX. 

If needed, install the required packages using: pip install -r requirements.txt

To run the code and explain the prediction of a node in Cora dataset, use: python3 script_explain.py
All settings are described in the corresponding python file. 

The structure of the code is as follows: 
In src: 
    - explainers.py: defines GraphSVX and main baselines
    - data.py: import and process the data 
    - models.py: define GNN models
    - train.py: train GNN models
    - utils.py: stores useful variables
    - eval.py: one of the evaluation of the paper, with real world datasets
    - eval_multiclass.py: explain all classes predictions
    - plots.py: code for nice renderings

Outside: 
    - script_train.py: train a GCN/GAT model on Cora/PubMed datasets
    - script_explain.py: explain node prediction with GraphSVX
    - script_eval.py: evaluate GraphSVX on noisy dataset and observe number of 
    noisy features included in explanations 
    - results folder: stores visualisation and evaluation results
