# MIT License
#
# Copyright (c) 2026 D-Stiv
#
# See the LICENSE file in the repository root for full license text.

import argparse
import numpy as np
import torch
import random
import logging
import os
import sys
import copy
    
class Configuration:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        
        # config parsed by the default parser
        self._config = None

        # individual configurations for different runs
        self._configs = []
        
        # arguments with more than one value
        self._multivalue_args = []       
        
    def parse(self):
        self._config = self._parser.parse_args()
    
        # find values with more than one entry
        dict_config = vars(self._config)
        for k in dict_config :
            if isinstance(dict_config[k], list):
                self._multivalue_args.append(k)

        self._configs.append(self._config)
        for ma in self._multivalue_args:
            new_configs = []

            # in each config
            for c in self._configs:
                # split each attribute with multiple values
                for v in dict_config[ma]:
                    connectionrent = copy.deepcopy(c)
                    setattr(connectionrent, ma, v)
                    new_configs.append(connectionrent)

            # store splitted values
            self._configs = new_configs
        
    def get_configs(self):
        return self._configs


def setup_config(config):
    print('Configuration setup ...')

    #Adaptations
    config._parser.add_argument("--task", default='edge_class', type=str, help="['edge_class', 'node_class', 'lp']")
    config._parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")
    config._parser.add_argument("--reverse_mp", action='store_true', help="Use reverse MP in GNN training")
    config._parser.add_argument("--temporal_mp", default=0, type=int, help="Use temporal MP in GNN training", nargs='*')
    config._parser.add_argument("--mp_scale", default=1.0, type=float, help="beta = 1 + mp_scale*alpha", nargs='*')
    config._parser.add_argument("--reverse_mp_lp", action='store_true', help="Use reverse MP in LP GNN training")
    config._parser.add_argument("--ports", action='store_true', help="Use port numberings in GNN training")
    config._parser.add_argument("--ports_batch", action='store_true', help="Use port numberings in GNN training but compute port numbers after neighborhood sampling.")
    config._parser.add_argument("--tds", action='store_true', help="Use time deltas (i.e. the time between subsequent transactions) in GNN training")
    config._parser.add_argument("--ego", action='store_true', help="Use ego IDs in GNN training")
    config._parser.add_argument("--edge_agg_type", default='pna', type=str, help="Select the aggregation method on parallel edges [gin, pna]", nargs='*')
    config._parser.add_argument("--node_agg_type", default='sum', type=str, help="Select the aggregation method on parallel edges [sum]", nargs='*')

    config._parser.add_argument("--batch_size", default=8192, type=int, help="Select the batch size for GNN training", nargs='*')
    config._parser.add_argument("--n_epochs", default=100, type=int, help="Select the number of epochs for GNN training", nargs='*')
    config._parser.add_argument('--num_neighs', default='[100,100]', type=str, help='Pass the number of neighors to be sampled in each hop (descending).', nargs='*')
    config._parser.add_argument("--flatten_edges", action='store_true', help="Flatten parallel edges")

    config._parser.add_argument("--device", default="cuda:0", type=str, help="Select a GPU", required=False)
    config._parser.add_argument("--seed", default=42, type=int, help="Select the random seed for reproducability != 1", nargs='*')
    config._parser.add_argument("--expid", default=-1, type=int, help="experiment id", nargs='*')
    config._parser.add_argument("--patience", default=50, type=int, help="experiment id", nargs='*')
    config._parser.add_argument("--majority_class", default=0, type=int, help="majority class")
    config._parser.add_argument("--minority_class", default=1, type=int, help="minority class")
    config._parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    config._parser.add_argument("--data", default=None, type=str, help="Select the AML dataset. Needs to be either small or medium.", required=True, nargs='*')
    config._parser.add_argument("--model", default=None, type=str, help="Select the model architecture. Needs to be one of [gin, pn]. for [gat, rgcn] we need to extend the code", required=True, nargs='*')
    config._parser.add_argument("--model_name", default='MEGA', type=str, help="Name of the entire model")
    config._parser.add_argument("--ts_reduce", default='max', type=str, help="reduce function for timestamp: min, max, sum, mean", nargs='*')
    config._parser.add_argument("--version", default='base', type=str, help="version", nargs='*')
    config._parser.add_argument("--resultTable", default='mega_gnn_final', type=str, help="name of the database table")
    config._parser.add_argument("--testing", action='store_true', help="Disable wandb logging while running the script in 'testing' mode.")
    config._parser.add_argument("--save_model", action='store_true', help="Save the best model.")
    config._parser.add_argument("--unique_name", action='store_true', help="Unique name under which the model will be stored.")
    config._parser.add_argument("--finetune", action='store_true', help="Fine-tune a model. Note that args.unique_name needs to point to the pre-trained model.")
    config._parser.add_argument("--exp_comment", default=None, type=str, help="")
    config._parser.add_argument("--edge_rules", default=1, type=int, help="Construct edge rules and append to node features", nargs='*')
    config._parser.add_argument("--temp_rules", default=0, type=int, help="Use temporal edge rules", nargs='*')

    config._parser.add_argument("--in_outlier_separation", default=0, type=int, help="0: outlier VS mo outlier. 0: illicit outlier VS licit outlier", nargs='*')
    config._parser.add_argument("--treedepth", default=5, type=int, help="Random Forest Classifier (all rules)", nargs='*')
    config._parser.add_argument("--min_cluster_size", default=3, type=int, help="HDBSCAN (rules refinement)", nargs='*')
    config._parser.add_argument("--min_samples", default=3, type=int, help="HDBSCAN (rules refinement)", nargs='*')
    config._parser.add_argument("--median_threshold", default=1, type=int, help="Use median threshold for features", nargs='*')

    config._parser.add_argument("--N0", default=0.0, type=float, help="Use learnable parameter for N0: 0 to have learnable parameter", nargs='*')
    config._parser.add_argument("--half_life", default=2.0, type=float, help="half life for exponential decay", nargs='*')
    config._parser.add_argument("--base", default=0.0, type=float, help="base for exponential decay: 0 for base e. < 0 to deactivate the exponential decay", nargs='*')
    config._parser.add_argument("--class_weight", default="[1,3]", type=str, help="class weights for the loss function", nargs='*')

    config._parser.add_argument("--tem_feat_class", default="[]", type=str, help="list of temporal categories", nargs='*')
    config._parser.add_argument("--edge_based", default=0, type=int, help="Insert edge features into nodes for training", nargs='*')
    config._parser.add_argument("--rules_in_node", default=0, type=int, help="Insert rules into nodes for IBM dataset", nargs='*')
    config._parser.add_argument("--debug", default=0, type=int, help="Used a reduced amount of transactions", nargs='*')
    
    config._parser.add_argument("--lr", default=1e-3, type=float, help="learning rate", nargs='*')
    config._parser.add_argument("--n_hidden", default=2, type=int, help="number of hidden layers", nargs='*')
    config._parser.add_argument("--n_gnn_layers", default=2, type=int, help="number of GNN layers", nargs='*')
    config._parser.add_argument("--dropout", default=0.3, type=float, help="dropout rate", nargs='*')
    config._parser.add_argument("--final_dropout", default=0.3, type=float, help="final dropout rate", nargs='*')

    config._parser.add_argument("--num_workers", default=4, type=int, help="number of workers for data loading", nargs='*')


    config._parser.add_argument("--trade_data_path", default='path to directory for datasets', type=str, help="trade data path") 
    config._parser.add_argument("--model_path", default='path to directory for models', type=str, help="model path")
    config._parser.add_argument("--prepared_data_path", default='path to directory for prepared data', type=str, help="prepared data path") 

    return config.parse()

def set_seed(seed: int = 0, device='cpu') -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.cudnn_enabled = False
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
 