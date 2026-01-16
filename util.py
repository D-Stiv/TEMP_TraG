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

    config._parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")
    config._parser.add_argument("--reverse_mp", action='store_true', help="Use reverse MP in GNN training")
    config._parser.add_argument("--temporal_mp", default=0, type=int, help="Use temporal MP in GNN training", nargs='*')
    config._parser.add_argument("--ports", action='store_true', help="Use port numberings in GNN training")
    config._parser.add_argument("--ports_batch", action='store_true', help="Use port numberings in GNN training but compute port numbers after neighborhood sampling.")
    config._parser.add_argument("--ego", action='store_true', help="Use ego IDs in GNN training")
    config._parser.add_argument("--edge_agg_type", default=None, type=str, help="Select the aggregation method on parallel edges [genagg, gin, sum]", nargs='*')
    config._parser.add_argument("--node_agg_type", default='sum', type=str, help="Select the aggregation method on nodes in message passing", nargs='*')

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
    config._parser.add_argument("--model_name", default='TeMP-TraG', type=str, help="Name of the entire model")
    config._parser.add_argument("--ts_reduce", default='max', type=str, help="reduce function for timestamp: min, max, sum, mean", nargs='*')
    config._parser.add_argument("--save_model", action='store_true', help="Save the best model.")
    config._parser.add_argument("--treedepth", default=5, type=int, help="Random Forest Classifier (all rules)", nargs='*')
    config._parser.add_argument("--class_weight", default="[1,3]", type=str, help="class weights for the loss function", nargs='*')
    config._parser.add_argument("--tem_feat_class", default="[]", type=str, help="list of temporal categories", nargs='*')

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
    logging.info(f"Random seed set as {seed}")
    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.cudnn_enabled = False
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
 