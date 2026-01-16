# MIT License
#
# Copyright (c) 2026 D-Stiv
#
# See the LICENSE file in the repository root for full license text.


import time
import logging
from util import Configuration, setup_config
from load_data import get_data, get_node_classification_data
from training_edge import train_edge
from training_node import train_node
import json


exp_config = Configuration() 
setup_config(exp_config)

def main(args):
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)


    # Check Argument consistency
    if args.data in ['list of edge class. datasets']:
        args.task = 'edge_class'
    elif args.data in ['list of node class. datasets']:
        args.task = 'node_class'
  
    temp_cat = eval(args.tem_feat_class)
    if 'TAEC' in temp_cat or 'TAC' in temp_cat or 'TRM' in temp_cat or 'TEMP' in temp_cat:
        args.tem_feat_ext = 1
    else:
        args.tem_feat_ext = 0

    t1 = time.perf_counter()


    if args.data == "Small_HI":
        list_tr_data, list_val_data, list_te_data, tr_inds, val_inds, te_inds = get_data(args, config)     
    elif args.data == "Node_Class_Data":
        list_tr_data, list_val_data, list_te_data, tr_inds, val_inds, te_inds = get_node_classification_data(args, config)
     
    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")

    # Training
    logging.info(f"Running Training")
    if args.data == "Small_HI":
        results = train_edge(list_tr_data, list_val_data, list_te_data, tr_inds, val_inds, te_inds, args, config)
    else:
        results = train_node(list_tr_data, list_val_data, list_te_data, tr_inds, val_inds, te_inds, args, config)

if __name__ == "__main__":
    exp_num = 1
    tot_exp = len(exp_config.get_configs())
    print('Number of experiments: ', tot_exp)

    for args in exp_config.get_configs():   
        print(f'Starting experiment number {exp_num}/{tot_exp} ...')  
        args.expnum = exp_num  
        exp_num += 1    

        print(vars(args))         
        main(args=args)
