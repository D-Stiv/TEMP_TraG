# MIT License
#
# Copyright (c) 2026 D-Stiv
#
# See the LICENSE file in the repository root for full license text.


import time
import logging
from util import Configuration, setup_config
from load_data_multi import get_edge_data, get_node_data
from training_edge import train_edge
from training_node import train_node
import json


exp_config = Configuration() 
setup_config(exp_config)

def main(args):

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)


    "Node_Class_Data" = 'list of node classification datasets'
    "EDGE_Class_Data" = 'list of edge classification datasets'

    # Check Argument consistency
    if args.data in ['EDGE_Class_Data']:
        args.task = 'edge_class'
    elif args.data in ['Node_Class_Data']:
        args.task = 'node_class'


    t1 = time.perf_counter()

    if (args.data != "Node_Class_Data") and ((args.data != "Node_Class_Data")):
        list_tr_data, list_val_data, list_te_data, tr_inds, val_inds, te_inds = get_edge_data(args, config)     
    elif args.data == "Node_Class_Data":
        list_tr_data, list_val_data, list_te_data, tr_inds, val_inds, te_inds = get_node_data(args, config)
            

    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")

    # Training
    logging.info(f"Running Training")
    if args.data != "Node_Class_Data" and args.data != "Node_Class_Data":
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
