# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----
# Modifications: lines 28-36; 44-46; 49-53; 57-59; 61-64; 68; 75; 85; 141-184; 212-213; 225-226; 233-237; 242-245; 254-255; 258; 268-272; 300-302; 326-335; 342; 349; 356; 362-370; 
# - Integrated temporal feature extraction and symbolic rule generation (temporal_feature_extraction, detect_outliers, extract_outliers, compute_rule_vectors, create_dgl_graph, create_symbolic_rules, substitute_features)
#
# Modifications Copyright (c) 2026 D-Stiv

import pandas as pd
import numpy as np
import torch
import logging
import itertools
from data_util import GraphData, HeteroData, z_norm, create_hetero_obj, find_parallel_edges, assign_ports_with_cpp
from utils import temporal_feature_extraction

from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDOneClassSVM
from utils import allrules, rulerefinement, addsplitnodes_vec
from tqdm import tqdm

def get_edge_data(args, config=None):
    '''Loads the AML transaction data.
    
    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''

    # columns = ['transaction_id', 'source', 'target', 'timestamp', 'f1', 'f2', 'label'] # general schema for edges

    transaction_file = "path to the transactions file"
    df_edges = pd.read_csv(transaction_file)

    # perform temporal_feature_extraction
    date_feat = 'date_time'
    temp_df = temporal_feature_extraction(df_edges, datetime_col=date_feat, src_col='source', dst_col='target', timestamp_col='norm_timestamp', quantitative_col=['f1', 'f2'], categ=args.tem_feat_class)
    temp_cols = temp_df.columns.to_list()
    df_edges = pd.concat([df_edges, temp_df], axis=1)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    df_edges['timestamp'] = df_edges['timestamp'] - df_edges['timestamp'].min()

    max_n_id = df_edges.loc[:, ['source', 'target']].to_numpy().max() + 1
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
    timestamps = torch.Tensor(df_edges['timestamp'].to_numpy())
    y = torch.LongTensor(df_edges['label'].to_numpy())

    logging.info(f"Minority ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
    logging.info(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
    logging.info(f"Number of transactions = {df_edges.shape[0]}")

    edge_features = ['timestamp', 'f1', 'f2'] + temp_cols
    node_features = ['Feature']

    logging.info(f'Edge features being used: {edge_features}')
    logging.info(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
    edge_index = torch.LongTensor(df_edges.loc[:, ['source', 'target']].to_numpy().T)
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    simp_edge_batch = find_parallel_edges(edge_index) if args.flatten_edges else None

    n_days = int(timestamps.max() / (3600 * 24) + 1)
    n_samples = y.shape[0]
    logging.info(f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')

    #data splitting
    daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], [] #irs = Minority ratios, inds = indices, trans = transactions
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_irs.append(y[day_inds].float().mean())
        weighted_daily_irs.append(y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])
    
    split_per = [0.6, 0.2, 0.2]
    daily_totals = np.array(daily_trans)
    d_ts = daily_totals
    I = list(range(len(d_ts)))
    split_scores = dict()
    for i,j in itertools.combinations(I, 2):
        if j >= i:
            split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            split_totals_sum = np.sum(split_totals)
            split_props = [v/split_totals_sum for v in split_totals]
            split_error = [abs(v-t)/t for v,t in zip(split_props, split_per)]
            score = max(split_error) #- (split_totals_sum/total) + 1
            split_scores[(i,j)] = score
        else:
            continue

    i,j = min(split_scores, key=split_scores.get)
    #split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
    split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
    logging.info(f'Calculate split: {split}')

    #Now, we seperate the transactions based on their indices in the timestamp array
    split_inds = {k: [] for k in range(3)}
    for i in range(3):
        for day in split[i]:
            split_inds[i].append(daily_inds[day]) #split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately

    tr_inds = torch.cat(split_inds[0])
    val_inds = torch.cat(split_inds[1])
    te_inds = torch.cat(split_inds[2])

    logging.info(f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
            f"{y[tr_inds].float().mean() * 100 :.2f}% || Train days: {split[0][:5]}")
    logging.info(f"Total val samples: {val_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
        f"{y[val_inds].float().mean() * 100:.2f}% || Val days: {split[1][:5]}")
    logging.info(f"Total test samples: {te_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
        f"{y[te_inds].float().mean() * 100:.2f}% || Test days: {split[2][:5]}")
    
    #Creating the final data objects
    e_tr = tr_inds.numpy()
    e_val = np.concatenate([tr_inds, val_inds])

    tr_edge_index,  tr_edge_attr,  tr_y,  tr_edge_times  = edge_index[:,e_tr],  edge_attr[e_tr],  y[e_tr],  timestamps[e_tr]
    val_edge_index, val_edge_attr, val_y, val_edge_times = edge_index[:,e_val], edge_attr[e_val], y[e_val], timestamps[e_val]
    te_edge_index,  te_edge_attr,  te_y,  te_edge_times  = edge_index,          edge_attr,        y,        timestamps

    def detect_outliers(features, nu=0.1, gamma=1.0, n_components=500, seed=42):
        rbf_feature = RBFSampler(gamma=gamma, random_state=seed, n_components=n_components)
        clf = make_pipeline(rbf_feature, SGDOneClassSVM(nu=nu, random_state=seed))
        clf.fit(features)
        return clf.predict(features)

    def extract_outliers(features, labels, predictions):
        mask = predictions == -1
        outlier_features = features[mask]
        outlier_labels = labels[mask]
        
        # Handle case where all outliers have same label
        if torch.unique(outlier_labels).numel() == 1:
            # Treat SVM output as pseudo-labels instead
            predictions = np.where(predictions == 1, 0, 1)
            outlier_features = features
            outlier_labels = torch.tensor(predictions, dtype=torch.long)
        
        return outlier_features, outlier_labels

    def compute_rule_vectors(edge_attr_list, rules):
        return [torch.tensor([addsplitnodes_vec(attr, rules) for attr in tqdm(edge_attr)], dtype=torch.float32)
                for edge_attr in tqdm(edge_attr_list)]

    # Outlier detection
    predictions = detect_outliers(tr_edge_attr.numpy())  # sklearn requires numpy input
    print(np.unique(predictions, return_counts=True))

    if len(np.unique(predictions)) == 2:
        outlier_features, outlier_labels = extract_outliers(tr_edge_attr, tr_y, predictions)
        print(torch.unique(outlier_labels, return_counts=True))

        decisions, _ = allrules(outlier_features, outlier_labels, 5, 100)
        refined_rules = rulerefinement(decisions)

        if refined_rules:
            edge_attr_list = [tr_edge_attr, val_edge_attr, te_edge_attr]
            rule_vectors = compute_rule_vectors(edge_attr_list, refined_rules)
            
            tr_edge_attr = torch.cat((tr_edge_attr, rule_vectors[0]), dim=1)
            val_edge_attr = torch.cat((val_edge_attr, rule_vectors[1]), dim=1)
            te_edge_attr = torch.cat((te_edge_attr, rule_vectors[2]), dim=1)
    
    tr_x, val_x, te_x = x, x, x

    tr_simp_edge_batch, val_simp_edge_batch, te_simp_edge_batch = simp_edge_batch[e_tr], simp_edge_batch[e_val], simp_edge_batch
    tr_data = GraphData (x=tr_x,  y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr,  timestamps=tr_edge_times , simp_edge_batch = tr_simp_edge_batch)
    val_data = GraphData(x=val_x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times, simp_edge_batch = val_simp_edge_batch)
    te_data = GraphData (x=te_x,  y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr,  timestamps=te_edge_times , simp_edge_batch = te_simp_edge_batch)
    
    # Adding ports and time-deltas if applicable
    if args.ports and not args.ports_batch:
        logging.info(f"Start: adding ports")
        assign_ports_with_cpp(tr_data, process_batch=False)
        assign_ports_with_cpp(val_data, process_batch=False)
        assign_ports_with_cpp(te_data, process_batch=False)
        logging.info(f"Done: adding ports")

    
    tr_data.x = val_data.x = te_data.x = z_norm(tr_data.x)
    if not args.model == 'rgcn':
        tr_data.edge_attr, val_data.edge_attr, te_data.edge_attr = z_norm(tr_data.edge_attr), z_norm(val_data.edge_attr), z_norm(te_data.edge_attr)
    else:
        tr_data.edge_attr[:, :-1], val_data.edge_attr[:, :-1], te_data.edge_attr[:, :-1] = z_norm(tr_data.edge_attr[:, :-1]), z_norm(val_data.edge_attr[:, :-1]), z_norm(te_data.edge_attr[:, :-1])

    # Create heterogenous if reverese MP is enabled
    if args.reverse_mp:
        tr_data = create_hetero_obj(tr_data.x,  tr_data.y,  tr_data.edge_index,  tr_data.edge_attr, tr_data.timestamps, args, tr_simp_edge_batch)
        val_data = create_hetero_obj(val_data.x,  val_data.y,  val_data.edge_index,  val_data.edge_attr, val_data.timestamps, args, val_simp_edge_batch)
        te_data = create_hetero_obj(te_data.x,  te_data.y,  te_data.edge_index,  te_data.edge_attr, te_data.timestamps, args, te_simp_edge_batch)
    
    logging.info(f'train data object: {tr_data}')
    logging.info(f'validation data object: {val_data}')
    logging.info(f'test data object: {te_data}')

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds


def get_node_data(args, config=None):
    '''Loads the Node Classification data.
    
    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''

    # node_columns = ['node', 'first_transaction_timestamp', 'label'] # general schema for nodes
    # edge_columns = ['transaction_id', 'source', 'target', 'timestamp', 'f1', 'f2'] # general schema for edges

    nodes = pd.read_csv("path to the nodes file")
    edges = pd.read_csv("path to the edges file")

    logging.info(f'Available Edge Features: {edges.columns.tolist()}')
    logging.info(f"Number of Nodes: {nodes.shape[0]}")
    logging.info(f"Number of Edges: {edges.shape[0]}")
    logging.info(f"Number of Minority Nodes: {nodes['label'].sum()}")
    logging.info(f"Minority Ratio: {nodes['label'].sum()/ nodes.shape[0] * 100 :.2f}%")

    nodes = nodes.sort_values(by='first_transaction_timestamp').reset_index(drop=True)

    assign_dict = {}
    for row in nodes.itertuples():
        assign_dict[row[1]] = row[0]

    def assign_node_ids(node_id):
        return assign_dict[node_id] 

    edges['target'] = edges['target'].apply(assign_node_ids)
    edges['source'] = edges['source'].apply(assign_node_ids)
    nodes.drop(columns=['node'], inplace=True)

    edge_features = ['f1', 'f2', 'timestamp']
    node_features = ['Feature']

    logging.info(f'Edge features being used: {edge_features}')
    logging.info(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    max_n_id = nodes.shape[0]

    splits = [0.65, 0.15, 0.20]

    t1 = nodes.iloc[int(max_n_id * splits[0])]['first_transaction_timestamp']
    t2 = nodes.iloc[int(max_n_id * (splits[0] + splits[1]))]['first_transaction_timestamp']

    tr_nodes = nodes.loc[nodes['first_transaction_timestamp'] <= t1]
    val_nodes = nodes.loc[nodes['first_transaction_timestamp'] <= t2]
    te_nodes = nodes

    tr_nodes_max_id = tr_nodes.index[-1]
    val_nodes_max_id = val_nodes.index[-1]
    te_nodes_max_id = te_nodes.index[-1]

    tr_inds = torch.arange(0, tr_nodes_max_id+1)
    val_inds = torch.arange(tr_nodes_max_id+1, val_nodes_max_id+1)
    te_inds = torch.arange(val_nodes_max_id+1, te_nodes_max_id+1)


    logging.info(f"Total train samples: {tr_nodes.shape[0] / nodes.shape[0] * 100 :.2f}% || IR: "
            f"{tr_nodes['label'].mean() * 100 :.2f}%")
    logging.info(f"Total validation samples: {val_inds.shape[0] / nodes.shape[0] * 100 :.2f}% || IR: "
            f"{val_nodes.loc[val_inds.numpy(),'label'].mean() * 100 :.2f}%")
    logging.info(f"Total test samples: {te_inds.shape[0] / nodes.shape[0] * 100 :.2f}% || IR: "
            f"{te_nodes.loc[te_inds.numpy(), 'label'].mean() * 100 :.2f}%")
    

    tr_nodes_max_id = tr_nodes.index[-1]
    val_nodes_max_id = val_nodes.index[-1]
    te_nodes_max_id = te_nodes.index[-1]

    split_name = []
    for row in edges.itertuples():
        '''
        row[0]: index
        row[1]: source
        row[2]: target
        row[3]: f1
        row[4]: timestamp
        '''
        if row[1] <= tr_nodes_max_id and row[2] <= tr_nodes_max_id:
            if row[4] <= t1:
                split_name.append('train')
            elif row[4] > t1 and row[4] <= t2:
                split_name.append('val')
            else:
                split_name.append('test') 
            continue 
        elif row[1] <= val_nodes_max_id and row[2] <= val_nodes_max_id:
            if row[4] <= t2:
                split_name.append('val')
            else:
                split_name.append('test') 
            continue
        else:
            split_name.append('test')


    edges['split'] = split_name
    edges['timestamp'] = edges['timestamp'] - edges['timestamp'].min()

    # normalize timestamp with z-score
    edges['norm_timestamp'] = (edges['timestamp'] - edges['timestamp'].mean()) / edges['timestamp'].std()
    edges['norm_timestamp'] = edges['norm_timestamp'].fillna(0)

    # perform temporal_feature_extraction
    date_feat = 'date_time'
    temp_df = temporal_feature_extraction(edges, datetime_col=date_feat, src_col='source', dst_col='target', timestamp_col='norm_timestamp', quantitative_col=['f1', 'f2'], categ=args.tem_feat_class)
    temp_cols = temp_df.columns.to_list()
    edges = pd.concat([edges, temp_df], axis=1)
    edge_features += temp_cols

    tr_edges = edges.loc[edges['split'] == 'train']
    val_edges = edges.loc[(edges['split'] == 'train') | (edges['split'] == 'val')]
    te_edges = edges 

    tr_x = torch.tensor(np.ones(tr_nodes.shape[0])).float()
    tr_edge_index = torch.LongTensor(tr_edges.loc[:, ['source', 'target']].to_numpy().T)
    tr_edge_attr = torch.tensor(tr_edges.loc[:, edge_features].to_numpy()).float()
    tr_edge_times = torch.Tensor(tr_edges['timestamp'].to_numpy())
    tr_y = torch.LongTensor(tr_nodes['label'].to_numpy())
    tr_simp_edge_batch = find_parallel_edges(tr_edge_index)

    val_x = torch.tensor(np.ones(val_nodes.shape[0])).float()
    val_edge_index = torch.LongTensor(val_edges.loc[:, ['source', 'target']].to_numpy().T)
    val_edge_attr = torch.tensor(val_edges.loc[:, edge_features].to_numpy()).float()
    val_edge_times = torch.Tensor(val_edges['timestamp'].to_numpy())
    val_y = torch.LongTensor(val_nodes['label'].to_numpy())
    val_simp_edge_batch = find_parallel_edges(val_edge_index)

    te_x = torch.tensor(np.ones(te_nodes.shape[0])).float()
    te_edge_index = torch.LongTensor(te_edges.loc[:, ['source', 'target']].to_numpy().T)
    te_edge_attr = torch.tensor(te_edges.loc[:, edge_features].to_numpy()).float()
    te_edge_times = torch.Tensor(te_edges['timestamp'].to_numpy())
    te_y = torch.LongTensor(te_nodes['label'].to_numpy())
    te_simp_edge_batch = find_parallel_edges(te_edge_index)

    logging.info(f"Start: extracting symbolic rules")
    from utils import create_dgl_graph, create_symbolic_rules, substitute_features
    g = create_dgl_graph(tr_x, tr_edge_index, tr_edge_attr, tr_y,
                    val_x, val_edge_index, val_edge_attr, val_y,
                    te_x, te_edge_index, te_edge_attr, te_y)
    node_features, _ = create_symbolic_rules(g, g.ndata['train_mask'], treedepth=args.treedepth, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
    if node_features is None:
        raise ValueError("No symbolic rules were created. Please check the data and the edge rules configuration.")
    tr_x, val_x, te_x = substitute_features(g, node_features)

    tr_data = GraphData (x=tr_x,  y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr, timestamps=tr_edge_times , simp_edge_batch = tr_simp_edge_batch)
    val_data = GraphData(x=val_x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr,timestamps=val_edge_times, simp_edge_batch = val_simp_edge_batch)
    te_data = GraphData (x=te_x,  y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr, timestamps=te_edge_times , simp_edge_batch = te_simp_edge_batch)

    if args.ports and not args.ports_batch:
        logging.info(f"Start: adding ports")
        assign_ports_with_cpp(tr_data, process_batch=False)
        assign_ports_with_cpp(val_data, process_batch=False)
        assign_ports_with_cpp(te_data, process_batch=False)
        # tr_data.add_ports()
        # val_data.add_ports()
        # te_data.add_ports()
        logging.info(f"Done: adding ports")


    tr_data.x, val_data.x, te_data.x = z_norm(tr_data.x), z_norm(val_data.x), z_norm(te_data.x)

    tr_data.edge_attr, val_data.edge_attr, te_data.edge_attr = z_norm(tr_data.edge_attr), z_norm(val_data.edge_attr), z_norm(te_data.edge_attr)

    if args.reverse_mp:
        tr_data = create_hetero_obj(tr_data.x,  tr_data.y,  tr_data.edge_index,  tr_data.edge_attr, tr_data.timestamps, args, tr_simp_edge_batch)
        val_data = create_hetero_obj(val_data.x,  val_data.y,  val_data.edge_index,  val_data.edge_attr, val_data.timestamps, args, val_simp_edge_batch)
        te_data = create_hetero_obj(te_data.x,  te_data.y,  te_data.edge_index,  te_data.edge_attr, te_data.timestamps, args, te_simp_edge_batch)

    logging.info(f'train data object: {tr_data}')
    logging.info(f'validation data object: {val_data}')
    logging.info(f'test data object: {te_data}')

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds

import os
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader

def get_loaders_eth(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args):
    ''' 
        Sampled nodes are sorted based on the order in which they were sampled. In particular, the first batch_size nodes represent 
        the set of original mini-batch nodes.

        In particular, the data loader will add the following attributes to the returned mini-batch:
            `batch_size`        The number of seed nodes (first nodes in the batch)
            `n_id`              The global node index for every sampled node
            `e_id`              The global edge index for every sampled edge
            `input_id`          The global index of the input_nodes
            `num_sampled_nodes` The number of sampled nodes in each hop
            `num_sampled_edges` The number of sampled edges in each hop
    '''

    # Worker initialization function for DataLoader
    def worker_init_fn(worker_id):
        import random
        worker_seed = args.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        os.environ["PYTHONHASHSEED"] = str(worker_seed)

    if isinstance(tr_data, HeteroData):

        tr_loader = NeighborLoader(tr_data, 
                                   num_neighbors= {key: args.num_neighs for key in tr_data.edge_types}, 
                                   batch_size=args.batch_size, 
                                   shuffle=False, 
                                   transform=transform, 
                                   input_nodes=('node', None), 
                                   worker_init_fn=worker_init_fn,
                                   num_workers=4
                                   )

        val_loader = NeighborLoader(val_data, 
                                    num_neighbors= {key: args.num_neighs for key in val_data.edge_types}, 
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    transform=transform,
                                    input_nodes=('node', val_inds), 
                                    worker_init_fn=worker_init_fn,
                                    num_workers=4
                                    )
        
        te_loader = NeighborLoader(te_data, 
                                    num_neighbors= {key: args.num_neighs for key in te_data.edge_types}, 
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    transform=transform,
                                    input_nodes=('node', te_inds), 
                                    worker_init_fn=worker_init_fn,
                                    num_workers=4
                                    )

    return tr_loader, val_loader, te_loader


def get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args):
    # Worker initialization function for DataLoader
    def worker_init_fn(worker_id):
        import random
        worker_seed = args.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        os.environ["PYTHONHASHSEED"] = str(worker_seed)
    
    if isinstance(tr_data, HeteroData):
        tr_edge_label_index = tr_data['node', 'to', 'node'].edge_index
        tr_edge_label = tr_data['node', 'to', 'node'].y


        tr_loader =  LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), tr_edge_label_index), 
                                    edge_label=tr_edge_label, batch_size=args.batch_size, shuffle=False, transform=transform, 
                                    worker_init_fn=worker_init_fn, num_workers=4)
        
        val_edge_label_index = val_data['node', 'to', 'node'].edge_index[:,val_inds]
        val_edge_label = val_data['node', 'to', 'node'].y[val_inds]


        val_loader =  LinkNeighborLoader(val_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), val_edge_label_index), 
                                    edge_label=val_edge_label, batch_size=args.batch_size, shuffle=False, transform=transform, 
                                    worker_init_fn=worker_init_fn, num_workers=4)
        
        te_edge_label_index = te_data['node', 'to', 'node'].edge_index[:,te_inds]
        te_edge_label = te_data['node', 'to', 'node'].y[te_inds]


        te_loader =  LinkNeighborLoader(te_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), te_edge_label_index), 
                                    edge_label=te_edge_label, batch_size=args.batch_size, shuffle=False, transform=transform, 
                                    worker_init_fn=worker_init_fn, num_workers=4)

    return tr_loader, val_loader, te_loader
