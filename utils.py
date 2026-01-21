# MIT License
#
# Copyright (c) 2026 D-Stiv
#
# See the LICENSE file in the repository root for full license text.


import torch
import tqdm
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from typing import Dict
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
import numpy as np


def temp_aware_edge_clust(edge_df, datetime_col, src_col, dst_col, timestamp_col):
    taec_df = edge_df.copy()

    # temporal distance between consecutive transactions having the the same source / destination
    taec_df['src_dt_prev'] = taec_df.groupby(src_col)[timestamp_col].diff().fillna(pd.Timedelta(seconds=0))
    taec_df['dst_dt_prev'] = taec_df.groupby(dst_col)[timestamp_col].diff().fillna(pd.Timedelta(seconds=0))
    taec_df['src_dt_next'] = taec_df.groupby(src_col)[timestamp_col].diff(-1).abs().fillna(pd.Timedelta(seconds=0))
    taec_df['dst_dt_next'] = taec_df.groupby(dst_col)[timestamp_col].diff(-1).abs().fillna(pd.Timedelta(seconds=0))

    # convert the the value from type time delta to seconds
    taec_df['src_dt_prev'] = taec_df['src_dt_prev'].apply(lambda x: x if isinstance(x, float) else x.total_seconds())
    taec_df['dst_dt_prev'] = taec_df['dst_dt_prev'].apply(lambda x: x if isinstance(x, float) else x.total_seconds())
    taec_df['src_dt_next'] = taec_df['src_dt_next'].apply(lambda x: x if isinstance(x, float) else x.total_seconds())
    taec_df['dst_dt_next'] = taec_df['dst_dt_next'].apply(lambda x: x if isinstance(x, float) else x.total_seconds())

    return taec_df.iloc[:,4:]  

def time_augmented_clustering(edge_df, datetime_col):
    tac_df = edge_df.copy()

    # cos and sine for hour of day, day of week, and month of year
    tac_df['hour_sin'] = np.sin(2 * np.pi * (tac_df[datetime_col].dt.hour +1) / 24)
    tac_df['hour_cos'] = np.cos(2 * np.pi * (tac_df[datetime_col].dt.hour+1) / 24)
    tac_df['day_sin'] = np.sin(2 * np.pi * (tac_df[datetime_col].dt.dayofweek +1) / 7)
    tac_df['day_cos'] = np.cos(2 * np.pi * (tac_df[datetime_col].dt.dayofweek +1) / 7)
    tac_df['month_sin'] = np.sin(2 * np.pi * (tac_df[datetime_col].dt.month +1) / 12)
    tac_df['month_cos'] = np.cos(2 * np.pi * (tac_df[datetime_col].dt.month +1) / 12)

    return tac_df.iloc[:, 4:]  

def temporal_rule_mining(edge_df, datetime_col, src_col, dst_col, key_cols):
    trm_df = edge_df.copy()
    
    # define time delta as (max - min) / 100 in days
    time_delta = ((trm_df[datetime_col].max() - trm_df[datetime_col].min()) / 100).days
    if time_delta == 0:
        time_delta = 15
    # get the average of key for each in the interval [t, t - time_delta]
    for key_col in key_cols:
        trm_df[f'{key_col}_src_avg_prev'] = trm_df.groupby(src_col)[key_col].transform(lambda x: x.shift(1).rolling(window=time_delta, min_periods=1).mean())
        trm_df[f'{key_col}_dst_avg_prev'] = trm_df.groupby(dst_col)[key_col].transform(lambda x: x.shift(1).rolling(window=time_delta, min_periods=1).mean())
        trm_df[f'{key_col}_src_avg_next'] = trm_df.groupby(src_col)[key_col].transform(lambda x: x.shift(-1).rolling(window=time_delta, min_periods=1).mean())
        trm_df[f'{key_col}_dst_avg_next'] = trm_df.groupby(dst_col)[key_col].transform(lambda x: x.shift(-1).rolling(window=time_delta, min_periods=1).mean())

    # fill NaN values with 0
    trm_df.fillna(0, inplace=True)

    return trm_df.iloc[:, 4+len(key_cols):]  

def temporal_feature_extraction(edge_df, datetime_col, src_col, dst_col, timestamp_col, key_cols, categ):
    categ = eval(categ) 
    temp_df = []
    if 'TAEC' in categ:
        print("Extracting temporal features...")
        taec_df = temp_aware_edge_clust(edge_df[[datetime_col, src_col, dst_col, timestamp_col]], datetime_col, src_col, dst_col, timestamp_col)
        temp_df.append(taec_df)
    if 'TAC' in categ:        
        print("Extracting time-augmented clustering features...")
        tac_df = time_augmented_clustering(edge_df[[datetime_col, src_col, dst_col, timestamp_col]], datetime_col)
        temp_df.append(tac_df)
    if 'TRM' in categ:
        print("Extracting temporal rule mining features...")
        trm_df = temporal_rule_mining(edge_df[[datetime_col, src_col, dst_col, timestamp_col, *key_cols]], datetime_col, src_col, dst_col, key_cols)
        temp_df.append(trm_df)

    print("Combining all temporal features...")
    temp_df = pd.concat(temp_df, axis=1)
    return temp_df

def exponential_decay(N0, decay_constant, t):
    """
    Compute exponential decay: N(t) = N0 * e^(-λt)
    
    Parameters:
    N0 (float): Initial quantity
    decay_constant (float): Decay constant (λ)
    t (np.ndarray or torch.Tensor): 1D time tensor
    
    Returns:
    np.ndarray or torch.Tensor: 1D tensor of decayed quantities
    """
    N0 = 1e-10 if N0 <= 0 else N0  # Avoid division by zero
    if isinstance(t, torch.Tensor):
        return N0 * torch.exp(-decay_constant * t)
    return N0 * np.exp(-decay_constant * t)



def create_dgl_graph(tr_x, tr_edge_index, tr_edge_attr, tr_y,
                    val_x, val_edge_index, val_edge_attr, val_y,
                    te_x, te_edge_index, te_edge_attr, te_y):
    import dgl

    # Combine all nodes
    num_nodes = tr_x.shape[0] + val_x.shape[0] + te_x.shape[0]
    
    # Create graph
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    
    # Combine all edge indices (need to offset the node indices for val and test)
    val_offset = tr_x.shape[0]
    te_offset = val_offset + val_x.shape[0]

    # Store all edges and their attributes
    all_edges = []
    all_edge_feats = []
    
    # Add training edges
    src, dst = tr_edge_index
    all_edges.extend(list(zip(src.tolist(), dst.tolist())))
    all_edge_feats.append(tr_edge_attr)
    
    # Add validation edges (with offset)
    src, dst = val_edge_index
    all_edges.extend(list(zip((src + val_offset).tolist(), (dst + val_offset).tolist())))
    all_edge_feats.append(val_edge_attr)
    
    # Add test edges (with offset)
    src, dst = te_edge_index
    all_edges.extend(list(zip((src + te_offset).tolist(), (dst + te_offset).tolist())))
    all_edge_feats.append(te_edge_attr)
    
    # Add all edges to the graph at once to maintain proper edge IDs
    src, dst = zip(*all_edges)
    g.add_edges(torch.tensor(src), torch.tensor(dst))

    # Combine all edge features
    g.edata['feat'] = torch.cat(all_edge_feats, dim=0)
   
    # Combine all node features
    all_feat = torch.cat([tr_x, val_x, te_x], dim=0)
    all_labels = torch.cat([tr_y, val_y, te_y], dim=0)
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[:tr_x.shape[0]] = True
    val_mask[val_offset:val_offset + val_x.shape[0]] = True
    test_mask[te_offset:te_offset + te_x.shape[0]] = True
    
    # Add node features and masks to graph
    g.ndata['feat'] = all_feat
    g.ndata['label'] = all_labels
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask  
    g.ndata['test_mask'] = test_mask  
    
    return g



def rulerefinement(decisions, median_threshold=1):
    # dicdec = {} 
    dicdec = defaultdict(list)
    for d in decisions:
        if (d[0], d[1]) in dicdec:
            dicdec[(d[0], d[1])].append(d[2])
        else:
            dicdec[(d[0], d[1])] = [d[2]]

    # order dicdec by feature sign and id
    dicdec = dict(sorted(dicdec.items(), key=lambda item: (item[0][1], item[0][0])))

    rules = defaultdict(list)
    for (f_id, sign), values in dicdec.items():
        data = np.array(values).reshape(-1, 1)
        if median_threshold:
            # median selection
            rules[sign].append([f_id, np.median(data)])
        else:
            raise NotImplementedError("Only median_threshold is implemented")
    print("Number of decisions (after refinement): " + str(len(rules['>='])+len(rules['<='])))

    return rules


def gettreestruct(dot_data):
    string = dot_data.split('\n')
    parent = {}

    for s in string:
        if '->' in s:
            mother = int(s.split(' -> ')[0])
            child = int(s.split(' -> ')[1].split(' ')[0])
            parent[child] = mother

    return parent

def getDecisions(tr):
    splitnodes = []
    for ind, imp in enumerate(tr.tree_.impurity):
        splitnodes.append(ind)

    dot_data = tree.export_graphviz(tr, out_file=None)
    parents = gettreestruct(dot_data)

    decisions = []
    for ind, r in enumerate(splitnodes):
        if r in tr.tree_.children_left:
            decisions.append([tr.tree_.feature[parents[r]], "<=", tr.tree_.threshold[parents[r]]])
        elif r in tr.tree_.children_right:
            decisions.append([tr.tree_.feature[parents[r]], ">=", tr.tree_.threshold[parents[r]]])

    return decisions

def allrules(nodes, labels, treedepth=5, num_trees=100):
    clf = RandomForestClassifier(max_depth=treedepth, n_estimators=num_trees)
    clf = clf.fit(nodes.to(torch.device("cpu")), labels.to(torch.device("cpu")))

    splitnodes = []
    for tr in clf.estimators_:
        decisions = getDecisions(tr)
        for d in decisions:
            splitnodes.append(d)

    numdecisions = len(splitnodes)
    print("Number of decisions: " + str(numdecisions))

    return splitnodes, numdecisions


def addsplitnodes_vec(f, decisions):
    """Vectorized version of addsplitnodes for a single feature vector."""
    # decisions are in the form of defaultdict(list) ('sign': list of ('f_id', 'threshold'))
    # decisions = torch.tensor(decisions)
    f = torch.tensor(f)

    vectorized_decisions = []
    # decisions = np.array(decisions)
    for sign, values in decisions.items():
        decisions_id = torch.tensor(values)[:, 0].long().sort()[0]
        decisions_features = f[decisions_id]
        decisions_threshold = torch.tensor(values)[:, 1].float()
        if sign == "<=":
            # vectorized_decision = (decisions_features <= decisions_threshold).long()
            vectorized_decision = (decisions_features < decisions_threshold).long()
            vectorized_decisions.append(vectorized_decision)

    return torch.cat(vectorized_decisions, dim=0).tolist()


def get_edge_rules(g, train_mask, device='cpu', treedepth=5, min_cluster_size=10, min_samples=10, median_threshold=1):
    g = g.to(device)
    train_mask = train_mask.to(device)

    # Get Train_ids
    train_nodes = torch.nonzero(train_mask).squeeze(1)

    # Get all edges connected to training nodes
    all_edge_features = []
    edge_label = []
    added_ids = set()  # Use a set for faster membership checks

    for edge_id in tqdm(g.edges(form='eid')):
        source, target = g.find_edges(edge_id)

        # Check if either source or target is in training nodes
        if source in train_nodes or target in train_nodes:
            if edge_id not in added_ids:
                added_ids.add(edge_id)
                edge_label.append(int(g.ndata['label'][source] or g.ndata['label'][target]))  # if either node is illicit, the edge is illicit
                all_edge_features.append(g.edata['feat'][edge_id])
    
    all_edge_features = torch.stack(all_edge_features)
    edge_label = torch.tensor(edge_label, dtype=torch.long)

    rbf_feature = RBFSampler(gamma=1.0, random_state=42, n_components=500)
    clf = make_pipeline(rbf_feature, SGDOneClassSVM(nu=0.1, random_state=42))
    clf.fit(all_edge_features)


    predictions = clf.predict(all_edge_features)

    predictions = np.where(predictions == 1, 0, 1)
    final_edge_features = all_edge_features
    final_edge_label = torch.tensor(predictions, dtype=torch.long)

    decisions, _ = allrules(final_edge_features, final_edge_label, treedepth=treedepth)
    refined_rules = rulerefinement(decisions, min_cluster_size=min_cluster_size, min_samples=min_cluster_size, median_threshold=median_threshold)

    return refined_rules

def create_symbolic_rules(g, train_mask, device='cpu', treedepth=5, min_cluster_size=10, min_samples=10, median_threshold=1):

    print("Find Edge Rules")
    edge_rules = get_edge_rules(g, train_mask, treedepth=treedepth, min_cluster_size=min_cluster_size, min_samples=min_samples, median_threshold=median_threshold)
    
    if len(edge_rules) == 0:
        print("No edge rules found. Returning empty node features.")
        return None
    n_edge_feats = g.edata['feat'].shape[1]

    num_nodes = g.num_nodes()
    edge_feats = g.edata['feat'].cpu().numpy()

    # Precompute symbolic rule vectors for all edges
    rule_vectors = np.array([addsplitnodes_vec(edge_feat, edge_rules) for edge_feat in tqdm(edge_feats)])

    edge_dict = defaultdict(list)
    u, v = g.edges()
    u, v = u.tolist(), v.tolist()

    print("Collecting edge features for symbolic rules...")
    for i, (src, dst) in tqdm(enumerate(zip(u, v))):
        edge_dict[(src, dst)].append(i)

    print("Collecting edge features for symbolic rules... done.")

    pair_to_union_feat = {}  # (src, dst) → mean(rule + edge_feat)
    print("Computing union features for each (src, dst) pair...")
    for pair, eids in tqdm(edge_dict.items()):
        rules = rule_vectors[eids]  # shape: (len(eids), len(edge_rules))
        feats = edge_feats[eids]    # shape: (len(eids), n_edge_feats)
        union = np.concatenate([rules, feats], axis=1)  # shape: (len(eids), total_dim)
        pair_to_union_feat[pair] = torch.tensor(np.mean(union, axis=0), dtype=torch.float32, device=device)

    print("Computing union features for each (src, dst) pair... done.")

    # Step 2: Collect node features
    node_feats = g.ndata['feat']
    if node_feats.dim() == 1:
        node_feats = node_feats.unsqueeze(1)  # Ensure 2D shape (num_nodes, feat_dim)

    # Aggregate union features per node (i.e., destination)
    node_union_feats = defaultdict(list)

    n_rules = rule_vectors.shape[1]  # Number of symbolic rules
    print("Aggregating union features per node...")
    for (src, dst), union_feat in tqdm(pair_to_union_feat.items()):
        node_union_feats[dst].append(union_feat)

    zero_union = torch.zeros(n_rules + n_edge_feats, device=device)

    # Build enriched node features
    enriched_node_feats = []
    for nid in tqdm(range(num_nodes)):
        node_feat = node_feats[nid]
        if node_feat.dim() == 0:
            node_feat = node_feat.unsqueeze(0)
        
        if nid in node_union_feats:
            union_feats = torch.stack(node_union_feats[nid], dim=0)
            mean_union_feat = union_feats.mean(dim=0)
        else:
            mean_union_feat = zero_union

        enriched_feat = torch.cat([node_feat.to(device), mean_union_feat], dim=0).float()
        enriched_node_feats.append(enriched_feat)

    print("Aggregating union features per node... done.")

    return torch.stack(enriched_node_feats), edge_rules

def substitute_features(g, node_features):
    # Split rules into train, val, test
    tr_feats = node_features[g.ndata['train_mask']]
    val_feats = node_features[g.ndata['val_mask']]
    te_feats = node_features[g.ndata['test_mask']]
    
    # Concatenate original features with rules
    tr_x_new = tr_feats
    val_x_new = val_feats
    te_x_new = te_feats

    return tr_x_new, val_x_new, te_x_new


def save_model(model, optimizer, epoch):
    # Save the model in a dictionary
    save_path = f'path to the directory where to save the model / file name'

    torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, 
                save_path
            )
    
def load_model(model, device, config):
    save_path = f'path to the directory where to save the model / file name'
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

