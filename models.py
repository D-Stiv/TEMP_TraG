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
# Modifications:
# - Integrated timestamp information into edge features for temporal message passing.
# - MultiMPNN forward method.
#
# Modifications Copyright (c) 2026 D-Stiv


import torch.nn as nn
from torch_geometric.nn import BatchNorm, Linear, to_hetero
from modif_gnn.gin import GINEConv
from modif_gnn.pna import PNAConv
from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import HeteroData
import torch.nn.functional as F
import torch
import logging
import numpy as np
from torch_scatter import scatter
from torch_geometric.utils import degree
from genagg import GenAgg
from genagg.MLPAutoencoder import MLPAutoencoder



class MultiMPNN(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, n_hidden=100, 
                 edge_updates=False,edge_dim=None, final_dropout=0.5, 
                 index_ = None, deg=None, args=None):
        super().__init__()
        self.args = args
        self.n_hidden = n_hidden
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        if args.reverse_mp:
            self.edge_emb_rev = nn.Linear(edge_dim, n_hidden)

        self.gnn = GnnHelper(num_gnn_layers=num_gnn_layers, n_hidden=n_hidden, edge_updates=edge_updates, final_dropout=final_dropout,
                             index_=index_, deg=deg, args=args)
        
        if args.reverse_mp:
            self.gnn  = to_hetero(self.gnn, metadata= (['node'], [('node', 'to', 'node'), ('node', 'rev_to', 'node')]), aggr='mean')

        if args.task == 'edge_class':
            self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                                Linear(25, n_classes))
        elif args.task == 'node_class':
            self.mlp = nn.Sequential(Linear(n_hidden, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                                Linear(25, n_classes))

    def forward(self, data):
        if isinstance(data, HeteroData):

            # Initial Embedding Layers
            x_dict = {"node": self.node_emb(data['node'].x)}
            
            num_edges = data['node', 'to', 'node'].edge_index.size(1)
            num_edges_rev = data['node', 'rev_to', 'node'].edge_index.size(1)
            device = data['node', 'to', 'node'].edge_index.device
            if self.args.temporal_mp:
                ts = data['node', 'to', 'node'].timestamps.unsqueeze(1)
                ts_rev = data['node', 'rev_to', 'node'].timestamps.unsqueeze(1)
                ts = (ts - ts.min()) / (ts.max() - ts.min()) 
                ts_rev = (ts_rev - ts_rev.min()) / (ts_rev.max() - ts_rev.min())
            else:                
                ts = torch.ones(num_edges, 1, device=device)
                ts_rev = torch.ones(num_edges_rev, 1, device=device)

            edge_attr_dict = {
                    ("node", 'to', 'node'): torch.cat([ts, self.edge_emb(data['node', 'to', 'node'].edge_attr)], dim=1), 
                    ("node", 'rev_to', 'node'): torch.cat([ts_rev, self.edge_emb_rev(data['node', 'rev_to', 'node'].edge_attr)], dim=1),    
                } 

            simp_edge_batch_dict = data.simp_edge_batch_dict if self.args.flatten_edges else None

            # Message Passing Layers
            x_dict, edge_attr_dict = self.gnn(x_dict, data.edge_index_dict, edge_attr_dict, simp_edge_batch_dict)
            x = x_dict['node']
            edge_attr = edge_attr_dict['node', 'to', 'node'][:, 1:] # from 1 given that the first column is the timestamp

            # Prediction Heads
            if self.args.task == 'edge_class':
                x = x[data['node', 'to', 'node'].edge_index.T].reshape(-1, 2*self.n_hidden).relu()
                x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
                out = self.mlp(x)
            elif self.args.task == 'node_class':
                out = self.mlp(x)

        else:
            # Initial Embedding Layers
            x = self.node_emb(data.x)
            edge_attr = self.edge_emb(data.edge_attr) 
            simp_edge_batch = data.simp_edge_batch if self.args.flatten_edges else None

            # Message Passing Layers
            x, edge_attr = self.gnn(x, data.edge_index, edge_attr, simp_edge_batch)

            # Prediction Heads
            if self.args.task == 'edge_class':
                x = x[data.edge_index.T].reshape(-1, 2*self.n_hidden).relu()
                x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
                out = self.mlp(x)
            elif self.args.task == 'node_class':
                out = self.mlp(x)
        return out
    

