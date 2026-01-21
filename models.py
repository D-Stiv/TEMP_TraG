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
# Modifications: lines 60-61; 108-111; 128-134
# - Integrated timestamp information into edge features for temporal message passing.
#
# Modifications Copyright (c) 2026 D-Stiv


import torch.nn as nn
from torch_geometric.nn import BatchNorm, Linear, to_hetero
from torch_geometric.data import HeteroData
import torch.nn.functional as F
import torch
from torch_scatter import scatter

from pna import TemporalPNAConv

class SumAgg(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, index):
        return scatter(x, index, dim=0, reduce='sum')

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class MultiEdgeAggModule(nn.Module):
    def __init__(self, n_hidden=None, agg_type=None, index=None, ts_reduce='mean'):
        super().__init__()
        self.agg_type = agg_type
        self.ts_reduce = ts_reduce

        if agg_type == 'sum':
            self.agg = SumAgg()
        
    def forward(self, edge_index, edge_attr, simp_edge_batch, times=None):
        _, inverse_indices = torch.unique(simp_edge_batch, return_inverse=True)
        new_edge_index = scatter(edge_index, inverse_indices, dim=1, reduce='mean') if self.agg_type is not None else edge_index
        ts = edge_attr[:, :1] 
        edge_attr = edge_attr[:, 1:]

        new_edge_attr = self.agg(x=edge_attr, index=inverse_indices)
        
        new_ts = scatter(ts, inverse_indices, dim=0, reduce=self.ts_reduce) # find the best aggregation here (min, max, sum, mean)
        new_edge_attr = torch.cat([new_ts, new_edge_attr], dim=1)
        return new_edge_index, new_edge_attr, inverse_indices
    
    def reset_parameters(self):
        try:
            self.agg.reset_parameters()
        except:
            pass

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

            simp_edge_batch_dict = data.simp_edge_batch_dict

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
        return out
    
class GnnHelper(torch.nn.Module):
    def __init__(self, num_gnn_layers, n_hidden=100, edge_updates=False, 
                final_dropout=0.5, index_ = None, deg = None, args=None):
        super().__init__()

        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout
        self.edge_agg_type = args.edge_agg_type
        self.args = args
    
        self.node_agg = 'sum'
     
        self.edge_aggrs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            if args.model == 'pna':
                aggregators = ['mean', 'min', 'max', 'std']
                scalers = ['identity', 'amplification', 'attenuation']
                conv = TemporalPNAConv(in_channels=n_hidden, out_channels=n_hidden,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=n_hidden, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
                
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            edge_agg = MultiEdgeAggModule(n_hidden, agg_type=args.edge_agg_type, index=index_, ts_reduce=args.ts_reduce)
            self.edge_aggrs.append(edge_agg)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))
               

    def forward(self, x, edge_index, edge_attr, simp_edge_batch=None):
        src, dst = edge_index
        for i in range(self.num_gnn_layers):
            n_edge_index, n_edge_attr, inverse_indices  = self.edge_aggrs[i](edge_index, edge_attr, simp_edge_batch)
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, n_edge_index, n_edge_attr)))) / 2
            ts = edge_attr[:, :1]
            edge_attr = edge_attr[:, 1:]
            if self.edge_updates: 
                remapped_edge_attr = torch.index_select(n_edge_attr, 0, inverse_indices) # artificall node attributes 
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], remapped_edge_attr[:, 1:], edge_attr], dim=-1)) / 2
            edge_attr = torch.cat([ts, edge_attr], dim=1) 
        return x, edge_attr


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

            simp_edge_batch_dict = data.simp_edge_batch_dict

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

        return out
    

