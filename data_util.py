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
# No modification has been made to this file. We selected the useful functions from the original code.


import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import OptTensor
import numpy as np
import multiprocessing as mp
import pandas as pd
import tqdm
from torch_geometric.transforms import BaseTransform
from typing import Union


def assign_ports(edges, edge_index, num_nodes):
    """
    Assigns ports to edges based on source and destination nodes.

    Args:
        edges (np.ndarray): Array of shape [num_edges, 3] containing (source, destination, timestamp).
        edge_index (np.ndarray): Array of shape [2, num_edges] containing (source, destination) pairs.
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        ports_1 (list): List of ports for the source nodes.
        ports_2 (list): List of ports for the destination nodes.
    """
    # Initialize port counters for each node
    port_counters = np.zeros(num_nodes, dtype=int)

    # Lists to store ports for each edge
    ports_1 = []
    ports_2 = []

    # Iterate through edges and assign ports
    for edge in edges:
        src, dst, _ = edge  # Extract source, destination, and timestamp
        # Assign port for source node
        ports_1.append(port_counters[src])
        # Assign port for destination node
        ports_2.append(port_counters[dst])
        # Increment port counters for both nodes
        port_counters[src] += 1
        port_counters[dst] += 1

    return ports_1, ports_2

def assign_ports_with_cpp(graph, process_batch=True):
    if isinstance(graph, HeteroGraphData):
        if process_batch:
            '''
                If we have a Hetero Graph then merge the edge indices of forward and reverse edges for consistent port_id assignment.
            '''
            edge_index = torch.cat([graph['node', 'to', 'node'].edge_index, graph['node', 'rev_to', 'node'].edge_index], dim=1)
            timestamp = torch.cat([graph['node', 'to', 'node'].timestamps, graph['node', 'rev_to', 'node'].timestamps], dim=0)
        else:
            edge_index = graph['node', 'to', 'node'].edge_index
            timestamp = graph['node', 'to', 'node'].timestamps

    edges = torch.cat([edge_index.T, timestamp.reshape((-1,1))], dim=1).numpy().astype('int')
    ports_1, ports_2 = assign_ports(edges,edge_index.numpy().astype('int'), graph.num_nodes)
    ports_arr = torch.stack([torch.tensor(ports_1), torch.tensor(ports_2)], dim=1)

    if isinstance(graph, HeteroGraphData):
        if process_batch:
            cut_point = graph['node', 'to', 'node'].num_edges
            graph['node', 'to', 'node'].edge_attr = torch.cat([graph['node', 'to', 'node'].edge_attr, ports_arr[:cut_point, :]], dim=1)
            graph['node', 'rev_to', 'node'].edge_attr = torch.cat([graph['node', 'rev_to', 'node'].edge_attr, ports_arr[cut_point:, :]], dim=1)
        else:
            graph['node', 'to', 'node'].edge_attr = torch.cat([graph['node', 'to', 'node'].edge_attr, ports_arr], dim=1)
   
    return


def to_adj_nodes_with_times(data):
    num_nodes = data.num_nodes
    timestamps = torch.zeros((data.edge_index.shape[1], 1)) if data.timestamps is None else data.timestamps.reshape((-1,1))
    edges = torch.cat((data['node', 'to', 'node'].edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u,v,t in edges:
        u,v,t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out

def to_adj_edges_with_times(data):
    num_nodes = data.num_nodes
    timestamps = torch.zeros((data.edge_index.shape[1], 1)) if data.timestamps is None else data.timestamps.reshape((-1,1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1)
    # calculate adjacent edges with times per node
    adj_edges_out = dict([(i, []) for i in range(num_nodes)])
    adj_edges_in = dict([(i, []) for i in range(num_nodes)])
    for i, (u,v,t) in enumerate(edges):
        u,v,t = int(u), int(v), int(t)
        adj_edges_out[u] += [(i, v, t)]
        adj_edges_in[v] += [(i, u, t)]
    return adj_edges_in, adj_edges_out

def ports(edge_index, adj_list):
    ports = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1: continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:,[0]],return_index=True,axis=0)
        nbs_unique = a[np.sort(idx)][:,0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u,v)] = i
    for i, e in enumerate(edge_index.T):
        ports[i] = ports_dict[tuple(e.numpy())]
    return ports

def time_deltas(data, adj_edges_list):
    time_deltas = torch.zeros(data.edge_index.shape[1], 1)
    if data.timestamps is None:
        return time_deltas
    for v, edges in adj_edges_list.items():
        if len(edges) < 1: continue
        a = np.array(edges)
        a = a[a[:, -1].argsort()]
        a_tds = [0] + [a[i+1,-1] - a[i,-1] for i in range(a.shape[0]-1)]
        tds = np.hstack((a[:,0].reshape(-1,1), np.array(a_tds).reshape(-1,1)))
        for i,td in tds:
            time_deltas[i] = td
    return time_deltas

class GraphData(Data):
    '''This is the homogenous graph object we use for GNN training if reverse MP is not enabled'''
    def __init__(
        self, x: OptTensor = None, edge_index: OptTensor = None, edge_attr: OptTensor = None, y: OptTensor = None, pos: OptTensor = None, 
        readout: str = 'edge', 
        num_nodes: int = None,
        timestamps: OptTensor = None,
        node_timestamps: OptTensor = None,
        **kwargs
        ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.readout = readout
        self.loss_fn = 'ce'
        self.num_nodes = int(self.x.shape[0])
        self.node_timestamps = node_timestamps
        if timestamps is not None:
            self.timestamps = timestamps  
        elif edge_attr is not None:
            self.timestamps = edge_attr[:,0].clone()
        else:
            self.timestamps = None

    def add_ports(self):
        '''Adds port numberings to the edge features'''
        reverse_ports = True
        adj_list_in, adj_list_out = to_adj_nodes_with_times(self)
        in_ports = ports(self.edge_index, adj_list_in)
        out_ports = [ports(self.edge_index.flipud(), adj_list_out)] if reverse_ports else []
        self.edge_attr = torch.cat([self.edge_attr, in_ports] + out_ports, dim=1)
        return self

    def add_time_deltas(self):
        '''Adds time deltas (i.e. the time between subsequent transactions) to the edge features'''
        reverse_tds = True
        adj_list_in, adj_list_out = to_adj_edges_with_times(self)
        in_tds = time_deltas(self, adj_list_in)
        out_tds = [time_deltas(self, adj_list_out)] if reverse_tds else []
        self.edge_attr = torch.cat([self.edge_attr, in_tds] + out_tds, dim=1)
        return self

class HeteroGraphData(HeteroData):
    '''This is the heterogenous graph object we use for GNN training if reverse MP is enabled'''
    def __init__(
        self,
        readout: str = 'edge',
        **kwargs
        ):
        super().__init__(**kwargs)
        self.readout = readout

    @property
    def num_nodes(self):
        return self['node'].x.shape[0]
        
    @property
    def timestamps(self):
        return self['node', 'to', 'node'].timestamps

    def add_ports(self):
        '''Adds port numberings to the edge features'''
        adj_list_in, adj_list_out = to_adj_nodes_with_times(self)
        in_ports = ports(self['node', 'to', 'node'].edge_index, adj_list_in)
        out_ports = ports(self['node', 'rev_to', 'node'].edge_index, adj_list_out)
        self['node', 'to', 'node'].edge_attr = torch.cat([self['node', 'to', 'node'].edge_attr, in_ports], dim=1)
        self['node', 'rev_to', 'node'].edge_attr = torch.cat([self['node', 'rev_to', 'node'].edge_attr, out_ports], dim=1)
        return self

    def add_time_deltas(self):
        '''Adds time deltas (i.e. the time between subsequent transactions) to the edge features'''
        adj_list_in, adj_list_out = to_adj_edges_with_times(self)
        in_tds = time_deltas(self, adj_list_in)
        out_tds = time_deltas(self, adj_list_out)
        self['node', 'to', 'node'].edge_attr = torch.cat([self['node', 'to', 'node'].edge_attr, in_tds], dim=1)
        self['node', 'rev_to', 'node'].edge_attr = torch.cat([self['node', 'rev_to', 'node'].edge_attr, out_tds], dim=1)
        return self
    
def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std

def create_hetero_obj(x,  y,  edge_index,  edge_attr, timestamps, args, simp_edge_batch=None, batch=None):
    '''Creates a heterogenous graph object for reverse message passing'''
    data = HeteroGraphData()

    data['node'].x = x

    if args.data in ['list of node class. datasets']:
        data['node', 'to', 'node'].y = y
    else:
        data['node'].y = y
    data['node', 'to', 'node'].edge_index = edge_index
    data['node', 'rev_to', 'node'].edge_index = edge_index.flipud()
    data['node', 'to', 'node'].edge_attr = edge_attr
    data['node', 'rev_to', 'node'].edge_attr = edge_attr

    if args.flatten_edges:
        data['node', 'to', 'node'].simp_edge_batch = simp_edge_batch
        data['node', 'rev_to', 'node'].simp_edge_batch = simp_edge_batch

    if args.ports and not args.ports_batch:
        #swap the in- and outgoing port numberings for the reverse edges
        data['node', 'rev_to', 'node'].edge_attr[:, [-1, -2]] = data['node', 'rev_to', 'node'].edge_attr[:, [-2, -1]]
    data['node', 'to', 'node'].timestamps = timestamps
    data['node', 'rev_to', 'node'].timestamps = timestamps

    return data

def find_parallel_edges(edge_index):
    simplified_edge_mapping = {}
    simplified_edge_batch = []
    i = 0
    for edge in edge_index.T:
        tuple_edge = tuple(edge.tolist())
        if tuple_edge not in simplified_edge_mapping:
            simplified_edge_mapping[tuple_edge] = i
            i += 1
        simplified_edge_batch.append(simplified_edge_mapping[tuple_edge])
    simplified_edge_batch = torch.LongTensor(simplified_edge_batch)

    return simplified_edge_batch



class AddEgoIds(BaseTransform):
    r"""Add IDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        x = data['node'].x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
       
        return data


def add_arange_ids(data_list):
    '''
    Add the index as an id to the edge features to find seed edges in training, validation and testing.

    Args:
    - data_list (str): List of tr_data, val_data and te_data.
    '''
    for data in data_list:
        if isinstance(data, HeteroData):
            data['node', 'to', 'node'].edge_attr = torch.cat([torch.arange(data['node', 'to', 'node'].edge_attr.shape[0]).view(-1, 1), data['node', 'to', 'node'].edge_attr], dim=1)
            offset = data['node', 'to', 'node'].edge_attr.shape[0]
            data['node', 'rev_to', 'node'].edge_attr = torch.cat([torch.arange(offset, data['node', 'rev_to', 'node'].edge_attr.shape[0] + offset).view(-1, 1), data['node', 'rev_to', 'node'].edge_attr], dim=1)
