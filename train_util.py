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
# Modifications: lines 84; 88
# - Compute and return new metrics for model evaluation
#
# Modifications Copyright (c) 2026 D-Stiv

from tqdm import tqdm
import torch
from metrics import compute_binary_metrics
from data_util import assign_ports_with_cpp

@torch.no_grad()
def evaluate_hetero(loader, inds, model, data, device, args):
    '''Evaluates the model performane for heterogenous graph data.'''
    model.eval()
    assert not model.training, "Test error: Model is not in evaluation mode"

    preds = []
    ground_truths = []
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        
        if args.ports and args.ports_batch:
            # To be consistent, sample the edges for forward and backward edge types.
            assign_ports_with_cpp(batch) 
    
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
        batch_edge_ids = loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #add the seed edges that have not been sampled to the batch
        missing = ~torch.isin(batch_edge_ids, batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu())

        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            # Just ignore this part we rae not entering here args.data == "Small_HI"
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch['node'].n_id
            add_edge_index = data['node', 'to', 'node'].edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = data['node', 'to', 'node'].edge_attr[missing_ids, :].detach().clone()
            add_y = data['node', 'to', 'node'].y[missing_ids].detach().clone()
        
            batch['node', 'to', 'node'].edge_index = torch.cat((batch['node', 'to', 'node'].edge_index, add_edge_index), 1)
            batch['node', 'to', 'node'].edge_attr = torch.cat((batch['node', 'to', 'node'].edge_attr, add_edge_attr), 0)
            batch['node', 'to', 'node'].y = torch.cat((batch['node', 'to', 'node'].y, add_y), 0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        #remove the unique edge id from the edge features, as it's no longer needed
        batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
        batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
        
        with torch.no_grad():
            batch.to(device)

            out = model(batch)
                
            out = out[mask]
            pred = out

            preds.append(pred.detach().cpu())
            ground_truths.append(batch['node', 'to', 'node'].y[mask].detach().cpu())

    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()

    # Compute Metrics
    metrics = compute_binary_metrics(pred, ground_truth)


    model.train()
    return metrics 
