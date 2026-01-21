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
# Modifications: lines 42-48; 124-143; 168-180
# - Monitoring of the best validation model
#
# Modifications Copyright (c) 2026 D-Stiv

import torch
import tqdm
from train_util import evaluate_hetero
from utils import save_model
from metrics import compute_binary_metrics
from data_util import assign_ports_with_cpp, add_arange_ids, AddEgoIds
from load_data import get_loaders_eth
from models import MultiMPNN
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn import summary
from torch_geometric.utils import degree
import wandb
import logging
import numpy as np
import time


def train_hetero_eth(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data):
    # Initialize variables
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    his_f1 = []  # List of all validation F1 scores
    his_best_f1 = {}  # Dictionary of best validation F1 scores (epoch: val_f1)
    start_time = time.time()  # Start time for total training
    best_epoch = 0  # Epoch with the best validation F1
    best_val_metrics = None

    for epoch in range(config.epochs):
        try:
            epoch_start_time = time.time()  # Start time for the current epoch
            logging.info(f"---------- Epoch {epoch} ----------")
            total_loss = total_examples = 0
            preds = []
            ground_truths = []
            
            assert model.training, "Training error: Model is not in training mode"

            for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
                if args.ports and args.ports_batch:
                    assign_ports_with_cpp(batch) 
                
                optimizer.zero_grad()

                inds = tr_inds.detach().cpu()
                batch_node_inds = inds[batch['node'].input_id.detach().cpu()]
                batch_node_ids = tr_loader.data['node'].x.detach().cpu()[batch_node_inds, 0]
                mask = torch.isin(batch['node'].x[:, 0].detach().cpu(), batch_node_ids)
                
                batch['node'].x = batch['node'].x[:, 1:]
                batch.to(device)

                out = model(batch)
                pred = out[mask]
                ground_truth = batch['node'].y[mask]
                loss = loss_fn(pred, ground_truth)

                loss.backward()
                optimizer.step()

                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()

                preds.append(pred.detach().cpu())
                ground_truths.append(ground_truth.detach().cpu())
                
            pred = torch.cat(preds, dim=0).numpy()
            ground_truth = torch.cat(ground_truths, dim=0).numpy()

            # Compute metrics
            metrics = compute_binary_metrics(pred, ground_truth)
            for metric_name, metric_value in metrics.items():
                if 'matrix' in metric_name:
                    wandb.log({f"{metric_name}/train": str(metric_value)}, step=epoch)
                    logging.info(f'Train {metric_name.capitalize()}: {str(metric_value)}')
                    continue
                if isinstance(metric_value, np.ndarray):
                    for class_idx, value in enumerate(metric_value):
                        wandb.log({f"{metric_name}/class_{class_idx}/train": value}, step=epoch)
                        logging.info(f'Train {metric_name.capitalize()} (Class {class_idx}): {value:.4f}')
                else:
                    wandb.log({f"{metric_name}/train": metric_value}, step=epoch)
                    logging.info(f'Train {metric_name.capitalize()}: {metric_value:.4f}')

            # Evaluate
            val_metrics = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)

            for metric_name, metric_value in val_metrics.items():
                if 'matrix' in metric_name:
                    wandb.log({f"{metric_name}/validation": str(metric_value)}, step=epoch)
                    logging.info(f'Val {metric_name.capitalize()}: {str(metric_value)}')
                    continue
                if isinstance(metric_value, np.ndarray):
                    for class_idx, value in enumerate(metric_value):
                        wandb.log({f"{metric_name}/class_{class_idx}/validation": value}, step=epoch)
                        logging.info(f'Val {metric_name.capitalize()} (Class {class_idx}): {value:.4f}')
                else:
                    wandb.log({f"{metric_name}/validation": metric_value}, step=epoch)
                    logging.info(f'Val {metric_name.capitalize()}: {metric_value:.4f}')

            wandb.log({"Loss": total_loss / total_examples}, step=epoch)
            val_f1 = val_metrics['f1_macro']

            # Append validation F1 to history
            his_f1.append(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = model.state_dict()
                best_epoch = epoch
                his_best_f1[epoch] = val_f1  # Add to best F1 history
                best_val_metrics = val_metrics
                save_model(model, optimizer, epoch)
                te_metrics = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logging.info(f"Early stopping at epoch {epoch} due to no improvement in validation F1.")
                break

            # Log epoch time
            epoch_time = time.time() - epoch_start_time
            logging.info(f"Epoch {epoch} time: {epoch_time:.2f} seconds")
        except:
            if epoch == 0:
                raise ValueError("Error in the first epoch. Please check the data and the model.")
            else:
                break

    # Load the best model state
    model.load_state_dict(best_model_state)

    # Inference on test data
    infer_start_time = time.time()
    _ = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)
    infer_time = time.time() - infer_start_time

    # Calculate total time and average time per epoch
    end_time = time.time()
    tot_time = end_time - start_time
    avg_time_epoch = (infer_start_time-start_time) / (epoch + 1)
    eff_epochs = epoch + 1  # Effective epochs used for training

    # Create the results dictionary
    results = {
        "his_f1": his_f1,
        "his_best_f1": his_best_f1,
        "tot_time": tot_time,
        "avg_time_epoch": avg_time_epoch,
        "infer_time": infer_time,
        "eff_epochs": eff_epochs,
        "best_epoch": best_epoch,
        "best_f1": best_val_f1,
    }

    return model, (te_metrics, best_val_metrics, results)


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
            assign_ports_with_cpp(batch) 
    
        inds = inds.detach().cpu()
        batch_node_inds = inds[batch['node'].input_id.detach().cpu()]
        batch_node_ids = loader.data['node'].x.detach().cpu()[batch_node_inds, 0]
        mask = torch.isin(batch['node'].x[:, 0].detach().cpu(), batch_node_ids)

        # remove the unique node id from the node features, as it's no longer needed
        batch['node'].x = batch['node'].x[:, 1:]
        
        with torch.no_grad():
            batch.to(device)
            out = model(batch)
                
            out = out[mask]
            pred = out
            preds.append(pred.detach().cpu())
            ground_truths.append(batch['node'].y[mask].detach().cpu())

    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()
    metrics = compute_binary_metrics(pred, ground_truth)

    model.train()
    return metrics # f1, auc, precision, recall


# Get the model for heterogenous data
def get_model(sample_batch, config, args):
    n_feats = (sample_batch.x.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node'].x.shape[1] - 1)
    e_dim = sample_batch.edge_attr.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node', 'to', 'node'].edge_attr.shape[1]
    
    index_ = sample_batch.simp_edge_batch if not isinstance(sample_batch, HeteroData) else sample_batch['node', 'to', 'node'].simp_edge_batch

    
    # Instead of in-degree use Fan-in
    if not isinstance(sample_batch, HeteroData):
        s_edges = torch.unique(sample_batch.edge_index, dim=1)
        d = degree(s_edges[1], num_nodes=sample_batch.num_nodes, dtype=torch.long)

    deg = torch.bincount(d, minlength=1)

    model = MultiMPNN(num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2, 
                    n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim, 
                    final_dropout=config.final_dropout, index_=index_, deg=deg, args=args)   
     
    return model

# Train node classification model
def train_node(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, config):
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")  
    device = args.device
    config = wandb.config

    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders_eth(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    sample_batch = next(iter(tr_loader)) 

    if args.ports and args.ports_batch:
        # Add a placeholder for the port features so that the model is loaded correctly!
        if isinstance(sample_batch, HeteroData):
            sample_batch['node', 'to', 'node'].edge_attr = torch.cat([sample_batch['node', 'to', 'node'].edge_attr, torch.zeros((sample_batch['node', 'to', 'node'].edge_attr.shape[0], 2))], dim=1)
            sample_batch['node', 'rev_to', 'node'].edge_attr = torch.cat([sample_batch['node', 'rev_to', 'node'].edge_attr, torch.zeros((sample_batch['node', 'rev_to', 'node'].edge_attr.shape[0], 2))], dim=1)
        else:
            sample_batch.edge_attr = torch.cat([sample_batch.edge_attr, torch.zeros((sample_batch.edge_attr.shape[0], 2))], dim=1)

    model = get_model(sample_batch, config, args)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    sample_batch.to(device)

    if isinstance(sample_batch, HeteroData):
        sample_batch['node'].x = sample_batch['node'].x[:, 1:]
    else:
        sample_batch.x = sample_batch.x[:, 1:]


    logging.info(summary(model, sample_batch))
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(args.class_weight).to(device))

    if args.reverse_mp:
        model, results = train_hetero_eth(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, config)
    
    return results


