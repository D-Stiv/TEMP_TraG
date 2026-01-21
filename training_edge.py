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
# Modifications: lines 42-48; 109-122
# - Monitoring of the best validation model
#
# Modifications Copyright (c) 2026 D-Stiv


import torch
import tqdm
from train_util import evaluate_hetero
from utils import save_model
from metrics import compute_binary_metrics
from data_util import assign_ports_with_cpp, add_arange_ids, AddEgoIds
from load_data import get_loaders
from models import MultiMPNN
from torch_geometric.data import HeteroData
from torch_geometric.nn import summary
from torch_geometric.utils import degree
import wandb
import logging
import numpy as np
import time

def train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data):
    start_time = time.time()
    best_val_f1 = 0
    best_val_metrics = None
    patience_counter = 0
    history = {
        "his_f1": [],
        "his_best_f1": {},
    }
    best_model_state = None
    
    for epoch in range(config.epochs):
        try:
            epoch_start = time.time()
            logging.info(f"---------- Epoch {epoch} ----------")
            total_loss = total_examples = 0
            preds = []
            ground_truths = []
            
            assert model.training, "Training error: Model is not in training mode"

            for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
                if args.ports and args.ports_batch:
                    assign_ports_with_cpp(batch, process_batch=True)
                
                optimizer.zero_grad()
                inds = tr_inds.detach().cpu()
                batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
                batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
                mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)
                
                batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
                batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
                
                batch.to(device)
                out = model(batch)
                pred = out[mask]
                ground_truth = batch['node', 'to', 'node'].y[mask]
                loss = loss_fn(pred, ground_truth)
                
                loss.backward()
                optimizer.step()
                
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
                preds.append(pred.detach().cpu())
                ground_truths.append(ground_truth.detach().cpu())
            
            pred = torch.cat(preds, dim=0).numpy()
            ground_truth = torch.cat(ground_truths, dim=0).numpy()
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

            
            val_metrics = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
            val_f1 = val_metrics['f1_macro']
            history["his_f1"].append(val_f1)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_metrics = val_metrics
                history["his_best_f1"][epoch] = val_f1
                patience_counter = 0
                best_model_state = model.state_dict()
                save_model(model, optimizer, epoch)
                te_metrics = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)
            else:
                patience_counter += 1
                
            if patience_counter >= args.patience:
                logging.info("Early stopping triggered.")
                break
            
            epoch_time = time.time() - epoch_start
            wandb.log({"Loss": total_loss / total_examples, "Time per epoch": epoch_time}, step=epoch)
        except:
            if epoch == 0:
                raise Exception("Error in training. Please check the input data and model configuration.")
            else:
                break
    
    model.load_state_dict(best_model_state)

    infer_start = time.time()
    te_control = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)
    infer_time = time.time() - infer_start
    
    total_time = time.time() - start_time
    eff_epochs = len(history["his_f1"])
    avg_time_epoch = (total_time - infer_time) / eff_epochs
    best_epoch = max(history["his_best_f1"], key=history["his_best_f1"].get, default=0)
    best_f1 = best_val_f1
    
    history.update({
        "tot_time": total_time,
        "avg_time_epoch": avg_time_epoch,
        "infer_time": infer_time,
        "eff_epochs": eff_epochs,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
    })
    
    return model, (te_metrics, best_val_metrics, history)

def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1]) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1])
    
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

def train_edge(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args):
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")    
    device = args.device
    config = wandb.config

    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)
    
    sample_batch = next(iter(tr_loader))

    if isinstance(sample_batch, HeteroData):
        sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:]
        sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
    else:
        sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]

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

    logging.info(summary(model, sample_batch))
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(args.class_weight).to(device))

    if args.reverse_mp:
        model, results = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, config)

    wandb.finish()
    
    return results