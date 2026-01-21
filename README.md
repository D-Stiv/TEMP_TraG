# TEMP-TraG
This repository contains the code accompanying the paper
“TEMP-TraG: Time-Aware Multigraph Enrichment and Message Passing for Transaction Graphs.”


## Summary
In this repository, you can find the code to train and evaluate TEMP-TraG, a novel graph neural network mechanism that performs temporal features extraction, rule-based graph enrichment and incorporates temporal dynamics into message passing. TEMP-TraG prioritizes more recent transactions when aggregating node messages (leveraging exponential decay), enabling better detection of time-sensitive patterns.

## Setup

- Create a new virtual environment, and activate.
```bash
virtualenv venv
source venv/bin/activate
```

- Install Pytorch and Pytorch Geometric
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install torch-geometric
pip install -r requirements.txt
```

## Data

### Data Format

The minimal required features for transaction data are:
- *transaction_id*: unique identifier of a transaction
- *source*: the source entity of the transaction
- *target*: the target entity of the transaction
- *timestamp*: the timestamp of the transaction
- *src_label*: the label of the source entity (fraudulent or not) [for node classification tasks]
- *tgt_label*: the label of the target entity (fraudulent or not) [for node classification tasks]
- *edge_label*: the label of the transaction (fraudulent or not) [for edge classification tasks]
- *features*: a set of quantitative features associated with the transaction (e.g., amount, type, quantity, etc.)


## Run the Code
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python3 main.py --data dataset_name --model pna --emlps --reverse_mp --ego --flatten_edges --edge_agg_type pna --n_epochs 200 --patience 50 --temporal_mp 1 --seed 42 --tem_feat_class "['TAEC', 'TAC', 'TRM']" --class_weight "[1,5]" --treedepth 8 --median_threshold 1
```


## License

This repository contains code under **multiple licenses**.

### MIT License
All original code written by the author of this repository is
licensed under the MIT License (see `LICENSE`).

We extend PNAConv (PyG) with a temporal message reweighting mechanism based on exponential decay (TemporalPNAConv in ```pna.py```). As a derivative works of **PyTorch Geometric**, it remains Licensed under the MIT License.

### Apache License 2.0

The following files are derivative works from ```MEGA-GNN``` (Multigraph Message Passing with Bi-Directional Multi-Edge Aggregations, Bilgi et al.) and remain under
Apache 2.0:
- ``load_data.py``: we integrated temporal feature extraction and symbolic rule generation (temporal_feature_extraction, detect_outliers, extract_outliers, compute_rule_vectors, create_dgl_graph, create_symbolic_rules, substitute_features)
- ``models.py``: we incorporated timestamp information into edge features for temporal message passing.
- ``train_utils``.py: we computed new metrics for model evaluation
- ``training_node.py``: we inserted a mechanism to monito the best model based on validation perfprmance
- ``training_edge``.py: we inserted a mechanism to monito the best model based on validation perfprmance

The content of ``data_util.py`` is taken from ```MEGA-GNN```. We selected relevant modules and did not insert any own content. They remain licensed under Apache 2.0


The modifications are indicated in the header of each file.

See `LICENSE-APACHE` and `NOTICE` for details.
