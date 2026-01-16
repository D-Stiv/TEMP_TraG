# TEMP-TraG
**Important:** This repository contains a review version of the code accompanying the paper
“TEMP-TraG: Time-Aware Multigraph Enrichment and Message Passing for Transaction Graphs.”

The code is provided exclusively for academic review and evaluation purposes.
It is not intended for production use, redistribution, or derivative work at this stage.

A cleaned, documented, and fully reproducible version of the code will be released publicly upon paper acceptance.

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
- Lastly, install genagg (Kortvelesy, Ryan, Steven Morad, and Amanda Prorok. "Generalised f-mean aggregation for graph neural networks." Advances in Neural Information Processing Systems 36 (2023): 34439-34450.)
```bash
cd genagg 
pip install -e .
cd ..
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
You can run TeMP-TraG with and without edge feature aggregation for GIN and PNA models. To use different datasets, change the dataset name and run the model with the additional parameters provided in util.py. Below are some examples to run the code from the command line.

### Our Method
- TeMP-TraG (PNA with agg)
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python3 main.py --data dataset_name --model pna --emlps --reverse_mp --ego --flatten_edges --edge_agg_type pna --n_epochs 200 --patience 50 --temporal_mp 1 --seed 42 --tem_feat_class "['TAEC', 'TAC', 'TRM']" --class_weight "[1,5]" --treedepth 8 --median_threshold 1
```

### Other Variants

- TeMP-TraG (GIN w/o agg)
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python3 main_multi.py --data dataset_name --model gin --emlps --reverse_mp --ego --n_epochs 200 --patience 50 --temporal_mp 1 --seed 42 --tem_feat_class "['TAEC', 'TAC', 'TRM']" --class_weight "[1,5]" --treedepth 8 --median_threshold 1 
```
- TeMP-TraG (PNA w/o agg)
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python3 main_multi.py --data dataset_name --model pna --emlps --reverse_mp --ego --n_epochs 500 --patience 50 --temporal_mp 1 --seed 42 --tem_feat_class "['TAEC', 'TAC', 'TRM']" --class_weight "[1,5]" --treedepth 8 --median_threshold 1
```
- TeMP-TraG (GIN with agg)
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python3 main.py --data dataset_name --model gin --emlps --reverse_mp --ego --flatten_edges --edge_agg_type gin --n_epochs 200 --patience 50 --temporal_mp 1 --seed 42 --tem_feat_class "['TAEC', 'TAC', 'TRM']" --class_weight "[1,5]" --treedepth 8 --median_threshold 1
```

## License

This repository contains code under **multiple licenses**.

### MIT License
All original code written by the author of this repository is
licensed under the MIT License (see `LICENSE`).

### Apache License 2.0
This repository includes modified source code from
**PyTorch Geometric**, licensed under the Apache License 2.0.

The following files are derivative works and remain under
Apache 2.0:
- load_data.py
- load_data_multi.py
- models.py
- training_node.py
- training_edge.py

See `LICENSE-APACHE` and `NOTICE` for details.
