# ConRep4CO
[ICLR 2026] ConRep4CO: Contrastive Representation Learning of Combinatorial Optimization Instances across Types

## Installation

You can run the following commands to get started:

```bash
conda create -n newenv python=3.9
conda activate newenv
bash scripts/install.sh
```

## Datasets

To generate our used datasets, you can refer to the following scripts:

```bash
# generate datasets
bash scripts/gen_data.sh
```


## Training

To train the graph and SAT models, you can refer to the following scripts:

```bash
# train models
bash scripts/run_training.sh
```

