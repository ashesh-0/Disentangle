# μSplit
<!-- ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"](https://arxiv.org/pdf/2103.14030.pdf) -->
This is the official implementation of [μSplit: image decomposition for fluorescence microscopy](https://arxiv.org/abs/2106.01883).

## Installation
```bash
git clone https://github.com/juglab/uSplit.git
cd uSplit
./install_deps.sh
pip install -e .
```
One also needs to create an account on [wandb](https://docs.wandb.ai/quickstart). This is used for logging training and evaluation metrics. In case you do not want to use wandb, you can replace the logger [here](usplit/training.py#L406) with a logger of your choice and comment out [here](usplit/training.py#L349).

## Usage
To train a Regular-LC model on the Hagen dataset, run this command:
```bash
python /home/ubuntu/code/uSplit/uSplit/scripts/run.py --workdir=/home/ubuntu/training/uSplit/ -mode=train --datadir=/home/ubuntu/data/ventura_gigascience/ --config=/home/ubuntu/code/uSplit/uSplit/configs/lc_hagen_config.py
```

## Evaluation
```bash
```