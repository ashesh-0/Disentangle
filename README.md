# μSplit
<!-- ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"](https://arxiv.org/pdf/2103.14030.pdf) -->
This is the official implementation of [μSplit: image decomposition for fluorescence microscopy](https://arxiv.org/abs/2106.01883).

## Installation
```bash
git clone
cd usplit
pip install -r requirements.txt
```

## Usage
To train a Regular-LC model on the Hagen dataset, run this command:
```bash
python /home/ubuntu/code/uSplit/uSplit/scripts/run.py --workdir=/home/ubuntu/training/uSplit/ -mode=train --datadir=/home/ubuntu/data/ventura_gigascience/ --config=/home/ubuntu/code/uSplit/uSplit/configs/lc_hagen_config.py
```

## Evaluation
```bash
```