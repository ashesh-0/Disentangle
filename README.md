# μSplit
<img src="./images/SplittedImgs_small.png" width="700px"></img>

This is the official implementation of [μSplit: image decomposition for fluorescence microscopy](https://arxiv.org/abs/2211.12872).

## Installation
```bash
git clone https://github.com/juglab/uSplit.git
cd uSplit
./install_deps.sh
pip install -e .
```
One also needs to create an account on [wandb](https://docs.wandb.ai/quickstart). This is used for logging training and evaluation metrics. In case you do not want to use wandb, you can replace the logger [here](usplit/training.py#L406) with a logger of your choice and comment out [here](usplit/training.py#L349).

## Data
For Hagen et al. dataset, please download the data from [here](http://gigadb.org/dataset/100888). You need to download the following files:
```
1. actin-60x-noise2-highsnr.tif
2. mito-60x-noise2-highsnr.tif
```
For our PaviaATN dataset, please download the data from [here](https://zenodo.org/record/1203745#.YKZ2ZegzZPY). 
For our Synthetic dataset SinosoidalCritters, please download the data from [here](https://zenodo.org/record/1203745#.YKZ2ZegzZPY).

For each dataset, create a separate directory and put the files in the directory. The directory should not contain any other files.

## Usage
To train a Regular-LC model on the Hagen dataset, run this command:
```bash
python /home/ubuntu/code/uSplit/uSplit/scripts/run.py --workdir=/home/ubuntu/training/uSplit/ -mode=train --datadir=/home/ubuntu/data/ventura_gigascience/ --config=/home/ubuntu/code/uSplit/usplit/configs/lc_hagen_config.py
```

## Evaluation
For evaluation, we have provided pre-trained models and the notebook [here](notebooks/evaluation.ipynb). Please download the pre-trained models from [here](https://drive.google.com/drive/folders/1Z3Z3Q2Z3Z3Q2Z3Q2Z3Q2Z3Q2Z3Q2Z3Q2?usp=sharing).