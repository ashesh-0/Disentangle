# Objective

Here, our goal is to disentangle the images by taking as input the sum of the two images. For example, given image A &
B, input to our model is A+B and the output is A,B

## Data (Hagen Et al.)
Download two data files: actin-60x-noise2-highsnr.tif and mito-60x-noise2-highsnr.tif from http://gigadb.org/dataset/100888 and put them in a folder (later used with --datadir)
## Training
### Go to the code directory and set the python path
```
cd /home/ubuntu/code/Disentangle
export PYTHONPATH=`pwd`
```
### Run the training script
To train on Hagen Et al. dataset, run the following command.
```
python /home/ubuntu/code/Disentangle/disentangle/scripts/run.py --workdir=/home/ubuntu/training/disentangle/ -mode=train --datadir=/home/ubuntu/data/ventura_gigascience/ --config=/home/ubuntu/code/Disentangle/disentangle/configs/twotiff_config.py --logdir=/home/ubuntu/logs
```

## Evaluation
For tiled prediction with 24 pixels of inner padding and 64 being patch size, tile size becomes \(64-2*24=16\). This is the configuration being used in the below command.
One also needs to give the checkpoint directory (first argument) where the weights for the model are stored along with the training configuration.
```
python /home/ubuntu/code/disentangle/scripts/evaluate.py /home/ubuntu/training/disentangle/2210/D7-M3-S0-L0/77/ 64 16 --datadir=/home/ubuntu/data/ventura_gigascience/
```
