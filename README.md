# Objective

Here, our goal is to disentangle the images by taking as input the sum of the two images. For example, given image A &
B, input to our model is A+B and the output is A,B

## Training
### Go to the code directory and set the python path
```
cd /home/ubuntu/code/Disentangle
export PYTHONPATH=`pwd`
```
### Run the training script
```
python /home/ubuntu/code/Disentangle/disentangle/scripts/run.py --workdir=/home/ubuntu/training/disentangle/ -mode=train --datadir=/home/ubuntu/data/ventura_gigascience/ --config=/home/ubuntu/code/Disentangle/disentangle/configs/twotiff_unet_config.py --logdir=/home/ubuntu/logs
```
### Run the evaluation script
```
python /home/ubuntu/code/disentangle/scripts/evaluate.py /home/ubuntu/training/disentangle/2210/D7-M3-S0-L0/77/ 64 64 /home/ubuntu/data/ventura_gigascience/
```
