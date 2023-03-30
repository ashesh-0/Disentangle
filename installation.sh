conda create -n Disentangle python=3.9
conda activate Disentangle
conda install pytorch==1.13.1 torchvision==0.14.1 git=11.6 -c pytorch -c nvidia
conda install -c conda-forge pytorch-lightning
conda install -c conda-forge wandb -y
conda install -c conda-forge tensorboard -y
python -m pip install ml-collections 
conda install -c anaconda scikit-learn -y
conda install -c conda-forge matplotlib -y
conda install -c anaconda ipython -y
conda install -c conda-forge tifffile -y
python -m pip install albumentations
conda install -c conda-forge nd2reader -y
conda install -c conda-forge yapf -y
conda install -c conda-forge isort -y
python -m pip install pre-commit
conda install -c conda-forge czifile -y

