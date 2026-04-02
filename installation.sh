# conda create -n naturemethods python=3.9
mamba activate naturemethods
mamba install pytorch torchvision pytorch-cuda -c pytorch -c nvidia -y
mamba install -c conda-forge pytorch-lightning -y
mamba install -c conda-forge wandb -y
mamba install -c conda-forge tensorboard -y
mamba install -c anaconda scikit-learn -y
mamba install -c conda-forge matplotlib -y
mamba install -c anaconda ipython -y
mamba install -c conda-forge tifffile -y
mamba install -c conda-forge nd2reader -y
mamba install -c conda-forge yapf -y
mamba install -c conda-forge isort -y
mamba install -c conda-forge czifile -y
mamba install seaborn -c conda-forge -y
mamba install nbconvert -y
mamba install -c anaconda ipykernel -y
mamba install -c conda-forge czifile -y
mamba install scikit-image -y
python -m pip install ml-collections 
python -m pip install albumentations
python -m pip install pre-commit
pip install nd2
pip install nis2pyr
pip install read-lif