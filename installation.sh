mamba create -n split_hpc python=3.9
mamba activate split_hpc
# python -m pip install cellpose => installs cpu version.
# mamba install torchvision==0.14.1  -c pytorch -c nvidia -y
mamba install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
# pip install --force-reinstall -v "numpy==1.25.2"
# mamba install conda-forge::stardist
# python -m pip install cellpose # I had to manually change two things. 
mamba install -c conda-forge pytorch-lightning==2.0.8 -y
mamba install -c conda-forge wandb -y
mamba install -c conda-forge tensorboard -y
python -m pip install ml-collections 
mamba install -c anaconda scikit-learn -y
mamba install -c conda-forge matplotlib -y
mamba install -c anaconda ipython -y
mamba install -c conda-forge tifffile -y
python -m pip install albumentations
mamba install -c conda-forge nd2reader -y
mamba install -c conda-forge yapf -y
mamba install -c conda-forge isort -y
python -m pip install pre-commit
mamba install -c conda-forge czifile -y
mamba install seaborn -c conda-forge -y
mamba install nbconvert -y
mamba install -c anaconda ipykernel -y
mamba install -c conda-forge czifile -y
pip install nd2
pip install scikit-image
# numpy 2 has issues.
pip install --upgrade numpy==1.25.2
# install deepinv locally.
pip install nis2pyr
pip install cellpose
python -m pip install faiss-gpu
pip install papermill