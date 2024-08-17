from unittest.mock import Mock

import numpy as np

from disentangle.analysis.stitch_prediction import stitch_predictions
from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.patch_index_manager import TilingMode
from disentangle.data_loader.vanilla_dloader import MultiChDloader


def get_data(*args,**kwargs):
    n = 5
    H = 130
    W = 131
    C = 2
    return np.arange(n*H*W*C).reshape(n,H,W,C)

def get_3Ddata(*args,**kwargs):
    n = 5
    Z = 14
    H = 130
    W = 131
    C = 2
    return np.arange(n*H*W*Z*C).reshape(n,Z,H,W,C)*1.0

def _test_stich_prediction_2Dshifttiling(monkeypatch):
    from disentangle.configs.biosr_config import get_config
    config =get_config()
    config.data.poisson_noise_factor = -1
    config.data.enable_gaussian_noise = False

    monkeypatch.setattr('disentangle.data_loader.vanilla_dloader.get_train_val_data', get_data)
    
    dset = MultiChDloader(
        config.data,
        'random_path',
        # '/group/jug/ashesh/data/BioSR',
        DataSplitType.Val,
        val_fraction=config.training.val_fraction,
        test_fraction=config.training.test_fraction,
        normalized_input=config.data.normalized_input,
        enable_rotation_aug=False,
        max_val = get_data().max(),
        enable_random_cropping=False,#config.data.deterministic_grid is False,
        tiling_mode=TilingMode.ShiftBoundary,
        use_one_mu_std=config.data.use_one_mu_std)

    mean, std = dset.compute_mean_std(allow_for_validation_data=True)
    mean['input'] = np.zeros_like(mean['input'])
    std['input'] = np.ones_like(std['input'])
    mean['target'] = np.zeros_like(mean['target'])
    std['target'] = np.ones_like(std['target'])
    dset.set_mean_std(mean, std)
    dset.set_img_sz(64, 32)
    predictions = []
    for i in range(len(dset)):
        predictions.append(dset[i][1])
    
    predictions = np.stack(predictions)
    stitched_pred = stitch_predictions(predictions, dset)
    assert (stitched_pred== get_data()).all()


def test_stich_prediction_3Dshifttiling(monkeypatch):
    from disentangle.configs.elisa3D_config import get_config
    config =get_config()
    config.data.poisson_noise_factor = -1
    config.data.enable_gaussian_noise = False
    config.data.depth3D = 3
    config.data.image_size = min(64,get_3Ddata().shape[2]//2)

    monkeypatch.setattr('disentangle.data_loader.vanilla_dloader.get_train_val_data', get_3Ddata)
    
    dset = MultiChDloader(
        config.data,
        'random_path',
        # '/group/jug/ashesh/data/BioSR',
        DataSplitType.Val,
        val_fraction=config.training.val_fraction,
        test_fraction=config.training.test_fraction,
        normalized_input=config.data.normalized_input,
        enable_rotation_aug=False,
        max_val = get_3Ddata().max(),
        enable_random_cropping=False,#config.data.deterministic_grid is False,
        tiling_mode=TilingMode.ShiftBoundary,
        use_one_mu_std=config.data.use_one_mu_std)

    mean, std = dset.compute_mean_std(allow_for_validation_data=True)
    mean['input'] = np.zeros_like(mean['input'])
    std['input'] = np.ones_like(std['input'])
    mean['target'] = np.zeros_like(mean['target'])
    std['target'] = np.ones_like(std['target'])
    dset.set_mean_std(mean, std)
    dset.set_img_sz(16, 8)
    predictions = []
    for i in range(len(dset)):
        predictions.append(dset[i][1])
    
    predictions = np.stack(predictions)
    stitched_pred = stitch_predictions(predictions, dset)
    eq_tensor = (stitched_pred== get_3Ddata())
    assert eq_tensor.all()
