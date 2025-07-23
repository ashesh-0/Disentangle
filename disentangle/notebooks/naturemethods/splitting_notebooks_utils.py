import os

import numpy as np

from disentangle.config_utils import load_config


def get_input(gt, pred_channels:int, gaussian_sigma, poisson_noise_factor):

    synthetic_input = None
    if gt.shape[-1] == pred_channels:
        print('Synthetic input')
        if poisson_noise_factor is not None and poisson_noise_factor != -1.0:
            print('Adding poisson noise with factor', poisson_noise_factor)
            gt = np.random.poisson(gt // poisson_noise_factor) * poisson_noise_factor
        if gaussian_sigma is not None:
            print('Adding gaussian noise with sigma', gaussian_sigma)
            shape = gt.shape
            noise_data = np.random.normal(0, gaussian_sigma, shape).astype(np.int32)
            # noise_data = np.mean(noise_data, axis=-1)
            gt = gt + noise_data
        inp = gt.mean(axis=-1)
        synthetic_input = True
    else:
        print('Real input')
        assert gaussian_sigma is None and poisson_noise_factor is None, 'Expected no gaussian or poisson noise.'
        assert gt.shape[-1] == 1 + pred_channels, f'Expected {pred_channels} channels. Got {gt.shape[-1]} channels.'
        inp = gt[...,-1]
        synthetic_input = False
    return inp, synthetic_input

def get_configdir_from_saved_predictionfile_NM(pref_file_name, train_dir='/group/jug/ashesh/training/disentangle/'):
    """
    Example input: 'pred_training_disentangle_2507_D34-M3-S0-L8_4_1.tif'
    Returns: '/home/ashesh.ashesh/training/disentangle/2507/D34-M3-S0-L8/4'
    """
    fname = pref_file_name
    assert fname[-4:] == '.tif'
    fname = fname[:-4]
    *_, ym, modelcfg, modelid,_ = fname.split('_')
    subdir = '/'.join([ym, modelcfg, modelid])
    datacfg_dir = os.path.join(train_dir, subdir)
    return datacfg_dir

def get_config_from_saved_predictionfile_NM(pref_file_name, train_dir='/group/jug/ashesh/training/disentangle/'):
    """
    Example input: 'pred_training_disentangle_2507_D34-M3-S0-L8_4_1.tif'
    Returns: '/home/ashesh.ashesh/training/disentangle/2507/D34-M3-S0-L8/4/config.pkl'
    """
    datacfg_dir = get_configdir_from_saved_predictionfile_NM(pref_file_name, train_dir)
    config_fpath = os.path.join(datacfg_dir, 'config.pkl')
    assert os.path.exists(config_fpath), f'Config file {config_fpath} does not exist.'
    return load_config(config_fpath)

def get_gaussian_poisson_factors(cfg):
    gaussian_sigma = None if cfg.data.get('enable_gaussian_noise', False) is False else cfg.data.synthetic_gaussian_scale
    poisson_noise_factor = cfg.data.get('poisson_noise_factor', -1.0)
    if gaussian_sigma is not None or poisson_noise_factor != -1.0:
        print(f'Gaussian sigma: {gaussian_sigma}, Poisson noise factor: {poisson_noise_factor}')
    return gaussian_sigma, poisson_noise_factor