import json
import os

import numpy as np
import torch
import torch.nn as nn

from disentangle.core.model_type import ModelType
from disentangle.nets.gmm_nnbased_noise_model import DeepGMMNoiseModel
from disentangle.nets.gmm_noise_model import GaussianMixtureNoiseModel
from disentangle.nets.hist_noise_model import HistNoiseModel


class DisentNoiseModel(nn.Module):

    def __init__(self, n1model, n2model):
        super().__init__()
        self.n1model = n1model
        self.n2model = n2model

    def likelihood(self, obs, signal):
        if obs.shape[1] == 1:
            assert signal.shape[1] == 1
            assert self.n2model is None
            return self.n1model.likelihood(obs, signal)

        ll1 = self.n1model.likelihood(obs[:, :1], signal[:, :1])
        ll2 = self.n2model.likelihood(obs[:, 1:], signal[:, 1:])
        return torch.cat([ll1, ll2], dim=1)


def noise_model_config_sanity_check(noise_model_fpath, config, channel_key=None):
    config_fpath = os.path.join(os.path.dirname(noise_model_fpath), 'config.json')
    with open(config_fpath, 'r') as f:
        noise_model_config = json.load(f)

    # make sure that the amount of noise is consistent.
    if 'add_gaussian_noise_std' in noise_model_config:
        # data.enable_gaussian_noise = False
        # config.data.synthetic_gaussian_scale = 1000
        assert 'enable_gaussian_noise' in config.data
        assert config.data.enable_gaussian_noise == True, 'Gaussian noise is not enabled'

        assert 'synthetic_gaussian_scale' in config.data
        assert noise_model_config[
            'add_gaussian_noise_std'] == config.data.synthetic_gaussian_scale, f'{noise_model_config["add_gaussian_noise_std"]} != {config.data.synthetic_gaussian_scale}'
    if 'poisson_noise_factor' in noise_model_config:
        assert 'poisson_noise_factor' in config.data
        assert noise_model_config[
            'poisson_noise_factor'] == config.data.poisson_noise_factor, f'{noise_model_config["poisson_noise_factor"]} != {config.data.poisson_noise_factor}'

    # make sure that the same file is used for noise model and data.
    if channel_key is not None:
        fname = noise_model_config['fname']
        if '' in fname:
            fname.remove('')
        assert len(fname) == 1
        fname = fname[0]
        cur_data_fpath = os.path.join(config.datadir, config.data[channel_key])
        nm_data_fpath = os.path.join(noise_model_config['datadir'], fname)
        assert cur_data_fpath == nm_data_fpath, f'{cur_data_fpath} != {nm_data_fpath}'


def get_noise_model(config):
    if 'enable_noise_model' in config.model and config.model.enable_noise_model:
        if config.model.model_type == ModelType.Denoiser:
            if config.model.noise_model_type == 'hist':
                if config.model.denoise_channel == 'Ch1':
                    print(f'Noise model Ch1: {config.model.noise_model_ch1_fpath}')
                    hist1 = np.load(config.model.noise_model_ch1_fpath)
                    nmodel1 = HistNoiseModel(hist1)
                    nmodel2 = None
                elif config.model.denoise_channel == 'Ch2':
                    print(f'Noise model Ch2: {config.model.noise_model_ch2_fpath}')
                    hist2 = np.load(config.model.noise_model_ch2_fpath)
                    nmodel1 = HistNoiseModel(hist2)
                    nmodel2 = None
                elif config.model.denoise_channel == 'input':
                    print(f'Noise model Ch1: {config.model.noise_model_ch1_fpath}')
                    hist1 = np.load(config.model.noise_model_ch1_fpath)
                    nmodel1 = HistNoiseModel(hist1)
                    nmodel2 = None
            elif config.model.noise_model_type == 'gmm':
                if config.model.denoise_channel == 'Ch1':
                    nmodel_fpath = config.model.noise_model_ch1_fpath
                    print(f'Noise model Ch1: {nmodel_fpath}')
                    nmodel1 = GaussianMixtureNoiseModel(params=np.load(nmodel_fpath))
                    noise_model_config_sanity_check(nmodel_fpath, config, 'ch1_fname')
                    nmodel2 = None
                elif config.model.denoise_channel == 'Ch2':
                    nmodel_fpath = config.model.noise_model_ch2_fpath
                    print(f'Noise model Ch2: {nmodel_fpath}')
                    nmodel1 = GaussianMixtureNoiseModel(params=np.load(nmodel_fpath))
                    noise_model_config_sanity_check(nmodel_fpath, config, 'ch2_fname')
                    nmodel2 = None
                elif config.model.denoise_channel == 'input':
                    nmodel_fpath = config.model.noise_model_ch1_fpath
                    print(f'Noise model input: {nmodel_fpath}')
                    nmodel1 = GaussianMixtureNoiseModel(params=np.load(nmodel_fpath))
                    noise_model_config_sanity_check(nmodel_fpath, config)
                    nmodel2 = None
            else:
                raise ValueError(f'Invalid denoise_channel: {config.model.denoise_channel}')
        elif config.model.noise_model_type == 'hist':
            print(f'Noise model Ch1: {config.model.noise_model_ch1_fpath}')
            print(f'Noise model Ch2: {config.model.noise_model_ch2_fpath}')

            hist1 = np.load(config.model.noise_model_ch1_fpath)
            nmodel1 = HistNoiseModel(hist1)
            hist2 = np.load(config.model.noise_model_ch2_fpath)
            nmodel2 = HistNoiseModel(hist2)
        elif config.model.noise_model_type == 'gmm':
            print(f'Noise model Ch1: {config.model.noise_model_ch1_fpath}')
            print(f'Noise model Ch2: {config.model.noise_model_ch2_fpath}')

            nmodel1 = GaussianMixtureNoiseModel(params=np.load(config.model.noise_model_ch1_fpath))
            nmodel2 = GaussianMixtureNoiseModel(params=np.load(config.model.noise_model_ch2_fpath))
            noise_model_config_sanity_check(config.model.noise_model_ch1_fpath, config, 'ch1_fname')
            noise_model_config_sanity_check(config.model.noise_model_ch2_fpath, config, 'ch2_fname')
            # nmodel1 = DeepGMMNoiseModel(params=np.load(config.model.noise_model_ch1_fpath))
            # nmodel2 = DeepGMMNoiseModel(params=np.load(config.model.noise_model_ch2_fpath))

        if config.model.get('noise_model_learnable', False):
            nmodel1.make_learnable()
            if nmodel2 is not None:
                nmodel2.make_learnable()

        return DisentNoiseModel(nmodel1, nmodel2)
    return None
