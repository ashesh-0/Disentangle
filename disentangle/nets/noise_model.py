import os

import numpy as np
import torch
import torch.nn as nn

from disentangle.nets.gmm_nnbased_noise_model import DeepGMMNoiseModel
from disentangle.nets.gmm_noise_model import GaussianMixtureNoiseModel
from disentangle.nets.hist_noise_model import HistNoiseModel


class DisentNoiseModel(nn.Module):

    def __init__(self, n1model, n2model):
        super().__init__()
        self.n1model = n1model
        self.n2model = n2model

    def likelihood(self, obs, signal):
        ll1 = self.n1model.likelihood(obs[:, :1], signal[:, :1])
        ll2 = self.n2model.likelihood(obs[:, 1:], signal[:, 1:])
        return torch.cat([ll1, ll2], dim=1)


def get_noise_model(model_config):
    if 'enable_noise_model' in model_config and model_config.enable_noise_model:
        print(f'Noise model Ch1: {model_config.noise_model_ch1_fpath}')
        print(f'Noise model Ch2: {model_config.noise_model_ch2_fpath}')
        if model_config.noise_model_type == 'hist':
            hist1 = np.load(model_config.noise_model_ch1_fpath)
            nmodel1 = HistNoiseModel(hist1)
            hist2 = np.load(model_config.noise_model_ch2_fpath)
            nmodel2 = HistNoiseModel(hist2)
        elif model_config.noise_model_type == 'gmm':
            # nmodel1 = GaussianMixtureNoiseModel(params=np.load(model_config.noise_model_ch1_fpath))
            # nmodel2 = GaussianMixtureNoiseModel(params=np.load(model_config.noise_model_ch2_fpath))
            nmodel1 = DeepGMMNoiseModel(params=np.load(model_config.noise_model_ch1_fpath))
            nmodel2 = DeepGMMNoiseModel(params=np.load(model_config.noise_model_ch2_fpath))
            if model_config.get('noise_model_learnable', False):
                nmodel1.make_learnable()
                nmodel2.make_learnable()

        return DisentNoiseModel(nmodel1, nmodel2)
    return None