from disentangle.nets.hist_noise_model import HistNoiseModel
import numpy as np
import os
import torch.nn as nn
import torch


class DisentNoiseModel(nn.Module):

    def __init__(self, hist1, hist2):
        super().__init__()
        self.n1model = HistNoiseModel(hist1)
        self.n2model = HistNoiseModel(hist2)

    def likelihood(self, obs, signal):
        ll1 = self.n1model.likelihood(obs[:, :1], signal[:, :1])
        ll2 = self.n2model.likelihood(obs[:, 1:], signal[:, 1:])
        return torch.cat([ll1, ll2], dim=1)


def get_noise_model(data_fpath, model_config):
    if 'enable_noise_model' in model_config and model_config.enable_noise_model:
        hist1 = np.load(model_config.noise_model_ch1_fpath)
        hist2 = np.load(model_config.noise_model_ch2_fpath)
        return DisentNoiseModel(hist1, hist2)
    return None