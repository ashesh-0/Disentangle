import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


class LikelihoodModule(nn.Module):
    def distr_params(self, x):
        pass

    @staticmethod
    def mean(params):
        pass

    @staticmethod
    def mode(params):
        pass

    @staticmethod
    def sample(params):
        pass

    def log_likelihood(self, x, params):
        pass

    def forward(self, input_, x):
        distr_params = self.distr_params(input_)
        mean = self.mean(distr_params)
        mode = self.mode(distr_params)
        sample = self.sample(distr_params)
        if x is None:
            ll = None
        else:
            ll = self.log_likelihood(x, distr_params)
        dct = {
            'mean': mean,
            'mode': mode,
            'sample': sample,
            'params': distr_params,
        }
        return ll, dct


class NoiseModelLikelihood(LikelihoodModule):
    def __init__(self, ch_in, color_channels, data_mean, data_std, noiseModel):
        super().__init__()
        self.parameter_net = nn.Conv2d(ch_in, color_channels, kernel_size=3, padding=1)
        self.data_mean = data_mean
        self.data_std = data_std
        self.noiseModel = noiseModel

    def distr_params(self, x):
        x = self.parameter_net(x)
        # mean, lv = x.chunk(2, dim=1)
        mean = x
        lv = None
        params = {
            'mean': mean,
            'logvar': lv,
        }
        return params

    @staticmethod
    def mean(params):
        return params['mean']

    @staticmethod
    def mode(params):
        return params['mean']

    @staticmethod
    def sample(params):
        # p = Normal(params['mean'], (params['logvar'] / 2).exp())
        # return p.rsample()
        return params['mean']

    def log_likelihood(self, x, params):
        predicted_s_denormalized = params['mean'] * self.data_std + self.data_mean
        x_denormalized = x * self.data_std + self.data_mean
        predicted_s_cloned = predicted_s_denormalized
        predicted_s_reduced = predicted_s_cloned.permute(1, 0, 2, 3)

        x_cloned = x_denormalized
        x_cloned = x_cloned.permute(1, 0, 2, 3)
        x_reduced = x_cloned[0, ...]

        likelihoods = self.noiseModel.likelihood(x_reduced, predicted_s_reduced)
        logprob = torch.log(likelihoods)
        return logprob


class GaussianLikelihood(LikelihoodModule):
    def __init__(self, ch_in, color_channels, predict_logvar=False):
        super().__init__()
        # If True, then we also predict pixelwise logvar.
        self.predict_logvar = predict_logvar
        assert isinstance(self.predict_logvar, bool)
        self.parameter_net = nn.Conv2d(ch_in, color_channels * (1 + int(self.predict_logvar)), kernel_size=3, padding=1)

    def get_mean_lv(self, x):
        x = self.parameter_net(x)
        if self.predict_logvar:
            mean, lv = x.chunk(2, dim=1)
        else:
            mean = x
            lv = None
        return mean, lv

    def distr_params(self, x):
        mean, lv = self.get_mean_lv(x)

        params = {
            'mean': mean,
            'logvar': lv,
        }
        return params

    @staticmethod
    def mean(params):
        return params['mean']

    @staticmethod
    def mode(params):
        return params['mean']

    @staticmethod
    def sample(params):
        # p = Normal(params['mean'], (params['logvar'] / 2).exp())
        # return p.rsample()
        return params['mean']

    def log_likelihood(self, x, params):
        if self.predict_logvar:
            logprob = log_normal(x, params['mean'], params['logvar'])
        else:
            logprob = -0.5 * (params['mean'] - x) ** 2
        return logprob


def log_normal(x, mean, logvar):
    """
    Log of the probability density of the values x untder the Normal
    distribution with parameters mean and logvar.
    :param x: tensor of points, with shape (batch, channels, dim1, dim2)
    :param mean: tensor with mean of distribution, shape
                 (batch, channels, dim1, dim2)
    :param logvar: tensor with log-variance of distribution, shape has to be
                   either scalar or broadcastable
    """

    var = torch.exp(logvar)
    log_prob = -0.5 * (((x - mean) ** 2) / var + logvar + torch.tensor(2 * math.pi).log())
    return log_prob
