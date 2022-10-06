from disentangle.core.stable_exp import StableExponential
import torch


class StableLogVar:
    def __init__(self, logvar, enable_stable=True):
        self._lv = logvar
        self._enable_stable = enable_stable

    def get(self):
        if self._enable_stable is False:
            return self._lv

        return torch.log(self.get_var())

    def get_var(self):
        if self._enable_stable is False:
            return torch.exp(self._lv)
        return StableExponential(self._lv).exp()

    def get_std(self):
        return torch.sqrt(self.get_var())


class StableMean:
    def __init__(self, mean):
        self._mean = mean

    def get(self):
        return self._mean
