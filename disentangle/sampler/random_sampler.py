import numpy as np
from torch.utils.data import Sampler

from disentangle.sampler.base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    """
    Randomly yields the two indices
    """
    def init(self):
        self.index_batches = []

        l1_range = self.label_idx_dict['1']
        l1_idx = np.random.choice(np.arange(l1_range[0], l1_range[1]), size=len(self._dset), replace=True)

        l2_range = self.label_idx_dict['2']
        l2_idx = np.random.choice(np.arange(l2_range[0], l2_range[1]), size=len(self._dset), replace=True)

        self.index_batches = list(zip(l1_idx, l2_idx))
