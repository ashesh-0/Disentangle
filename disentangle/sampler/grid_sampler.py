"""
The idea is one can feed the grid_size along with index.
"""
import numpy as np

from disentangle.sampler.base_sampler import BaseSampler


class GridSampler(BaseSampler):
    """
    Randomly yields an index and an associated grid size.
    """
    def __init__(self, dataset, batch_size, randomized=True, grid_size=1) -> None:
        super().__init__(dataset, batch_size, grid_size)
        self._randomized =randomized

    def init(self):
        self.index_batches = []
        if self._randomized:
            idx = np.random.randint(low=0, high=self.idx_max, size=len(self._dset))
        else:
            idx = np.arange(0,len(self._dset))

        grid_size = np.array([self._grid_size] * len(idx))
        self.index_batches = list(zip(idx, grid_size))
