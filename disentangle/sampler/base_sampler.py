import numpy as np
from torch.utils.data import Sampler


class BaseSampler(Sampler):
    """
    Base sampler for the class which yields wo indices
    """
    def __init__(self, dataset, batch_size) -> None:
        super().__init__(dataset)
        self._dset = dataset
        self.label_idx_dict = self._dset.get_label_idx_range()
        self.l1_N = self.label_idx_dict['1'][1] - self.label_idx_dict['1'][0]
        self.l2_N = self.label_idx_dict['2'][1] - self.label_idx_dict['2'][0]

        self._batch_size = batch_size
        self.idx = None
        self.index_batches = None
        print(f'[{self.__class__.__name__}] ')

    def __iter__(self):
        self.init()
        start_idx = 0
        for _ in range(len(self.index_batches) // self._batch_size):
            yield self.index_batches[start_idx:start_idx + self._batch_size].copy()
            start_idx += self._batch_size
