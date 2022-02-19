import numpy as np
from torch.utils.data import Sampler


class RandomSampler(Sampler):
    """
    Randomly yields the two indices
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

    def init(self):
        self.index_batches = []

        l1_range = self.label_idx_dict['1']
        l1_idx = np.random.choice(np.arange(l1_range[0], l1_range[1]), size=len(self._dset), replace=True)

        l2_range = self.label_idx_dict['2']
        l2_idx = np.random.choice(np.arange(l2_range[0], l2_range[1]), size=len(self._dset), replace=True)

        self.index_batches = list(zip(l1_idx, l2_idx))
