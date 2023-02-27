import numpy as np
from torch.utils.data import Sampler


class LevelIndexIterator:

    def __init__(self, index_list) -> None:
        self._index_list = index_list
        self._N = len(self._index_list)
        self._cur_position = 0

    def next(self):
        output_pos = self._cur_position
        self._cur_position += 1
        self._cur_position = self._cur_position % self._N
        return self._index_list[output_pos]

    def next_k(self, N):
        return [self.next() for _ in range(N)]


class ContrastiveSamplerValSet(Sampler):

    def __init__(self, dataset, grid_size, batch_size, fixed_alpha=-1) -> None:
        super().__init__(dataset)
        # In validation, we just look at the cases which we'll find in the test case. alpha=0.5 is that case. This corresponds to the -1 class.
        self._alpha = fixed_alpha
        self._N = len(dataset)
        self._batch_N = batch_size
        self._grid_size = grid_size

    def __iter__(self):
        num_batches = int(np.ceil(self._N / self._batch_N))
        for batch_idx in range(num_batches):
            # 4 channels: ch1_idx, ch2_idx, grid_size, alpha_idx
            batch_data_idx = np.ones((self._batch_N, 4), dtype=np.int32) * self.INVALID
            batch_data_idx[:, 0] = np.arange(batch_idx * self._batch_N, (batch_idx + 1) * self._batch_N)
            batch_data_idx[:, 1] = batch_data_idx[:, 0]
            batch_data_idx[:, 2] = self._grid_size
            batch_data_idx[:, 3] = self._alpha
            yield batch_data_idx


class ContrastiveSampler(Sampler):
    INVALID = -955

    def __init__(self, dataset, data_size, ch1_alpha_interval_count, batch_size) -> None:
        super().__init__(dataset)
        self._dset = dataset
        self._N = data_size
        self._alpha_class_N = ch1_alpha_interval_count
        self._batch_N = batch_size
        assert batch_size % 2 == 0
        # We'll be using grid_size of 1, this allows us to pick from any random location in the frame. However,
        # as far as one epoch is concerned, we'll use data_size. So, values in self.idx will be much larger than
        # self._N
        self._grid_size = 1
        self.idx = np.arange(self._dset.idx_manager.grid_count(grid_size=self._grid_size))
        self.batches_idx_list = None
        self.level_iters = None
        print(f'[{self.__class__.__name__}] Alpha class count:{self._alpha_class_N}')

    def __iter__(self):
        """
        3 attributes are present. 
        32
        16 pairs => alpha is same. 
        0-1
        2-3
        4-5
        ...
        16 pairs => ch1 is same. 
        1-2
        3-4
        5-6
        7-8
        ...
        31-0
        15 pairs => ch2 is same.
        0-3
        2-5
        ...
        """
        self.init()
        for one_batch_idx in self.batches_idx_list:
            alpha_idx_list, ch1_idx_list, ch2_idx_list = one_batch_idx

            # 4 channels: ch1_idx, ch2_idx, grid_size, alpha_idx
            batch_data_idx = np.ones((self._batch_N, 4), dtype=np.int32) * self.INVALID
            # grid size will always be 1.
            batch_data_idx[:, 2] = self._grid_size

            # Set alpha indices appropriately.
            for idx in range(0, self._batch_N // 2):
                batch_data_idx[2 * idx, 3] = alpha_idx_list[idx]
                batch_data_idx[2 * idx + 1, 3] = alpha_idx_list[idx]

            # Set ch1 indices
            for idx in range(0, self._batch_N // 2 - 1):
                batch_data_idx[2 * idx + 1, 0] = ch1_idx_list[idx]
                batch_data_idx[2 * idx + 2, 0] = ch1_idx_list[idx]

            batch_data_idx[0, 0] = ch1_idx_list[-1]
            batch_data_idx[-1, 0] = ch1_idx_list[-1]

            # Set ch2 indices
            for idx in range(0, self._batch_N // 2 - 1):
                batch_data_idx[2 * idx, 1] = ch2_idx_list[idx]
                batch_data_idx[2 * idx + 3, 1] = ch2_idx_list[idx]

            batch_data_idx[1, 1] = ch2_idx_list[-1]
            batch_data_idx[-2, 1] = ch2_idx_list[-1]

            assert (batch_data_idx == self.INVALID).any() == False
            yield batch_data_idx

    def init(self):
        self.batches_idx_list = []
        total_size = self._N * self._alpha_class_N
        num_batches = int(np.ceil(total_size / self._batch_N))
        idx = self.idx.copy()
        np.random.shuffle(idx)
        self.ch1_idx_iterator = LevelIndexIterator(idx)

        idx = self.idx.copy()
        np.random.shuffle(idx)
        self.ch2_idx_iterator = LevelIndexIterator(idx)

        for _ in range(num_batches):
            if self._alpha_class_N >= self._batch_N / 2:
                alpha_idx = np.random.choice(np.arange(self._alpha_class_N), size=self._batch_N // 2, replace=False)
            else:
                alpha_idx = np.random.choice(np.arange(self._alpha_class_N), size=self._batch_N // 2, replace=True)
                while len(np.unique(alpha_idx)) == 1:
                    alpha_idx = np.random.choice(np.arange(self._alpha_class_N), size=self._batch_N // 2, replace=True)

            ch1_idx = self.ch1_idx_iterator.next_k(self._batch_N // 2)
            ch2_idx = self.ch2_idx_iterator.next_k(self._batch_N // 2)
            self.batches_idx_list.append((alpha_idx, ch1_idx, ch2_idx))

        self.idx = np.arange(self._N)
        np.random.shuffle(self.idx)


if __name__ == '__main__':
    ch1_alpha_interval_count = 30
    data_size = 1000
    batch_size = 32
    sampler = ContrastiveSampler(None, data_size, ch1_alpha_interval_count, batch_size)
    for batch in sampler:
        break
