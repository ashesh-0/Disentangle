import numpy as np


def max_rolling1D(a, window):
    """
    Taken from https://stackoverflow.com/questions/52218596/rolling-maximum-with-numpy
    """
    assert len(a.shape) == 1
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return np.max(rolling, axis=1)


def max_rolling(data, window, axis):
    shape = data.shape
    output = np.zer


class EmptyPatchFetcher:
    """
    The idea is to fetch empty patches so that real content can be replaced with this. 
    """

    def __init__(self, idx_manager, patch_size, data_frames, max_val_threshold=None):
        self._frames = data_frames
        self._idx_manager = idx_manager
        self._max_val_threshold = max_val_threshold
        self._idx_list = []
        self._patch_size = patch_size
        self._grid_size = 1
        self.set_empty_idx()

        print(f'[{self.__class__.__name__}] MaxVal:{self._max_val_threshold}')

    def compute_max(self, window):
        """
        Rolling compute.
        """
        N, H, W = self._frames.shape
        randnum = -954321
        assert self._grid_size == 1
        max_data = np.zeros((N, H - window, W - window)) * randnum

        for h in range(H - window):
            for w in range(W - window):
                max_data[:, h, w] = self._frames[:, h:h + window, w:w + window].max()

        assert (max_data != 954321).any()
        return max_data

    def set_empty_idx(self):
        patch_size = None
        max_data = self.compute_max(patch_size)
        empty_loc = np.where(max_data >= 0, max_data < self._max_val_threshold)
        self._idx_list = []
        for n_idx, h_start, w_start in empty_loc:
            self._idx_list.append(self._idx_manager.idx_from_hwt(h_start, w_start, n_idx))

        assert len(self._idx_list) > 0

    def sample(self):
        return (np.random.choice(self._idx_list), self._grid_size)
