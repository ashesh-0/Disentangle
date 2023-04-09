import numpy as np


class EmptyPatchFetcher:
    """
    The idea is to fetch empty patches so that real content can be replaced with this. 
    """

    def __init__(self, idx_manager, data_frames, max_val_threshold=None):
        self._frames = data_frames
        self._idx_manager = idx_manager
        self._max_val_threshold = max_val_threshold
        self._idx_list = []
        print(f'[{self.__class__.__name__}] MaxVal:{self._max_val_threshold}')

    def compute(self):
        """
        Rolling compute.
        """
        pass

        assert len(self._idx_list) > 0

    def sample(self):
        return np.random.choice(self._idx_list)
