import numpy as np
from tqdm import tqdm

from disentangle.sampler.grid_sampler import GridSampler


class DualArrays:
    """
    We want to have two arrays containing consequitive indices. First array will store X and second array will store @
    X @ X @ X @
    @ X @ X @ X
    X @ X @ X @
    """

    def __init__(self) -> None:
        self.arr0 = []
        self.arr1 = []
        self.last_updated = -1

    def append(self, value):
        if self.last_updated == 0:
            self.arr1.append(value)
            self.last_updated = 1
        else:
            self.arr0.append(value)
            self.last_updated = 0


class AlternateGridSampler(GridSampler):

    def __init__(self,
                 dataset,
                 batch_size,
                 randomized=True,
                 full_coverage_randomized=False,
                 full_coverage_overlap=None,
                 patch_size=None,
                 grid_size=1):
        super().__init__(dataset,
                         batch_size,
                         randomized=randomized,
                         full_coverage_randomized=full_coverage_randomized,
                         full_coverage_overlap=full_coverage_overlap,
                         patch_size=patch_size,
                         grid_size=grid_size)

        self._state = 0
        self._offsetrow = 0
        self._offsetcol = 0

    def get_indices_for_one_t(self, start_idx):
        """
        
        """
        indices = DualArrays()
        indices.append(start_idx)
        towards_right = True
        last_idx = start_idx
        gridsize = self._patch_size - 2 * self._overlap
        nrows = self._dset._data.shape[1] // gridsize
        ncols = self._dset._data.shape[2] // gridsize
        for _ in range(self._dset.idx_manager.N * nrows * ncols):

            if towards_right:
                next_idx = self.get_right_nbr_idx(last_idx)
                if next_idx is None:
                    towards_right = False
                    last_idx = self.get_bottom_nbr_idx(last_idx)
                    if last_idx is not None:
                        indices.append(last_idx)
                    else:
                        break
                else:
                    indices.append(next_idx)
                    last_idx = next_idx
            else:
                next_idx = self.get_left_nbr_idx(last_idx)
                if next_idx is None:
                    towards_right = True
                    last_idx = self.get_bottom_nbr_idx(last_idx)
                    if last_idx is not None:
                        indices.append(last_idx)
                    else:
                        break
                else:
                    indices.append(next_idx)
                    last_idx = next_idx
        return indices

    def init_full_coverage(self):
        if self._state == 0:
            grid_size = self._patch_size - 2 * self._overlap
            self._offsetrow = np.random.randint(0, grid_size // 2)
            self._offsetcol = np.random.randint(0, grid_size // 2)

        self.index_batches = []
        for t in range(self._dset.idx_manager.N):
            idx_start = self._dset.idx_manager.idx_from_hwt(self._offsetrow,
                                                            self._offsetcol,
                                                            t,
                                                            grid_size=self._grid_size)
            all_indices = self.get_indices_for_one_t(idx_start)
            if self._state == 0:
                self.index_batches += all_indices.arr0
            else:
                self.index_batches += all_indices.arr1

        grid_size = np.array([self._grid_size] * len(self.index_batches))
        self.index_batches = list(zip(self.index_batches, grid_size))
        np.random.shuffle(self.index_batches)
        self._state = 1 - self._state

    def __iter__(self):
        self.init()
        start_idx = 0
        num_batches = int(np.ceil(len(self.index_batches) / self._batch_size))
        for _ in range(num_batches):
            batch_indices = self.index_batches[start_idx:start_idx + self._batch_size].copy()
            yield batch_indices
            start_idx += len(batch_indices)


if __name__ == '__main__':
    from disentangle.configs.autoregressive_config import get_config
    from disentangle.data_loader.multi_channel_determ_tiff_dloader import (DataSplitType, GridAlignement,
                                                                           MultiChDeterministicTiffDloader)

    config = get_config()
    with config.unlocked():
        config.data.multiscale_lowres_count = None

    padding_kwargs = {
        'mode': config.data.get('padding_mode', 'constant'),
    }

    if padding_kwargs['mode'] == 'constant':
        padding_kwargs['constant_values'] = config.data.get('padding_value', 0)

    dset = MultiChDeterministicTiffDloader(config.data,
                                           '/group/jug/ashesh/data/microscopy/OptiMEM100x014_small.tif',
                                           DataSplitType.Train,
                                           val_fraction=config.training.val_fraction,
                                           test_fraction=config.training.test_fraction,
                                           normalized_input=config.data.normalized_input,
                                           enable_rotation_aug=False,
                                           enable_random_cropping=config.data.deterministic_grid is False,
                                           use_one_mu_std=config.data.use_one_mu_std,
                                           allow_generation=False,
                                           max_val=None,
                                           grid_alignment=GridAlignement.Center,
                                           overlapping_padding_kwargs=padding_kwargs)

    mean, std = dset.compute_mean_std()
    dset.set_mean_std(mean, std)

    from torch.utils.data import DataLoader
    overlap = 8
    patch_size = config.data.image_size
    sampler = AlternateGridSampler(dset,
                                   14,
                                   randomized=False,
                                   full_coverage_randomized=True,
                                   full_coverage_overlap=overlap,
                                   patch_size=patch_size,
                                   grid_size=1)

    train_dloader = DataLoader(dset, pin_memory=False, batch_sampler=sampler, num_workers=0)

    # sampler.init()

    from tqdm import tqdm
    stitched_data = np.zeros_like(dset._data)
    for batch_idx, batch in tqdm(enumerate(train_dloader)):
        inp, tar, indices, grid_sizes = batch
        for i in (range(len(inp))):
            h, w, t = dset.idx_manager.hwt_from_idx(indices[i], grid_size=grid_sizes[i])
            # h_start = h - overlap
            # w_start = w - overlap
            h_start = h
            w_start = w

            if h_start < 0 or w_start < 0:
                continue
            elif h_start + patch_size > stitched_data.shape[1] or w_start + patch_size > stitched_data.shape[2]:
                continue
            for ch_idx in range(stitched_data.shape[-1]):
                skiphs = overlap if h_start >= overlap else max(0, 0 - h_start)
                skipws = overlap if w_start >= overlap else max(0, 0 - w_start)
                # if skiphs == 0 or skipws == 0:
                #     print(h_start, w_start, t)
                stitched_data[t, h_start + skiphs:h_start + patch_size - overlap,
                              w_start + skipws:w_start + patch_size - overlap, ch_idx] = tar[i, ch_idx, skiphs:-overlap,
                                                                                             skipws:-overlap].numpy()

    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(6, 3), ncols=2)
    ax[0].imshow(dset._data[0, ..., 0])
    ax[1].imshow(stitched_data[0, ..., 0])
