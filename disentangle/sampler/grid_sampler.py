"""
The idea is one can feed the grid_size along with index.
"""
import numpy as np

from disentangle.data_loader.patch_index_manager import GridAlignement
from disentangle.sampler.base_sampler import BaseSampler


class GridSampler(BaseSampler):
    """
    Randomly yields an index and an associated grid size.
    if full_coverage_randomized is set to true, then it divides the image into overlapping tiles
    and it returns indices for all the tiles in a random order.
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 randomized=True,
                 full_coverage_randomized=False,
                 full_coverage_overlap=None,
                 patch_size=None,
                 grid_size=1) -> None:
        super().__init__(dataset, batch_size, grid_size)
        self._randomized = randomized
        self._randomized_full_coverage = full_coverage_randomized
        self._overlap = self._patch_size = None
        assert self._randomized_full_coverage is False or self._randomized is False, "Cannot have both randomized and full coverage randomized"
        if self._randomized_full_coverage:
            dataset.idx_manager.get_grid_alignment() == GridAlignement.Center
            self._overlap = full_coverage_overlap
            self._patch_size = patch_size
            assert self._overlap is not None, "Overlap cannot be None"
            assert self._patch_size is not None, "Patch size cannot be None"
        print(
            f'[{self.__class__.__name__}] randomized: {self._randomized}, full_coverage_randomized: {self._randomized_full_coverage} \
              full_coverage_overlap: {self._overlap}, patch_size: {self._patch_size} grid_size: {self._grid_size}')

    def get_right_nbr_idx(self, last_idx):
        tilesize = self._patch_size - 2 * self._overlap
        h, w, t = self._dset.idx_manager.hwt_from_idx(last_idx, grid_size=self._grid_size)
        if w + 2 * tilesize > self._dset.idx_manager.get_numcols():
            return None
        else:
            return self._dset.idx_manager.idx_from_hwt(h, w + tilesize, t, grid_size=self._grid_size)

    def get_left_nbr_idx(self, last_idx):
        tilesize = self._patch_size - 2 * self._overlap
        h, w, t = self._dset.idx_manager.hwt_from_idx(last_idx, grid_size=self._grid_size)
        if w - tilesize < 0:
            return None
        else:
            return self._dset.idx_manager.idx_from_hwt(h, w - tilesize, t, grid_size=self._grid_size)

    def get_bottom_nbr_idx(self, last_idx):
        tilesize = self._patch_size - 2 * self._overlap
        h, w, t = self._dset.idx_manager.hwt_from_idx(last_idx, grid_size=self._grid_size)
        if h + 2 * tilesize > self._dset.idx_manager.get_numrows():
            return None
        else:
            return self._dset.idx_manager.idx_from_hwt(h + tilesize, w, t, grid_size=self._grid_size)

    def get_indices_for_one_t(self, start_idx):
        """
        
        """
        indices = [start_idx]
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
        offsetrow = np.random.randint(0, self._overlap // 2)
        offsetcol = np.random.randint(0, self._overlap // 2)
        self.index_batches = []
        for t in range(self._dset.idx_manager.N):
            idx_start = self._dset.idx_manager.idx_from_hwt(offsetrow, offsetcol, t, grid_size=self._grid_size)
            self.index_batches += self.get_indices_for_one_t(idx_start)

        grid_size = np.array([self._grid_size] * len(self.index_batches))
        self.index_batches = list(zip(self.index_batches, grid_size))
        np.random.shuffle(self.index_batches)

    def init(self):
        if self._randomized_full_coverage:
            self.init_full_coverage()
            return

        self.index_batches = []
        if self._randomized:
            idx = np.random.randint(low=0, high=self.idx_max, size=len(self._dset))
        else:
            idx = np.arange(0, len(self._dset))

        grid_size = np.array([self._grid_size] * len(idx))
        self.index_batches = list(zip(idx, grid_size))
        np.random.shuffle(self.index_batches)


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
    overlap = 16
    patch_size = config.data.image_size
    sampler = GridSampler(dset,
                          16,
                          randomized=False,
                          full_coverage_randomized=True,
                          full_coverage_overlap=overlap,
                          patch_size=patch_size,
                          grid_size=1)

    train_dloader = DataLoader(dset, pin_memory=False, batch_sampler=sampler, num_workers=0)

    sampler.init()

    from tqdm import tqdm
    stitched_data = np.zeros_like(dset._data)
    for batch in tqdm(train_dloader):
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
                if skiphs == 0 or skipws == 0:
                    print(h_start, w_start, t)
                stitched_data[t, h_start + skiphs:h_start + patch_size - overlap,
                              w_start + skipws:w_start + patch_size - overlap, ch_idx] = tar[i, ch_idx, skiphs:-overlap,
                                                                                             skipws:-overlap].numpy()

    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(6, 3), ncols=2)
    ax[0].imshow(dset._data[0, :32, :32, 0])
    ax[1].imshow(stitched_data[0, :32, :32, 0])
