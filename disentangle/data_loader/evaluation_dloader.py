"""
This data loader is designed just for test images. 
"""
from typing import List

import numpy as np

from disentangle.data_loader.patch_index_manager import GridAlignement, GridIndexManager


class EvaluationDloader:
    def __init__(self, image: np.ndarray, normalizer_fn, pre_processor_fn, image_size:int , grid_size:int, grid_alignment:GridAlignement):
        assert len(image.shape) == 3, "Image should be 2D"
        # N x 1 x H x W
        self._data = image[..., np.newaxis]
        self._data = pre_processor_fn(self._data)

        self._normalizer = normalizer_fn
        self._img_sz = image_size
        self._grid_sz = grid_size
        self._grid_alignment = grid_alignment
        self.idx_manager = GridIndexManager(self._data.shape, self._grid_sz, self._img_sz, self._grid_alignment)
    
    def __len__(self):
        return len(self._data) * self.idx_manager.grid_rows(self._grid_sz) * self.idx_manager.grid_cols(self._grid_sz)

    def per_side_overlap_pixelcount(self):
        return (self._img_sz - self._grid_sz) // 2

    def pad(self, inp):
        sz = self._img_sz
        crop = inp
        if crop.shape[0] == sz and crop.shape[1] == sz:
            return inp
            
        padding = np.array([[0, 0], [0, 0]])
        padding[0] = [0, sz - crop.shape[0]]
        padding[1] = [0, sz - crop.shape[1]]
        assert np.all(padding >=0)
        new_crop = np.pad(crop, padding, mode='reflect')
        return new_crop

    def get_data_shape(self):
        return self._data.shape

    def get_grid_size(self):
        return self._grid_sz

    def __getitem__(self, idx):
        img = self._data[self.idx_manager.get_t(idx),...,0]
        h_start, w_start = self.idx_manager.get_deterministic_hw(idx)
        h_end = h_start + self._img_sz
        w_end = w_start + self._img_sz
        crop = img[h_start:h_end, w_start:w_end]
        crop = self.pad(crop)
        return self._normalizer(crop)[None]