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

    def _crop_img(self, img: np.ndarray, h_start: int, w_start: int):
        if self._grid_alignment == GridAlignement.LeftTop:
            # In training, this is used.
            # NOTE: It is my opinion that if I just use self._crop_img_with_padding, it will work perfectly fine.
            # The only benefit this if else loop provides is that it makes it easier to see what happens during training.
            new_img = img[..., h_start:h_start + self._img_sz, w_start:w_start + self._img_sz]
            return new_img
        elif self._grid_alignment == GridAlignement.Center:
            # During evaluation, this is used. In this situation, we can have negative h_start, w_start. Or h_start +self._img_sz can be larger than frame
            # In these situations, we need some sort of padding. This is not needed  in the LeftTop alignement.
            return self._crop_img_with_padding(img, h_start, w_start)

    def get_begin_end_padding(self, start_pos, max_len):
        """
        The effect is that the image with size self._grid_sz is in the center of the patch with sufficient
        padding on all four sides so that the final patch size is self._img_sz.
        """
        pad_start = 0
        pad_end = 0
        if start_pos < 0:
            pad_start = -1 * start_pos

        pad_end = max(0, start_pos + self._img_sz - max_len)

        return pad_start, pad_end

    def on_boundary(self, cur_loc, frame_size):
        return cur_loc + self._img_sz > frame_size or cur_loc < 0

    def _crop_img_with_padding(self, img: np.ndarray, h_start: int, w_start: int):
        H, W = img.shape
        h_on_boundary = self.on_boundary(h_start, H)
        w_on_boundary = self.on_boundary(w_start, W)

        assert h_start < H
        assert w_start < W

        assert h_start + self._img_sz <= H or h_on_boundary
        assert w_start + self._img_sz <= W or w_on_boundary
        # max() is needed since h_start could be negative.
        new_img = img[max(0, h_start):h_start + self._img_sz, max(0, w_start):w_start + self._img_sz]
        padding = np.array([[0, 0], [0, 0]])

        if h_on_boundary:
            pad = self.get_begin_end_padding(h_start, H)
            padding[0] = pad
        if w_on_boundary:
            pad = self.get_begin_end_padding(w_start, W)
            padding[1] = pad

        if not np.all(padding == 0):
            new_img = np.pad(new_img, padding, mode='reflect')

        return new_img

    def get_data_shape(self):
        return self._data.shape

    def get_grid_size(self):
        return self._grid_sz

    def __getitem__(self, idx):
        img = self._data[self.idx_manager.get_t(idx),...,0]
        h_start, w_start = self.idx_manager.get_deterministic_hw(idx)
        
        if self._grid_alignment == GridAlignement.Center:
            h_start -= self.per_side_overlap_pixelcount()
            w_start -= self.per_side_overlap_pixelcount()
        
        crop = self._crop_img(img, h_start, w_start)
        # h_end = h_start + self._img_sz
        # w_end = w_start + self._img_sz
        # crop = img[h_start:h_end, w_start:w_end]
        # crop = self.pad(crop)
        return self._normalizer(crop)[None]