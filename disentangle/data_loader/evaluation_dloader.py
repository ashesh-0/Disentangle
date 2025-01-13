"""
This data loader is designed just for test images. 
"""
from typing import List

import numpy as np

from disentangle.data_loader.patch_index_manager import GridIndexManager

"""
This data loader is designed just for test images. 
"""
from typing import List

import numpy as np

# from disentangle.data_loader.patch_index_manager import GridAlignement, GridIndexManager
"""
We would like to have a common logic to map between an index and location on the image.
We assume the data to be of shape N * H * W * C (C: channels, H,W: spatial dimensions, N: time/number of frames)
We assume the square patches.
The extra content on the right side will not be used( as shown below). 
.-----------.-.
|           | |
|           | |
|           | |
|           | |
.-----------.-.

"""
from tkinter import Grid

from disentangle.core.custom_enum import Enum


class GridAlignement(Enum):
    """
    A patch is formed by padding the grid with content. If the grids are 'Center' aligned, then padding is to done equally on all 4 sides.
    On the other hand, if grids are 'LeftTop' aligned, padding is to be done on the right and bottom end of the grid.
    In the former case, one needs (patch_size - grid_size)//2 amount of content on the right end of the frame. 
    In the latter case, one needs patch_size - grid_size amount of content on the right end of the frame. 
    """
    LeftTop = 0
    Center = 1


class GridIndexManager:

    def __init__(self, data_shape, grid_size, patch_size, grid_alignement) -> None:
        self._data_shape = data_shape
        self._default_grid_size = grid_size
        self.patch_size = patch_size
        self.N = self._data_shape[0]
        self._align = grid_alignement

    def get_data_shape(self):
        return self._data_shape

    def use_default_grid(self, grid_size):
        return grid_size is None or grid_size < 0

    def grid_rows(self, grid_size):
        if self._align == GridAlignement.LeftTop:
            extra_pixels = (self.patch_size - grid_size)
        elif self._align == GridAlignement.Center:
            # Center is exclusively used during evaluation. In this case, we use the padding to handle edge cases.
            # So, here, we will ideally like to cover all pixels and so extra_pixels is set to 0.
            # If there was no padding, then it should be set to (self.patch_size - grid_size) // 2
            extra_pixels = 0

        return ((self._data_shape[-3] - extra_pixels) // grid_size)

    def grid_cols(self, grid_size):
        if self._align == GridAlignement.LeftTop:
            extra_pixels = (self.patch_size - grid_size)
        elif self._align == GridAlignement.Center:
            extra_pixels = 0

        return ((self._data_shape[-2] - extra_pixels) // grid_size)

    def grid_count(self, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        return self.N * self.grid_rows(grid_size) * self.grid_cols(grid_size)

    def hwt_from_idx(self, index, grid_size=None):
        t = self.get_t(index)
        return (*self.get_deterministic_hw(index, grid_size=grid_size), t)

    def idx_from_hwt(self, h_start, w_start, t, grid_size=None):
        """
        Given h,w,t (where h,w constitutes the top left corner of the patch), it returns the corresponding index.
        """
        if grid_size is None:
            grid_size = self._default_grid_size

        nth_row = h_start // grid_size
        nth_col = w_start // grid_size

        index = self.grid_cols(grid_size) * nth_row + nth_col
        return index * self._data_shape[0] + t

    def get_t(self, index):
        return index % self.N

    def get_top_nbr_idx(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        ncols = self.grid_cols(grid_size)
        index -= ncols * self.N
        if index < 0:
            return None

        return index

    def get_bottom_nbr_idx(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        ncols = self.grid_cols(grid_size)
        index += ncols * self.N
        if index > self.grid_count(grid_size=grid_size):
            return None

        return index

    def get_left_nbr_idx(self, index, grid_size=None):
        if self.on_left_boundary(index, grid_size=grid_size):
            return None

        index -= self.N
        return index

    def get_right_nbr_idx(self, index, grid_size=None):
        if self.on_right_boundary(index, grid_size=grid_size):
            return None
        index += self.N
        return index

    def on_left_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        factor = index // self.N
        ncols = self.grid_cols(grid_size)

        left_boundary = (factor // ncols) != (factor - 1) // ncols
        return left_boundary

    def on_right_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        factor = index // self.N
        ncols = self.grid_cols(grid_size)

        right_boundary = (factor // ncols) != (factor + 1) // ncols
        return right_boundary

    def on_top_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        ncols = self.grid_cols(grid_size)
        return index < self.N * ncols

    def on_bottom_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        ncols = self.grid_cols(grid_size)
        return index + self.N * ncols > self.grid_count(grid_size=grid_size)

    def on_boundary(self, idx, grid_size=None):
        if self.on_left_boundary(idx, grid_size=grid_size):
            return True

        if self.on_right_boundary(idx, grid_size=grid_size):
            return True

        if self.on_top_boundary(idx, grid_size=grid_size):
            return True

        if self.on_bottom_boundary(idx, grid_size=grid_size):
            return True
        return False

    def get_deterministic_hw(self, index: int, grid_size=None):
        """
        Fixed starting position for the crop for the img with index `index`.
        """
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        # _, h, w, _ = self._data_shape
        # assert h == w
        factor = index // self.N
        ncols = self.grid_cols(grid_size)

        ith_row = factor // ncols
        jth_col = factor % ncols
        h_start = ith_row * grid_size
        w_start = jth_col * grid_size
        return h_start, w_start

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