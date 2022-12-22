"""
Get overlapping patches from the dataset
"""
import numpy as np


def get_overlapping_dset(dset_class):
    """
    dset_class must have _crop_img function and _get_deterministic_hw functions.
    (and ofcourse used in the same way as they should be :D)
    """
    class OverlappingDset(dset_class):
        def __init__(self, *args, **kwargs):
            image_size_for_grid_centers = kwargs.pop('image_size_for_grid_centers')
            overlapping_padding_kwargs = kwargs.pop('overlapping_padding_kwargs')
            super().__init__(*args, **kwargs)
            self._img_sz_for_hw = image_size_for_grid_centers
            self._repeat_factor = (self._data.shape[-2] // self._img_sz_for_hw)**2
            # used for multiscale data loader.
            self.enable_padding_while_cropping = True
            assert self._img_sz >= self._img_sz_for_hw
            self._overlapping_padding_kwargs = overlapping_padding_kwargs

        def per_side_overlap_pixelcount(self):
            return (self._img_sz - self._img_sz_for_hw) // 2

        def get_grid_size(self):
            return self._img_sz_for_hw

        def get_begin_end_padding(self, start_pos, max_len):
            """
            This assumes for simplicity that image is square shaped.
            The effect is that the image with size self._img_sz_for_hw is in the center of the patch with sufficient
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

        def _get_deterministic_hw(self, index: int, h: int, w: int):
            # Note that this assert is needed. Without it, even larger index can be allowed entry and it
            # would not cause the issue anywhere.
            h_start, w_start = self.idx_manager.get_deterministic_hw(index, h, w, img_sz=self._img_sz_for_hw)
            pad = self.per_side_overlap_pixelcount()
            return h_start - pad, w_start - pad

        def _crop_img(self, img: np.ndarray, h_start: int, w_start: int):
            _, H, W = img.shape
            h_on_boundary = self.on_boundary(h_start, H)
            w_on_boundary = self.on_boundary(w_start, W)

            assert h_start < H
            assert w_start < W

            assert h_start + self._img_sz <= H or h_on_boundary
            assert w_start + self._img_sz <= W or w_on_boundary
            # max() is needed since h_start could be negative.
            new_img = img[..., max(0, h_start):h_start + self._img_sz, max(0, w_start):w_start + self._img_sz]
            padding = np.array([[0, 0], [0, 0], [0, 0]])
            if h_on_boundary:
                pad = self.get_begin_end_padding(h_start, H)
                padding[1] = pad
            if w_on_boundary:
                pad = self.get_begin_end_padding(w_start, W)
                padding[2] = pad

            if not np.all(padding == 0):
                new_img = np.pad(new_img, padding, **self._overlapping_padding_kwargs)

            return new_img

    return OverlappingDset
