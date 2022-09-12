from disentangle.analysis.stitch_prediction import set_skip_boundary_pixels_mask, set_skip_central_pixels_mask, \
    _get_location, stitch_predictions, stitched_prediction_mask
import numpy as np


def test_skipping_boundaries():
    mask = np.full((10, 2, 8, 8), 1)
    extra_padding = 0
    hwt1 = (0, 0, 0)
    pred_h = 4
    pred_w = 4
    hwt2 = (pred_h, pred_w, 2)
    loc1 = _get_location(extra_padding, hwt1, pred_h, pred_w)
    loc2 = _get_location(extra_padding, hwt2, pred_h, pred_w)
    set_skip_boundary_pixels_mask(mask, loc1, 1)
    set_skip_boundary_pixels_mask(mask, loc2, 1)
    correct_mask = np.full((10, 2, 8, 8), 1)
    # boundary for hwt1
    correct_mask[0, :, 0, [0, 1, 2, 3]] = False
    correct_mask[0, :, 3, [0, 1, 2, 3]] = False
    correct_mask[0, :, [0, 1, 2, 3], 0] = False
    correct_mask[0, :, [0, 1, 2, 3], 3] = False

    # boundary for hwt2
    correct_mask[2, :, 4, [4, 5, 6, 7]] = False
    correct_mask[2, :, 7, [4, 5, 6, 7]] = False
    correct_mask[2, :, [4, 5, 6, 7], 4] = False
    correct_mask[2, :, [4, 5, 6, 7], 7] = False
    assert (mask == correct_mask).all()


def test_picking_boundaries():
    mask = np.full((10, 2, 8, 8), 1)
    extra_padding = 0
    hwt1 = (0, 0, 0)
    pred_h = 4
    pred_w = 4
    hwt2 = (pred_h, pred_w, 2)
    loc1 = _get_location(extra_padding, hwt1, pred_h, pred_w)
    loc2 = _get_location(extra_padding, hwt2, pred_h, pred_w)
    set_skip_central_pixels_mask(mask, loc1, 1)
    set_skip_central_pixels_mask(mask, loc2, 2)
    correct_mask = np.full((10, 2, 8, 8), 1)
    # boundary for hwt1
    correct_mask[0, :, 2, 2] = False
    # boundary for hwt2
    correct_mask[2, :, 5:7, 5:7] = False

    print(mask[hwt2[-1]])
    assert (mask == correct_mask).all()


class DummyDset:
    def __init__(self):
        self._data = np.zeros((100, 64, 64, 2))
        self.N = len(self._data)
        self._img_sz = 10
        self._img_sz_for_hw = 8

    def get_img_sz(self):
        return self._img_sz

    def set_img_sz(self, img_sz):
        self._img_sz = img_sz

    def get_t(self, index):
        return index % self.N

    def per_side_overlap_pixelcount(self):
        return (self._img_sz - self._img_sz_for_hw) // 2

    def hwt_from_idx(self, index):
        _, H, W, _ = self._data.shape
        t = self.get_t(index)
        return (*self._get_deterministic_hw(index, H, W), t)

    def _get_deterministic_hw(self, index: int, h: int, w: int):
        """
        Fixed starting position for the crop for the img with index `index`.
        """
        img_sz = self._img_sz_for_hw
        assert h == w
        factor = index // self.N
        nrows = h // img_sz

        ith_row = factor // nrows
        jth_col = factor % nrows
        h_start = ith_row * img_sz
        w_start = jth_col * img_sz
        pad = self.per_side_overlap_pixelcount()
        return h_start - pad, w_start - pad

    def __len__(self):
        return self.N * ((self._data.shape[-2] // self._img_sz_for_hw) ** 2)


def test_stitch_predictions():
    dset = DummyDset()
    h = w = dset._img_sz
    predictions = np.random.rand(len(dset), 2, h, w)
    output = stitch_predictions(predictions, dset)
    skip_boundary_pixel_count = 0
    skip_central_pixel_count = 0
    mask1 = stitched_prediction_mask(dset, (h, w), skip_boundary_pixel_count, skip_central_pixel_count)
    assert (mask1 == 1).all()

    skip_boundary_pixel_count = 2
    skip_central_pixel_count = 0
    mask2 = stitched_prediction_mask(dset, (h, w), skip_boundary_pixel_count, skip_central_pixel_count)

    skip_boundary_pixel_count = 0
    skip_central_pixel_count = 4
    mask3 = stitched_prediction_mask(dset, (h, w), skip_boundary_pixel_count, skip_central_pixel_count)

    assert ((mask2 + mask3) == 1).all()

    skip_boundary_pixel_count = 1
    skip_central_pixel_count = 2
    mask4 = stitched_prediction_mask(dset, (h, w), skip_boundary_pixel_count, skip_central_pixel_count)

    # import matplotlib.pyplot as plt;
    # plt.imshow(mask4[0, :, :, 0]);
    # plt.show()
