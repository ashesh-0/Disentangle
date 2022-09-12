from disentangle.analysis.stitch_prediction import set_skip_boundary_pixels_mask, set_skip_central_pixels_mask, _get_location
import numpy as np

def test_skipping_boundaries():
    mask = np.full((10, 8, 8), 1)
    extra_padding = 0
    hwt1 = (0, 0, 0)
    pred_h = 4
    pred_w = 4
    hwt2 = (pred_h, pred_w, 2)
    loc1 = _get_location(extra_padding, hwt1, pred_h, pred_w)
    loc2 = _get_location(extra_padding, hwt2, pred_h, pred_w)
    set_skip_boundary_pixels_mask(mask, loc1, 1)
    set_skip_boundary_pixels_mask(mask, loc2, 1)
    correct_mask = np.full((10,8,8),1)
    # boundary for hwt1
    correct_mask[0,0,[0,1,2,3]] = False
    correct_mask[0, 3, [0, 1, 2, 3]] = False
    correct_mask[0,[0,1,2,3],0] = False
    correct_mask[0,[0,1,2,3],3] = False

    # boundary for hwt2
    correct_mask[2, 4, [4, 5, 6, 7]] = False
    correct_mask[2, 7, [4, 5, 6, 7]] = False
    correct_mask[2, [4, 5, 6, 7], 4] = False
    correct_mask[2, [4, 5, 6, 7], 7] = False
    assert (mask == correct_mask).all()

def test_picking_boundaries():
    mask = np.full((10, 8, 8), 1)
    extra_padding = 0
    hwt1 = (0, 0, 0)
    pred_h = 4
    pred_w = 4
    hwt2 = (pred_h, pred_w, 2)
    loc1 = _get_location(extra_padding, hwt1, pred_h, pred_w)
    loc2 = _get_location(extra_padding, hwt2, pred_h, pred_w)
    set_skip_central_pixels_mask(mask, loc1, 1)
    set_skip_central_pixels_mask(mask, loc2, 2)
    correct_mask = np.full((10,8,8),1)
    # boundary for hwt1
    correct_mask[0,2,2] = False
    # boundary for hwt2
    correct_mask[2,5:7,5:7] = False

    print(mask[hwt2[-1]])
    assert (mask == correct_mask).all()
