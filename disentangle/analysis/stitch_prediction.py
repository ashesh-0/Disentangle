import numpy as np

class PatchLocation:
    """
    Encapsulates t_idx and spatial location.
    """

    def __init__(self, h_idx_range, w_idx_range, t_idx):
        self.t = t_idx
        self.h_start, self.h_end = h_idx_range
        self.w_start, self.w_end = w_idx_range


def _get_location(extra_padding,hwt, pred_h, pred_w):
    h_start, w_start, t_idx = hwt
    h_end = h_start + pred_h

    h_start += extra_padding
    h_end -= extra_padding

    w_end = w_start + pred_w
    w_start += extra_padding
    w_end -= extra_padding
    return PatchLocation((h_start, h_end), (w_start, w_end),t_idx)


def get_location_from_idx(dset, dset_input_idx, pred_h, pred_w):
    """
    For a given idx of the dataset, it returns where exactly in the dataset, does this prediction lies.
    Which time frame, which spatial location (h_start, h_end, w_start,w_end)
    Args:
        dset:
        dset_input_idx:
        pred_h:
        pred_w:

    Returns:

    """
    extra_padding = dset.per_side_overlap_pixelcount()
    htw = dset.hwt_from_idx(dset_input_idx)
    return _get_location(extra_padding,htw,pred_h,pred_w)

def set_skip_boundary_pixels_mask(mask, loc, skip_count):
    if skip_count == 0:
        return mask
    assert skip_count > 0
    assert loc.h_end - skip_count >= 0
    assert loc.w_end - skip_count >= 0
    mask[loc.t, loc.h_start:loc.h_start + skip_count, loc.w_start:loc.w_end] = False
    mask[loc.t, loc.h_end - skip_count:loc.h_end, loc.w_start:loc.w_end] = False
    mask[loc.t, loc.h_start:loc.h_end, loc.w_start:loc.w_start + skip_count] = False
    mask[loc.t, loc.h_start:loc.h_end, loc.w_end - skip_count:loc.w_end] = False


def set_skip_central_pixels_mask(mask, loc, skip_count):
    if skip_count == 0:
        return mask
    assert skip_count > 0
    h_mid = (loc.h_start + loc.h_end) // 2
    w_mid = (loc.w_start + loc.w_end) // 2
    l_skip = skip_count // 2
    r_skip = skip_count - l_skip
    mask[loc.t, h_mid - l_skip:h_mid + r_skip, w_mid - l_skip:w_mid + r_skip] = False


def stitched_prediction_mask(dset, padded_patch_shape, skip_boundary_pixel_count, skip_central_pixel_count):
    """
    Returns the boolean matrix. It will be 0 if it lies either in skipped boundaries or skipped central pixels
    Args:
        dset:
        padded_patch_shape:
        skip_boundary_pixel_count:
        skip_central_pixel_count:

    Returns:
    """
    mask = np.full(dset._data.shape, True)
    hN, wN = padded_patch_shape
    for dset_input_idx in range(len(dset)):
        loc = get_location_from_idx(dset, dset_input_idx, hN, wN)
        set_skip_boundary_pixels_mask(mask, loc, skip_boundary_pixel_count)
        set_skip_central_pixels_mask(mask, loc, skip_central_pixel_count)
    return mask


def stitch_predictions(predictions, dset):
    extra_padding = dset.per_side_overlap_pixelcount()
    output = np.zeros_like(dset._data)

    def remove_pad(pred):
        if extra_padding > 0:
            return pred[extra_padding:-extra_padding, extra_padding:-extra_padding]
        return pred

    for dset_input_idx in range(predictions.shape[0]):
        loc = get_location_from_idx(dset, dset_input_idx, predictions.shape[-2], predictions.shape[-1])
        output[loc.t, loc.h_start:loc.h_end, loc.w_start:loc.w_end, 0] = remove_pad(predictions[dset_input_idx, 0])
        output[loc.t, loc.h_start:loc.h_end, loc.w_start:loc.w_end, 1] = remove_pad(predictions[dset_input_idx, 1])

    return output

if __name__ == '__main__':
    extra_padding = 0
    hwt1 = (0,0,0)
    pred_h = 4
    pred_w = 4
    hwt2  = (pred_h,pred_w,2)
    loc1 = _get_location(extra_padding, hwt1, pred_h, pred_w)
    loc2 = _get_location(extra_padding, hwt2, pred_h, pred_w)
    mask = np.full((10,8,8),1)
    set_skip_boundary_pixels_mask(mask, loc1, 1)
    set_skip_boundary_pixels_mask(mask, loc2, 1)
    print(mask[hwt1[-1]])
    print(mask[hwt2[-1]])