import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from disentangle.analysis.lvae_utils import get_img_from_forward_output
from disentangle.data_loader.overlapping_dloader import get_overlapping_dset


def get_predictions(model, dset, batch_size, mmse_count=1, num_workers=4):
    dloader = DataLoader(dset,
                         pin_memory=False,
                         num_workers=num_workers,
                         shuffle=False,
                         batch_size=batch_size)

    predictions = []
    losses = []
    logvar_arr = []
    with torch.no_grad():
        for inp, tar in tqdm(dloader):
            inp = inp.cuda()
            x_normalized = model.normalize_input(inp)
            tar = tar.cuda()
            tar_normalized = model.normalize_target(tar)

            recon_img_list = []
            for _ in range(mmse_count):
                recon_normalized, td_data = model(x_normalized)
                rec_loss, imgs = model.get_reconstruction_loss(recon_normalized, tar_normalized,
                                                               return_predicted_img=True)
                recon_img_list.append(imgs.cpu()[None])

            mmse_imgs = torch.mean(torch.cat(recon_img_list, dim=0), dim=0)

            q_dic = model.likelihood.distr_params(recon_normalized)
            logvar_arr.append(q_dic['logvar'].cpu().numpy())

            losses.append(rec_loss['loss'].cpu().numpy())
            predictions.append(mmse_imgs.cpu().numpy())
    return np.concatenate(predictions, axis=0), np.array(losses), np.concatenate(logvar_arr)


class PatchLocation:
    """
    Encapsulates t_idx and spatial location.
    """

    def __init__(self, t_idx, h_idx_range, w_idx_range):
        self.t = t_idx
        self.h_start, self.h_end = h_idx_range
        self.w_start, self.w_end = w_idx_range


def get_hwt_from_idx(dset, dset_input_idx, pred_h, pred_w):
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
    h_start, w_start, t_idx = dset.hwt_from_idx(dset_input_idx)
    h_end = h_start + pred_h

    h_start += extra_padding
    h_end -= extra_padding

    w_end = w_start + pred_w
    w_start += extra_padding
    w_end -= extra_padding
    return PatchLocation(t_idx, (h_start, h_end), (w_start, w_end))


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
        loc = get_hwt_from_idx(dset, dset_input_idx, hN, wN)
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
        loc = get_hwt_from_idx(dset, dset_input_idx, predictions.shape[-2], predictions.shape[-1])
        output[loc.t, loc.h_start:loc.h_end, loc.w_start:loc.w_end, 0] = remove_pad(predictions[dset_input_idx, 0])
        output[loc.t, loc.h_start:loc.h_end, loc.w_start:loc.w_end, 1] = remove_pad(predictions[dset_input_idx, 1])

    return output
