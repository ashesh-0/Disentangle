"""
Coverage plot: 
Steps: 
For every patch (or full frame, or every pixel):
    1. Predict multiple samples.
    2. Predict MMSE sample.
    3. Compute the MSE between the predicted sample and GT. => create a histogram of these values. 
    4. Predict the MSE between MMSE sample and the GT.  
    5. find the quantile of MSE (MMSE vs GT) in the histogram.
    6. repeat this for all patches.
    7. what should come out is that: in 10% of the cases, the quantile of the MMSE sample is 10%. 
"""

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def divide_into_smaller_patches(np_array, elem_size=10):
    """
    np_array: Batch X C x H x W
    """
    batch_size, C, H, W = np_array.shape
    cropH = H % elem_size
    cropW = W % elem_size
    # crop center
    if cropH != 0:
        np_array = np_array[..., cropH//2:-cropH//2,:]
    if cropW != 0:
        np_array = np_array[..., cropW//2:-cropW//2]

    nH = H // elem_size
    nW = W // elem_size
    np_array = np_array.reshape(batch_size, C, nH, elem_size, nW, elem_size)
    np_array = np.transpose(np_array, axes=(0, 2, 4, 1, 3, 5))
    np_array = np_array.reshape(-1, C, elem_size, elem_size)
    return np_array

def how_many_lie_in_k_quantiles(calibration_coverage_data, k):
    assert k >= 0 and k <= 100
    return np.mean(calibration_coverage_data <= k) * 100


def compute_for_one_batch(patch_predictions, gt, elem_size = 10, calib_stats_dict=None, background_patch_detection_func=None, mmse_sample_for_error=None):
    """
    mmse_sample_for_error: Batch X C x H x W. Sometimes, MMSE sample we want to compute the error is not the mmse of the patch predictions.
    This can be the case when using a two forward passes. The first one will yield a better mmse sample, but the second one will yield a better variance.
    """
    masks = []
    if background_patch_detection_func is not None:
        for ch_idx in range(gt.shape[1]):
            bkg_mask = background_patch_detection_func(gt[:, ch_idx], ch_idx)
            masks.append(bkg_mask)

    all_var = []
    all_err = []
    for ch_idx in range(gt.shape[1]):
        if calib_stats_dict is not None:
            var_factor = calib_stats_dict[ch_idx]['scalar']
            var_offset = calib_stats_dict[ch_idx]['offset']
        else:
            var_factor = 1
            var_offset = 0
        
        if elem_size is not None:
            n_entries = int((patch_predictions.shape[-1]/elem_size) * (patch_predictions.shape[-2]/elem_size))
            all_var_one_ch = []
            all_err_one_ch = []
            for _ in range(n_entries):
                h = np.random.randint(0, patch_predictions.shape[-2] - elem_size)
                w = np.random.randint(0, patch_predictions.shape[-1] - elem_size)
                mmse_sample_err = mmse_sample_for_error[:, ch_idx,h:h+elem_size,w:w+elem_size] if mmse_sample_for_error is not None else None
                var, err = compute_for_one_channel(patch_predictions[:, :, ch_idx,h:h+elem_size,w:w+elem_size], 
                                              gt[:, ch_idx,h:h+elem_size,w:w+elem_size], 
                                              var_factor=var_factor, var_offset=var_offset,
                                              mmse_sample_for_error=mmse_sample_err)
                if len(masks) > 0:
                    var[masks[ch_idx]] = np.nan
                    err[masks[ch_idx]] = np.nan
                
                all_var_one_ch.append(var)
                all_err_one_ch.append(err)
            all_var.append(np.concatenate(all_var_one_ch, axis=0))
            all_err.append(np.concatenate(all_err_one_ch, axis=0))
        else:
            mmse_sample_err = mmse_sample_for_error[:, ch_idx] if mmse_sample_for_error is not None else None
            var, err = compute_for_one_channel(patch_predictions[:, :, ch_idx], gt[:, ch_idx], var_factor=var_factor, var_offset=var_offset,
                                               mmse_sample_for_error=mmse_sample_err)
            if len(masks) > 0:
                var[masks[ch_idx]] = np.nan
                err[masks[ch_idx]] = np.nan
            all_var.append(var)
            all_err.append(err)

    return np.stack(all_var, axis=1), np.stack(all_err, axis=1)

# def compute_for_one_channel(patch_predictions, gt, var_factor = 1, var_offset = 0):
#     """
#     patch_predictions: Batch X N x H x W 
#     gt: Batch X H x W
#     """
#     # print(patch_predictions.shape, gt.shape)
#     mmse_sample = patch_predictions.mean(axis=1)
#     var = ((mmse_sample[:,None] - patch_predictions) ** 2)
#     # print(var.min(), var.max())
#     std = np.sqrt(var)
#     std = std * var_factor + var_offset
#     var = std ** 2
#     var  =var.mean(axis=(2,3))
#     # print(var.min(), var.max())
#     mse_mmse = ((mmse_sample - gt) ** 2).mean(axis=(1,2))
    
#     true_error_percentiles  = []
#     for i in range(patch_predictions.shape[0]):
#         var_sample = var[i]
#         mse_mmse_sample = mse_mmse[i]
#         percentile = scipy.stats.percentileofscore(var_sample,mse_mmse_sample)
#         true_error_percentiles.append(percentile)
#     return np.array(true_error_percentiles)

def compute_for_one_channel(patch_predictions, gt, var_factor = 1, var_offset = 0, mmse_sample_for_error=None):
    """
    patch_predictions: Batch X N x H x W 
    gt: Batch X H x W
    """
    # print(patch_predictions.shape, gt.shape)
    mmse_sample = patch_predictions.mean(axis=1)
    var = ((mmse_sample[:,None] - patch_predictions) ** 2)
    # print(var.min(), var.max())
    # std = np.sqrt(var)
    var = var * var_factor + var_offset
    # var = std ** 2
    var  =var.mean(axis=(2,3))
    if mmse_sample_for_error is None:
        mmse_sample_for_error = mmse_sample
    # print(var.min(), var.max())
    mse_mmse = ((mmse_sample_for_error - gt) ** 2).mean(axis=(1,2))
    # var: Nxmmse_count, mse_mmse: N
    # breakpoint()
    return (var, mse_mmse)

    # true_error_percentiles  = []
    # for i in range(patch_predictions.shape[0]):
    #     var_sample = var[i]
    #     mse_mmse_sample = mse_mmse[i]
    #     percentile = scipy.stats.percentileofscore(var_sample,mse_mmse_sample)
    #     true_error_percentiles.append(percentile)
    # return np.array(true_error_percentiles)


def get_calibration_coverage_data(model, dset, num_workers=4, batch_size = 32, mmse_count = 10, elem_size=10, calib_stats_dict=None, background_patch_detection_func=None):
    dloader = DataLoader(dset, pin_memory=False, num_workers=num_workers, shuffle=False, batch_size=batch_size)
    var = []
    err = []
    with torch.no_grad():
        for batch in tqdm(dloader):
            inp, tar = batch[:2]
            inp = inp.cuda()
            tar = tar.cuda()

            x_normalized = model.normalize_input(inp)
            tar_normalized = model.normalize_target(tar)
            recon_img_list = []
            for _ in range(mmse_count):
                pred_imgs, _ = model(x_normalized)
                pred_imgs = pred_imgs[:,:tar.shape[1]]
                recon_img_list.append(pred_imgs.cpu().numpy()[:,None])
            samples = np.concatenate(recon_img_list, axis=1)
            var_batch, err_batch = compute_for_one_batch(samples, tar_normalized.cpu().numpy(), elem_size=elem_size, calib_stats_dict=calib_stats_dict, background_patch_detection_func=background_patch_detection_func)
            var.append(var_batch)
            err.append(err_batch)
    
    return np.concatenate(var, axis=0), np.concatenate(err, axis=0)

