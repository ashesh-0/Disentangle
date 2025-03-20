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
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def how_many_lie_in_k_quantiles(calibration_coverage_data, k):
    assert k >= 0 and k <= 100
    q_start = (50 - k/2)/100
    q_end = (50 + k/2)/100
    return np.sum((calibration_coverage_data >= q_start) & (calibration_coverage_data <= q_end))/len(calibration_coverage_data)

def compute_for_one(patch_predictions, gt, percentile_bins = 100, elem_size = 10, calib_stats_dict=None):
    output = []
    for ch_idx in range(gt.shape[1]):
        if calib_stats_dict is not None:
            std_factor = calib_stats_dict[ch_idx]['scalar']
            std_offset = calib_stats_dict[ch_idx]['offset']
        else:
            std_factor = 1
            std_offset = 0
        
        if elem_size is not None:
            n_entries = int((patch_predictions.shape[-1]/elem_size) * (patch_predictions.shape[-2]/elem_size))
            computed_vals = []
            for _ in range(n_entries):
                h = np.random.randint(0, patch_predictions.shape[-2] - elem_size)
                w = np.random.randint(0, patch_predictions.shape[-1] - elem_size)
                val = compute_for_one_channel(patch_predictions[:, :, ch_idx,h:h+elem_size,w:w+elem_size], 
                                              gt[:, ch_idx,h:h+elem_size,w:w+elem_size], 
                                              percentile_bins=percentile_bins,
                                              std_factor=std_factor, std_offset=std_offset)
                computed_vals.append(val)
            output.append(np.concatenate(computed_vals, axis=0))
        else:
            output.append(compute_for_one_channel(patch_predictions[:, :, ch_idx], gt[:, ch_idx], 
                                              percentile_bins=percentile_bins,
                                              std_factor=std_factor, std_offset=std_offset))
    return np.stack(output, axis=1)

def compute_for_one_channel(patch_predictions, gt, percentile_bins = 100, std_factor = 1, std_offset = 0):
    """
    patch_predictions: Batch X N x H x W 
    gt: Batch X H x W
    """
    # print(patch_predictions.shape, gt.shape)
    mmse_sample = patch_predictions.mean(axis=1)
    var = ((mmse_sample[:,None] - patch_predictions) ** 2)
    # print(var.min(), var.max())
    std = np.sqrt(var)
    std = std * std_factor + std_offset
    var = std ** 2
    var  =var.mean(axis=(2,3))
    # print(var.min(), var.max())
    mse_mmse = ((mmse_sample - gt) ** 2).mean(axis=(1,2))
    
    true_error_quantiles  = []
    for i in range(patch_predictions.shape[0]):
        var_sample = var[i]
        mse_mmse_sample = mse_mmse[i]
        # find the quantile of mse_mmse_sample in the mse histogram.
        quantiles = np.percentile(var_sample, np.linspace(0, 100, percentile_bins))
        quantile = np.searchsorted(quantiles, mse_mmse_sample) /percentile_bins
        # print(var_sample.shape, mse_mmse_sample.shape, quantiles.shape, quantile.shape)
        # raise ValueError('stop')
        true_error_quantiles.append(quantile)
    return np.array(true_error_quantiles)


def get_calibration_coverage_data(model, dset, percentile_bins = 100, num_workers=4, batch_size = 32, mmse_count = 10, elem_size=10, calib_stats_dict=None):
    dloader = DataLoader(dset, pin_memory=False, num_workers=num_workers, shuffle=False, batch_size=batch_size)
    q_values_dataset = []
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
            true_error_quantiles = compute_for_one(samples, tar_normalized.cpu().numpy(), percentile_bins=percentile_bins, elem_size=elem_size, calib_stats_dict=calib_stats_dict)
            q_values_dataset.append(true_error_quantiles)

    return np.concatenate(q_values_dataset, axis=0)

