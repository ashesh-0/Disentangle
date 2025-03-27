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
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm


def find_optimal_scalar_offset(pred, gt):
    """
    Optimal linear transform which minimizes the MSE between pred and gt.

    Start by making them zero mean. 
    This yields a simple formulation of the optimal scalar: covariance(pred,gt)/var(pred)
    then the offset is: mean(gt) - scalar * mean(pred)
    """
    pred_zero_mean = pred - np.mean(pred)
    gt_zero_mean = gt - np.mean(gt)
    scalar = np.sum(pred_zero_mean * gt_zero_mean) / np.sum(pred_zero_mean ** 2)
    offset = np.mean(gt) - scalar * np.mean(pred)
    return scalar, offset


def get_empirical_coverage_and_confidence_level(coverage_data):
    assert coverage_data.ndim == 1, f'coverage_data should be 1D, but got {coverage_data.ndim}D'
    # assert coverage_data.min() >= 0, f'coverage_data should be non-negative, but got {coverage_data.min()}'
    # assert coverage_data.max() <= 100, f'coverage_data should be less than 100, but got {coverage_data.max()}'

    confidence_level = np.arange(0, 100, 1)
    bin_values = np.zeros_like(confidence_level)
    for val in coverage_data:
        for idx, percentile_bin in enumerate(confidence_level):
            # 2 is important because we are looking at the symmetric percentile.
            if 2*val <= percentile_bin:
                bin_values[idx] += 1
    empirical_coverage = 100 * bin_values/len(coverage_data)
    return confidence_level, empirical_coverage


def compute_for_one(patch_predictions, gt, percentile_bins = 100, elem_size = 10, calib_stats_dict=None, background_patch_detection_func=None):
    output = []

    masks = []
    if background_patch_detection_func is not None:
        for ch_idx in range(gt.shape[1]):
            bkg_mask = background_patch_detection_func(gt[:, ch_idx], ch_idx)
            masks.append(bkg_mask)

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
                # print('see the shape', val.shape, gt.shape, masks[ch_idx].mean())
                val[masks[ch_idx]] = np.nan

                computed_vals.append(val)
            output.append(np.concatenate(computed_vals, axis=0))
        else:
            output.append(compute_for_one_channel(patch_predictions[:, :, ch_idx], gt[:, ch_idx], 
                                              percentile_bins=percentile_bins,
                                              std_factor=std_factor, std_offset=std_offset))
    output = np.stack(output, axis=1)
    return output

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
    
    true_error_symmetric_percentiles  = []
    for i in range(patch_predictions.shape[0]):
        var_sample = var[i]
        mse_mmse_sample = mse_mmse[i]
        assert np.isscalar(mse_mmse_sample), f'{mse_mmse_sample.shape} should be scalar'
        # find the percentile of mse_mmse_sample in the mse histogram.
        # NOTE: the np.percentile is an approximation, but it is; good enough for our purposes.
        percentile = scipy.stats.percentileofscore(var_sample,mse_mmse_sample)
        
        # quantiles = np.percentile(var_sample, np.linspace(0, 100, 100))
        # percentile = np.searchsorted(quantiles, mse_mmse_sample)

        # sym_percentile = np.abs(percentile - 50)
        # print(var_sample.shape, mse_mmse_sample.shape, quantiles.shape, quantile.shape)
        # raise ValueError('stop')
        true_error_symmetric_percentiles.append(sym_percentile)
    return np.array(true_error_symmetric_percentiles)


def get_calibration_coverage_data(model, dset, percentile_bins = 100, num_workers=4, batch_size = 32, mmse_count = 10, elem_size=10, calib_stats_dict=None, background_patch_detection_func=None):
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
            true_error_quantiles = compute_for_one(samples, tar_normalized.cpu().numpy(), percentile_bins=percentile_bins, elem_size=elem_size, calib_stats_dict=calib_stats_dict, background_patch_detection_func=background_patch_detection_func)
            q_values_dataset.append(true_error_quantiles)

    return np.concatenate(q_values_dataset, axis=0)


# ===========================================================
# finding the optimal transform for the calibration coverage.
# ===========================================================
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


def patches_for_optimal_transform(model, dset,num_workers=4, batch_size = 32, mmse_count = 10, elem_size=10, skip_pixels=0):
    dloader = DataLoader(dset, pin_memory=False, num_workers=num_workers, shuffle=False, batch_size=batch_size)
    tar_patches = []
    recons_patches = []
    skip = skip_pixels//2

    with torch.no_grad():
        for batch in tqdm(dloader):
            inp, tar = batch[:2]
            inp = inp.cuda()
            tar = tar.cuda()

            x_normalized = model.normalize_input(inp)
            tar_normalized = model.normalize_target(tar)
            _, nC, nH, nW = tar.shape
            recon_img_list = []
            for _ in range(mmse_count):
                pred_imgs, _ = model(x_normalized)
                pred_imgs = pred_imgs[:,:nC, skip:nH-skip, skip:nW-skip].cpu().numpy()
                if elem_size is not None:
                    pred_imgs = divide_into_smaller_patches(pred_imgs, elem_size)

                recon_img_list.append(pred_imgs[:,None])
            
            tar_normalized = tar_normalized.cpu().numpy()[..., skip:nH-skip, skip:nW-skip]
            if elem_size is not None:
                tar_normalized = divide_into_smaller_patches(tar_normalized, elem_size)
            recons_patches.append(np.concatenate(recon_img_list, axis=1))
            tar_patches.append(tar_normalized)
    tar_patches = np.concatenate(tar_patches, axis=0)
    recons_patches = np.concatenate(recons_patches, axis=0)
    return tar_patches, recons_patches

def fit_optimal_transform(recons_patches, tar_patches):
    mmse_recons = recons_patches.mean(axis=1)
    # N x C x H x W for both tar_patches and mmse_recons
    assert tar_patches.shape == mmse_recons.shape, f'{tar_patches.shape} != {mmse_recons.shape}'
    assert tar_patches.ndim == 4, f'{tar_patches.ndim} != 4'
    
    sample_var = ((mmse_recons[:,None] - recons_patches) ** 2).mean(axis=(-2,-1))
    sample_std = np.sqrt(sample_var)
    avg_sample_std = sample_std.mean(axis=1)

    rmse = ((mmse_recons - tar_patches) ** 2).mean(axis=(-2,-1))
    rmse = np.sqrt(rmse)
    output = {}
    for ch_idx in range(tar_patches.shape[1]):
        slope, intercept, *_ = stats.linregress(avg_sample_std[:,ch_idx],rmse[:,ch_idx])
        output[ch_idx] = {'scalar':slope, 'offset':intercept}
    return output
