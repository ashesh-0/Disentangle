import numpy as np
import torch

from disentangle.core.numpy_decorator import allow_numpy
from disentangle.core.psnr import fix, zero_mean
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


@allow_numpy
def range_invariant_multiscale_ssim(gt_, pred_):
    """
    Computes range invariant multiscale ssim for one channel.
    This has the benefit that it is invariant to scalar multiplications in the prediction.
    """

    shape = gt_.shape
    gt_ = torch.Tensor(gt_.reshape((shape[0],-1)))
    pred_ = torch.Tensor(pred_.reshape((shape[0],-1)))
    gt_ = zero_mean(gt_)
    pred_ = zero_mean(pred_)
    pred_ = fix(gt_, pred_)
    pred_ = pred_.reshape(shape)
    gt_ = gt_.reshape(shape)
    
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=gt_.max() - gt_.min())
    return ms_ssim(torch.Tensor(pred_[:, None]), torch.Tensor(gt_[:, None])).item()


def compute_multiscale_ssim(gt_, pred_, range_invariant=True):
    """
    Computes multiscale ssim for each channel.
    Args:
    gt_: ground truth image with shape (N, H, W, C)
    pred_: predicted image with shape (N, H, W, C)
    range_invariant: whether to use range invariant multiscale ssim
    """
    ms_ssim_values = {i: None for i in range(gt_.shape[-1])}
    for ch_idx in range(gt_.shape[-1]):
        tar_tmp = gt_[..., ch_idx]
        pred_tmp = pred_[..., ch_idx]
        if range_invariant:
            ms_ssim_values[ch_idx] = [range_invariant_multiscale_ssim(tar_tmp[i:i+1], pred_tmp[i:i+1]) for i in range(tar_tmp.shape[0])]
        else:
            ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=tar_tmp.max() - tar_tmp.min())
            ms_ssim_values[ch_idx] = [ms_ssim(torch.Tensor(pred_tmp[i:i+1, None]), torch.Tensor(tar_tmp[i:i+1, None])).item() for i in range(tar_tmp.shape[0])]

    output = [(np.mean(ms_ssim_values[i]), np.std(ms_ssim_values[i])) for i in range(gt_.shape[-1])]
    return output

def compute_SE(arr):
    """
    Computes standard error of the mean.
    """
    return np.std(arr) / np.sqrt(len(arr))


def compute_custom_ssim(gt_, pred_, ssim_obj_dict):
    """
    Computes multiscale ssim for each channel.
    Args:
    gt_: ground truth image with shape (N, H, W, C)
    pred_: predicted image with shape (N, H, W, C)
    range_invariant: whether to use range invariant multiscale ssim
    """
    ms_ssim_values = {i: None for i in range(gt_.shape[-1])}
    for ch_idx in range(gt_.shape[-1]):
        tar_tmp = gt_[..., ch_idx]
        pred_tmp = pred_[..., ch_idx]
        ms_ssim_values[ch_idx] = [ssim_obj_dict[ch_idx].score(tar_tmp[i], pred_tmp[i]) for i in range(tar_tmp.shape[0])]
    output = [(np.mean(ms_ssim_values[i]), compute_SE(ms_ssim_values[i])) for i in range(gt_.shape[-1])]
    return output
