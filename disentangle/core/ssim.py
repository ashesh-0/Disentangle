import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from disentangle.core.numpy_decorator import allow_numpy
from disentangle.core.psnr import zero_mean, fix

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
            ms_ssim_values[ch_idx] = range_invariant_multiscale_ssim(tar_tmp, pred_tmp)
        else:
            ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=tar_tmp.max() - tar_tmp.min())
            ms_ssim_values[ch_idx] = ms_ssim(torch.Tensor(pred_tmp[:, None]), torch.Tensor(tar_tmp[:, None])).item()

    output = [ms_ssim_values[i] for i in range(gt_.shape[-1])]
    return output

