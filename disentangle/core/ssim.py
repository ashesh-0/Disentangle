import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from disentangle.core.numpy_decorator import allow_numpy
from disentangle.core.psnr import zero_mean, fix

@allow_numpy
def range_invariant_multiscale_ssim(gt_, pred_):
    """
    Computes range invariant multiscale ssim for each channel.
    This has the benefit
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
