"""
Here, we have functions which can be used to quantify uncertainty in the predictions.
"""
import numpy as np
from typing import List, Dict
import torch
from disentangle.analysis.lvae_utils import get_img_from_forward_output
from disentangle.utils import PSNR, RangeInvariantPsnr


def sample_images(model, dset, idx_list, sample_count: int = 5) -> Dict[str, dict]:
    output = {}
    with torch.no_grad():
        for img_idx in idx_list:
            inp, tar = dset[img_idx]
            output[img_idx] = {'rec': [], 'tar': tar}
            inp = torch.Tensor(inp[None]).cuda()
            x_normalized = model.normalize_input(inp)
            for _ in range(sample_count):
                recon_normalized, _ = model(x_normalized)
                imgs = get_img_from_forward_output(recon_normalized, model)
                output[img_idx]['rec'].append(imgs[0].cpu().numpy())

    return output


def compute_regionwise_metric_one_pair(data1, data2, metric_types: List[str], regionsize: int) -> Dict[str, np.ndarray]:
    # ensure that we are working with a square
    assert data1.shape[-1] == data1.shape[-2]
    assert data1.shape == data2.shape
    Nh = data1.shape[-2] // regionsize
    Nw = data1.shape[-1] // regionsize
    output = {mtype: np.zeros((Nh, Nw)) for mtype in metric_types}
    for hidx in range(Nh):
        for widx in range(Nw):
            h = hidx * regionsize
            w = widx * regionsize
            d1 = data1[..., h:h + regionsize, w:w + regionsize]
            d2 = data2[..., h:h + regionsize, w:w + regionsize]
            met_dic = _compute_metrics(d1, d2, metric_types)
            for mtype in metric_types:
                output[mtype][hidx, widx] = met_dic[mtype]

    return output


def _compute_metrics(data1, data2, metric_types: List[str]) -> Dict[str, float]:
    output = {}
    for metric_type in metric_types:
        assert metric_type in ['PSNR', 'RangeInvariantPsnr', 'MSE']

        if metric_type == 'MSE':
            metric = np.mean((data1 - data2) ** 2)
        elif metric_type == 'PSNR':
            metric = PSNR(data1, data2)
        elif metric_type == 'RangeInvariantPsnr':
            metric = RangeInvariantPsnr(data1, data2)
        output[metric_type] = metric
    return output


def compute_regionwise_metric(model, dset, idx_list: List[int], metric_types: List[str], regionsize: int = 64,
                              sample_count: int = 5) -> Dict[int, dict]:
    """
    This will get the prediction multiple times for each of the idx. It would then compute the pairswise metric
    between the predictions, that too on small regions. So, if the model is not sure about a certain region, it would simply
    predict very different things every time and we should get a low PSNR in that region.
    Args:
        model: model
        dset: the dataset
        idx_list: list of idx for which we want to compute this metric
    """
    output = {}
    sample_dict = sample_images(model, dset, idx_list, sample_count=sample_count)
    for img_idx in idx_list:
        assert len(sample_dict[img_idx]['rec']) == sample_count
        rec_list = sample_dict[img_idx]['rec']
        for idx1 in range(sample_count):
            output[idx1] = {}
            # NOTE: we need to iterate starting from 0 and not from idx1 + 1 since not every metric is symmetric.
            # PSNR is definitely not.
            for idx2 in range(sample_count):
                if idx1 == idx2:
                    continue
                output[idx1][idx2] = compute_regionwise_metric_one_pair(rec_list[idx1], rec_list[idx2], metric_types,
                                                                        regionsize)
    return output
