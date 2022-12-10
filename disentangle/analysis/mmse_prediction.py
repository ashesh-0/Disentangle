from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from disentangle.core.model_type import ModelType

def get_mmse_prediction(model, dset, inp_idx, mmse_count, padded_size: int, prediction_size: int, batch_size=16,
                        track_progress: bool = True) -> \
        Tuple[
            torch.Tensor, torch.Tensor]:
    """
    The work here is to simply get the MMSE prediction for a specific input.
    Args:
        model:
        dset: the dataset.
        inp_idx: Input index of the dataset for which MMSE prediction needs to be computed.
        mmse_count: Averaging over how many times?
        padded_size: After padding what should be size of the input to the model.
        prediction_size: How much should be kept for prediction. Ex: padded_size=96 and prediction_size=64. 16 pixesls
                        are padded on all sides in this case.
        batch_size: Used for speeding up the computation.

    Returns:
        MMSE prediction and the target. Both are in normalized state.

    """
    assert padded_size >= prediction_size
    old_img_sz = dset.get_img_sz()
    dset.set_img_sz(padded_size)

    padN = (padded_size - prediction_size) // 2

    with torch.no_grad():
        inp, tar = dset[inp_idx]
        inp = torch.Tensor(inp[None])
        tar = torch.Tensor(tar[None])
        inp = inp.repeat(batch_size, 1, 1, 1)
        tar = tar.repeat(batch_size, 1, 1, 1)
        inp = inp.cuda()
        tar = tar.cuda()
        recon_img_list = []
        range_mmse = range(0, mmse_count, batch_size)
        if track_progress:
            range_mmse = tqdm(range_mmse)

        for i in range_mmse:
            end = min(i + batch_size, mmse_count) - i
            x_normalized = model.normalize_input(inp[:end])
            tar_normalized = model.normalize_target(tar[:end])
            recon_normalized, td_data = model(x_normalized)
            recon_img = model.likelihood.get_mean_lv(recon_normalized)[0]
            if padN > 0:
                tar_normalized = tar_normalized[:, :, padN:-padN, padN:-padN]
                recon_normalized = recon_normalized[:, :, padN:-padN, padN:-padN]
                recon_img = recon_img[:, :, padN:-padN, padN:-padN]

            assert tar_normalized.shape[-1] == prediction_size
            assert tar_normalized.shape[-2] == prediction_size
            assert tar_normalized.shape[-2:] == recon_normalized.shape[-2:]
            recon_img_list.append(recon_img.cpu())
        mmse_img = torch.mean(torch.cat(recon_img_list, dim=0), dim=0)[None]

    dset.set_img_sz(old_img_sz)
    return mmse_img, tar_normalized.cpu()


def get_dset_predictions(model, dset, batch_size, model_type=None, mmse_count=1, num_workers=4):
    dloader = DataLoader(dset, pin_memory=False, num_workers=num_workers, shuffle=False, batch_size=batch_size)

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
            for mmse_idx in range(mmse_count):
                if model_type in [ModelType.UNet, ModelType.BraveNet]:
                    recon_normalized = model(x_normalized)
                    if model_type == ModelType.BraveNet:
                        imgs = recon_normalized[0]
                    else:
                        imgs = recon_normalized
                    rec_loss = model.get_reconstruction_loss(recon_normalized, tar_normalized)

                    if mmse_idx == 0:
                        logvar_arr.append(np.array([-1]))
                        losses.append(rec_loss.cpu().numpy())

                else:
                    recon_normalized, _ = model(x_normalized)
                    rec_loss, imgs = model.get_reconstruction_loss(recon_normalized,
                                                                   tar_normalized,
                                                                   return_predicted_img=True)

                    if mmse_idx == 0:
                        q_dic = model.likelihood.distr_params(recon_normalized)
                        if q_dic['logvar'] is not None:
                            logvar_arr.append(q_dic['logvar'].cpu().numpy())
                        else:
                            logvar_arr.append(np.array([-1]))
                        losses.append(rec_loss['loss'].cpu().numpy())

                recon_img_list.append(imgs.cpu()[None])

            mmse_imgs = torch.mean(torch.cat(recon_img_list, dim=0), dim=0)
            predictions.append(mmse_imgs.cpu().numpy())
    return np.concatenate(predictions, axis=0), np.array(losses), np.concatenate(logvar_arr)
