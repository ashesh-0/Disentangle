"""
Functions used in Critic notebooks
"""
import numpy as np
import torch

from disentangle.utils import PSNR


def _get_critic_prediction(pred: torch.Tensor, tar: torch.Tensor, D) -> torch.Tensor:
    """
    Given a predicted image and a target image, here we return a per sample prediction of 
    the critic regarding whether they belong to real or predicted images.
    Args:
        pred: predicted image
        tar: target image
        D: discriminator model
    """
    pred_label = D(pred)
    tar_label = D(tar)
    pred_label = torch.sigmoid(pred_label)
    tar_label = torch.sigmoid(tar_label)
    N = len(pred_label)
    pred_label = pred_label.view(N, -1)
    tar_label = tar_label.view(N, -1)
    return {
        'generated': {
            'mu': pred_label.mean(dim=1),
            'std': pred_label.std(dim=1)
        },
        'target': {
            'mu': tar_label.mean(dim=1),
            'std': tar_label.std(dim=1)
        }
    }


def get_critic_prediction(model, pred_normalized, target_normalized):
    pred1, pred2 = pred_normalized.chunk(2, dim=1)
    tar1, tar2 = target_normalized.chunk(2, dim=1)
    cpred_1 = _get_critic_prediction(pred1, tar1, model.D1)
    cpred_2 = _get_critic_prediction(pred2, tar2, model.D2)
    return cpred_1, cpred_2


def get_mmse_dict(model, x_normalized, target_normalized, mmse_count):
    img_mmse = 0
    assert mmse_count >= 1
    for _ in range(mmse_count):
        recon_normalized, _ = model(x_normalized)
        ll, dic = model.likelihood(recon_normalized, target_normalized)
        recon_img = dic['mean']
        img_mmse += recon_img / mmse_count

    D_loss = model.get_critic_loss_stats(recon_img, target_normalized)['loss'].cpu().item()
    ll, dic = model.likelihood(recon_normalized, target_normalized)

    cpred_1, cpred_2 = get_critic_prediction(model, recon_img, target_normalized)
    loss_mmse = model.likelihood.log_likelihood(target_normalized, {'mean': img_mmse})
    psnrl1 = np.array(
        [PSNR(target_normalized[i, 0].cpu().numpy(), img_mmse[i, 0].cpu().numpy()) for i in range(len(x_normalized))])
    psnrl2 = np.array(
        [PSNR(target_normalized[i, 1].cpu().numpy(), img_mmse[i, 1].cpu().numpy()) for i in range(len(x_normalized))])

    return {
        'mmse_img': img_mmse,
        'mmse_rec_loss': loss_mmse,
        'img': recon_img,
        'rec_loss': ll,
        'psnr_l1': psnrl1,
        'psnr_l2': psnrl2,
        'critic': {
            'label1': cpred_1,
            'label2': cpred_2,
            'D_loss': D_loss,
        }
    }


def get_label_separated_loss(loss_tensor):
    assert loss_tensor.shape[1] == 2
    return -1 * loss_tensor[:, 0].mean(dim=(1, 2)).cpu().numpy(), -1 * loss_tensor[:, 1].mean(dim=(1, 2)).cpu().numpy()
