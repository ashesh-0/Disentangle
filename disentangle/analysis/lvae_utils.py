import numpy as np
import torch

from divnoising.core.data_utils import crop_img_tensor
from divnoising.nets.lvae_twinnoise import LadderVAETwinnoise


def get_img_from_forward_output(out, model):
    recons_img = model.likelihood.parameter_net(out)
    recons_img = recons_img * model.data_std[0] + model.data_mean[0]
    return recons_img


def get_z(img, model):
    with torch.no_grad():
        img = torch.Tensor(img[None]).cuda()
        x_normalized = (img - model.data_mean[0]) / model.data_std[0]
        recons_img_latent, td_data = model(x_normalized)
        q_mu = td_data['q_mu']
        recons_img = get_img_from_forward_output(recons_img_latent, model)
        return recons_img, q_mu


def get_recons_with_latent(img_shape, z, model):
    # Top-down inference/generation
    out, td_data = model.topdown_pass(None, forced_latent=z, n_img_prior=1)
    # Restore original image size
    out = crop_img_tensor(out, img_shape)

    return get_img_from_forward_output(out, model)
