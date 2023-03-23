import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from disentangle.analysis.lvae_utils import get_img_from_forward_output


def replace_z(into_latent, with_latent, ch_start, ch_end):
    desired_latent_arr = []
    with torch.no_grad():
        for i in range(len(into_latent)):
            z_inp = into_latent[i].clone()
            z_tar = with_latent[i]
            # z_inp will be a batch size, z_tar will be for one input and therefore first dimension should not be present.
            assert z_inp.shape[1:] == z_tar.shape, f'{z_inp.shape}, {z_tar.shape}'
            z_inp[:, ch_start:ch_end] = z_tar[None, ch_start:ch_end]
            desired_latent_arr.append(z_inp)
    return desired_latent_arr


def replace_alpha(into_latent, with_latent, ch_start, ch_end):
    pass


def replace_ch1(into_latent, with_latent, config):
    ch_start, ch_end = config.model.cl_latent_start_end_ch1
    return replace_z(into_latent, with_latent, ch_start, ch_end)


def replace_ch2(into_latent, with_latent, config):
    ch_start, ch_end = config.model.cl_latent_start_end_ch2
    return replace_z(into_latent, with_latent, ch_start, ch_end)


def decode(model, z):
    out, td_data = model.topdown_pass(None, n_img_prior=1, forced_latent=z)
    return out, td_data


def get_latent_normalized_input(model, inp):
    _, td_data = model(inp)
    reference_latent = [z.get()[0] for z in td_data['q_mu']]
    return reference_latent


def get_latent(model, dset, replace_idx):
    inp, _ = dset[replace_idx]
    inp = torch.Tensor(inp[None]).cuda()
    inp = model.normalize_input(inp)
    return get_latent_normalized_input(model, inp)


def get_dset_predictions_CL(model,
                            dset,
                            replace_latent,
                            replace_ch_start,
                            replace_ch_end,
                            batch_size,
                            model_type=None,
                            mmse_count=1,
                            num_workers=4):
    dloader = DataLoader(dset, pin_memory=False, num_workers=num_workers, shuffle=False, batch_size=batch_size)
    predictions = []
    losses = []
    logvar_arr = []
    with torch.no_grad():
        reference_latent = replace_latent
        for batch in tqdm(dloader):
            inp, alpha_class_idx, ch1_idx, ch2_idx = batch
            inp = inp.cuda()
            recon_img_list = []
            for mmse_idx in range(mmse_count):
                x_normalized = model.normalize_input(inp)
                recon_normalized, td_data = model(x_normalized)
                cl_loss_alpha, cl_loss_ch1, cl_loss_ch2 = model.compute_all_CL_losses(
                    td_data, alpha_class_idx, ch1_idx, ch2_idx)

                import pdb
                pdb.set_trace()
                losses.append((cl_loss_alpha, cl_loss_ch1, cl_loss_ch2))

                new_latents = replace_z([z.get() for z in td_data['q_mu']], reference_latent, replace_ch_start,
                                        replace_ch_end)
                out, _ = decode(new_latents)
                recons_img = get_img_from_forward_output(out, model)
                recon_img_list.append(recons_img.cpu()[None])

            mmse_imgs = torch.mean(torch.cat(recon_img_list, dim=0), dim=0)
            predictions.append(mmse_imgs.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def compute_mean_std_with_alpha(alpha, dset):
    mean, std = dset.get_mean_std()
    mean = mean.squeeze()
    std = std.squeeze()
    mean = mean[0] * alpha + mean[1] * (1 - alpha)
    std = std[0] * alpha + std[1] * (1 - alpha)
    return mean, std


def compute_input_with_alpha(img_tuples, alpha, dset):
    assert len(img_tuples) == 2
    assert dset._normalized_input is True, "normalization should happen here"

    inp = img_tuples[0] * alpha + img_tuples[1] * (1 - alpha)
    mean, std = compute_mean_std_with_alpha(alpha)
    inp = (inp - mean) / std
    return inp.astype(np.float32)