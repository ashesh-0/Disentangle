"""
Here, we ensure that contrastive loss
"""

import numpy as np
import torch
import torch.nn as nn


def nc2(n):
    return int(n * (n - 1) / 2)


def get_similarity_matrix(noise_levels):
    """
    noise_levels: N*1
    Returns:
        N*N boolean matrix X. X[i,j] is true iff noise_levels[i] == noise_levels[j]. For digonal elements, it has 0
    """
    batch_size = len(noise_levels)
    noise_levels = noise_levels.view(batch_size, -1)
    b = noise_levels[:, :, None].repeat(1, 1, batch_size)
    c = torch.swapaxes(b, 0, 2)
    sim_matrix = b == c
    assert sim_matrix.shape == (batch_size, 1, batch_size)
    sim_matrix = sim_matrix[:, 0, :]
    sim_matrix[np.arange(batch_size), np.arange(batch_size)] = 0
    return sim_matrix


def get_contrastive_loss_vectorized_elementwise(attribute_z, attribute_labels, tau_pos=None, tau_neg=None):
    assert tau_pos is not None
    assert tau_neg is not None
    # compute pairwise MSE between all attribute_zs.
    batch_size = len(attribute_z)
    attribute_z = attribute_z.view(batch_size, -1)
    b = attribute_z[:, :, None].repeat(1, 1, batch_size)
    c = torch.swapaxes(b, 0, 2)
    pairwise_mse = torch.sum((b - c)**2, dim=1) / b.shape[1]

    # clamp mse values
    pairwise_sim = get_similarity_matrix(attribute_labels)
    pairwise_mse[pairwise_sim] = torch.clamp(pairwise_mse[pairwise_sim] - tau_pos, min=0)
    pairwise_mse[~pairwise_sim] = torch.clamp(tau_neg - pairwise_mse[~pairwise_sim], min=0)
    pairwise_mse[np.arange(batch_size), np.arange(batch_size)] = 0
    return {'CL': pairwise_mse, 'sim_mask': pairwise_sim}


def get_contrastive_loss_vectorized(attribute_z, attribute_labels, tau_pos=None, tau_neg=None):
    mse_dict = get_contrastive_loss_vectorized_elementwise(
        attribute_z,
        attribute_labels,
        tau_pos=tau_pos,
        tau_neg=tau_neg,
    )
    pairwise_mse = mse_dict['CL']
    pairwise_sim = mse_dict['sim_mask']
    batch_size = len(pairwise_mse)
    # scale mse values
    c_pos = torch.sum(pairwise_sim) / 2
    c_neg = nc2(batch_size) - c_pos
    pairwise_mse[pairwise_sim] = pairwise_mse[pairwise_sim] * 1 / c_pos
    pairwise_mse[~pairwise_sim] = pairwise_mse[~pairwise_sim] * 1 / c_neg
    return pairwise_mse.sum() / 2


class ContrastiveLearninglossOnLatent(nn.Module):

    def __init__(self, latent_size_dict, tau_pos, tau_neg) -> None:
        super().__init__()
        self._lsizes = latent_size_dict

        self._tau_pos = tau_pos
        self._tau_neg = tau_neg

    def get_loss(self, latent_activations, attribute_labels, ch_start=None, ch_end=None, tau_pos=None, tau_neg=None):
        assert (tau_pos is None and tau_neg is None) or (tau_pos is not None and tau_neg is not None)
        if tau_pos is None:
            tau_pos = self._tau_pos
        if tau_neg is None:
            tau_neg = self._tau_neg

        latent_loss_dict = {}
        seen_loss_sz = []
        for one_layer_activation in latent_activations:
            sz = one_layer_activation.shape[-1]

            assert sz not in seen_loss_sz
            seen_loss_sz.append(sz)

            if self._lsizes[sz] == 0:
                continue
            if ch_start is None or ch_end is None:
                N = self._lsizes[sz]
                attribute_z = one_layer_activation[:, :N]
                assert ch_end is None and ch_start is None
            else:
                assert isinstance(ch_start, dict)
                assert isinstance(ch_end, dict)
                attribute_z = one_layer_activation[:, ch_start[sz]:ch_end[sz]]

            latent_loss_dict[sz] = get_contrastive_loss_vectorized(attribute_z,
                                                                   attribute_labels,
                                                                   tau_pos=tau_pos,
                                                                   tau_neg=tau_neg)

        return latent_loss_dict

    def forward(self, latent_activations, attribute_labels, ch_start=None, ch_end=None, tau_pos=None, tau_neg=None):
        return self.get_loss(latent_activations,
                             attribute_labels,
                             ch_start=ch_start,
                             ch_end=ch_end,
                             tau_pos=tau_pos,
                             tau_neg=tau_neg)


class DisentanglementModule(nn.Module):

    def __init__(self, latent_size_dict, image_model, cl_latent_weight=None) -> None:
        super().__init__()
        self._lsizes = latent_size_dict
        self._image_model = image_model
        for param in self._image_model.parameters():
            param.requires_grad = False

        self._tau_pos = 0.1
        self._tau_neg = 0.1
        assert cl_latent_weight is not None and cl_latent_weight <= 1 and cl_latent_weight >= 0
        self._w = cl_latent_weight

    def get_loss(self, activations, pred_image, noise_levels):
        latent_loss = 0
        cnt = 0
        for key in activations:
            if self._lsizes[key] == 0:
                continue
            N = self._lsizes[key]
            cnt += 1
            noise_z = activations[key][:, :N]
            latent_loss += get_contrastive_loss_vectorized(noise_z, noise_levels, tau_pos=0, tau_neg=self._tau_neg)

        latent_loss = latent_loss / cnt

        if self._w < 1:
            representation = self._image_model(pred_image)
            output_contrastive_loss = get_contrastive_loss_vectorized(representation,
                                                                      noise_levels,
                                                                      tau_pos=self._tau_pos,
                                                                      tau_neg=self._tau_neg)

            return self._w * latent_loss + (1 - self._w) * output_contrastive_loss
        return latent_loss

    def forward(self, activations, pred_image, noise_levels):
        return self.get_loss(activations, pred_image, noise_levels)


class ContrastiveLearningLossBatchHandler:

    def __init__(self, config) -> None:
        # Contrastive learning loss.
        # TODO: Figure out the logic for CL. This will be different.
        tau_pos = config.loss.cl_tau_pos
        tau_neg = config.loss.cl_tau_neg
        self._cl_latent_start_end_alpha = config.model.cl_latent_start_end_alpha
        self._cl_latent_start_end_ch1 = config.model.cl_latent_start_end_ch1
        self._cl_latent_start_end_ch2 = config.model.cl_latent_start_end_ch2
        self.cl_channels = config.model.z_dims
        self.cl_weight = config.loss.cl_weight
        self._skip_cl_on_alpha = config.loss.skip_cl_on_alpha

        self._cl_loss = ContrastiveLearninglossOnLatent(
            {config.data.image_size // 2**(i + 1): self.cl_channels[i]
             for i in range(len(self.cl_channels))},
            tau_pos,
            tau_neg,
        )

    def contrastive_learning_loss(self,
                                  latent_activations,
                                  class_idx,
                                  ch_start=None,
                                  ch_end=None,
                                  tau_pos=None,
                                  tau_neg=None):

        loss_dict = self._cl_loss.forward(
            latent_activations,
            class_idx,
            ch_start=ch_start,
            ch_end=ch_end,
            tau_pos=tau_pos,
            tau_neg=tau_neg,
        )
        return loss_dict

    def get_contrastive_learning_loss(self,
                                      latent_activations,
                                      class_idx,
                                      ch_start=None,
                                      ch_end=None,
                                      tau_pos=None,
                                      tau_neg=None):
        cl_loss = 0
        cl_loss_dict = self.contrastive_learning_loss(
            latent_activations,
            class_idx,
            ch_start=ch_start,
            ch_end=ch_end,
            tau_pos=tau_pos,
            tau_neg=tau_neg,
        )
        for _, v in cl_loss_dict.items():
            cl_loss += v
        cl_loss = cl_loss / len(cl_loss_dict)
        return cl_loss

    def compute_all_CL_losses(self, td_data, alpha_class_idx, ch1_idx, ch2_idx):
        alpha_ch_start, alpha_ch_end = self._cl_latent_start_end_alpha
        q_mu = [z.get() for z in td_data['q_mu']]

        def to_mu_dic(val):
            return {z.shape[-1]: val for z in q_mu}

        if self._skip_cl_on_alpha:
            cl_loss_alpha = 0.0
        else:
            cl_loss_alpha = self.get_contrastive_learning_loss(q_mu,
                                                               alpha_class_idx,
                                                               ch_start=to_mu_dic(alpha_ch_start),
                                                               ch_end=to_mu_dic(alpha_ch_end))

        ch1_start, ch1_end = self._cl_latent_start_end_ch1
        cl_loss_ch1 = self.get_contrastive_learning_loss(q_mu,
                                                         ch1_idx,
                                                         ch_start=to_mu_dic(ch1_start),
                                                         ch_end=to_mu_dic(ch1_end))

        ch2_start, ch2_end = self._cl_latent_start_end_ch2
        cl_loss_ch2 = self.get_contrastive_learning_loss(q_mu,
                                                         ch2_idx,
                                                         ch_start=to_mu_dic(ch2_start),
                                                         ch_end=to_mu_dic(ch2_end))
        return cl_loss_alpha, cl_loss_ch1, cl_loss_ch2


if __name__ == '__main__':
    rep = torch.Tensor(np.arange(40).reshape(8, 5))
    noise_level_list = torch.Tensor([10, 20, 10, 40, 30, 40, 20, 30])
    loss = get_contrastive_loss_vectorized(rep, noise_level_list, tau_pos=.1, tau_neg=0.1).item()
    assert abs(loss - 262.4) < 1e-4, f'{loss - 262.4}'
