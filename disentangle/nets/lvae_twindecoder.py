import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn

from disentangle.core.data_utils import Interpolate, crop_img_tensor, pad_img_tensor
from disentangle.core.likelihoods import GaussianLikelihood, NoiseModelLikelihood
from disentangle.core.loss_type import LossType
from disentangle.losses import free_bits_kl
from disentangle.nets.lvae import LadderVAE
from disentangle.nets.lvae_layers import (BottomUpDeterministicResBlock, BottomUpLayer, TopDownDeterministicResBlock,
                                          TopDownLayer)

from typing import List, Tuple


class LadderVAETwinDecoder(LadderVAE):
    def __init__(self, data_mean, data_std, config):
        super().__init__(data_mean, data_std, config, target_ch=1)

        del self.top_down_layers
        self.top_down_layers = None
        self.top_down_layers_l1 = nn.ModuleList([])
        self.top_down_layers_l2 = nn.ModuleList([])

        nonlin = self.get_nonlin()

        for i in range(self.n_layers):
            # Whether this is the top layer
            is_top = i == self.n_layers - 1

            self.top_down_layers_l1.append(
                TopDownLayer(
                    z_dim=self.z_dims[i],
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=self.n_filters // 2,
                    is_top_layer=is_top,
                    downsampling_steps=self.downsample[i],
                    nonlin=nonlin,
                    merge_type=self.merge_type,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    stochastic_skip=self.stochastic_skip,
                    learn_top_prior=self.learn_top_prior,
                    top_prior_param_shape=self.get_top_prior_param_shape(),
                    res_block_type=self.res_block_type,
                    gated=self.gated,
                    analytical_kl=self.analytical_kl,
                ))

            self.top_down_layers_l2.append(
                TopDownLayer(
                    z_dim=self.z_dims[i],
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=self.n_filters // 2,
                    is_top_layer=is_top,
                    downsampling_steps=self.downsample[i],
                    nonlin=nonlin,
                    merge_type=self.merge_type,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    stochastic_skip=self.stochastic_skip,
                    learn_top_prior=self.learn_top_prior,
                    top_prior_param_shape=self.get_top_prior_param_shape(),
                    res_block_type=self.res_block_type,
                    gated=self.gated,
                    analytical_kl=self.analytical_kl,
                ))

        # Final top-down layer
        self.final_top_down_l1 = self.get_final_top_down()
        self.final_top_down_l2 = self.get_final_top_down()
        # Define likelihood
        assert self.likelihood_form == 'gaussian'
        del self.likelihood
        self.likelihood = None
        self.likelihood_l1 = GaussianLikelihood(self.n_filters // 2, self.target_ch, predict_logvar=self.predict_logvar)
        self.likelihood_l2 = GaussianLikelihood(self.n_filters // 2, self.target_ch, predict_logvar=self.predict_logvar)

    def get_final_top_down(self):
        modules = list()
        nonlin = self.get_nonlin()
        if not self.no_initial_downscaling:
            modules.append(Interpolate(scale=2))
        for i in range(self.blocks_per_layer):
            modules.append(
                TopDownDeterministicResBlock(
                    c_in=self.n_filters // 2,
                    c_out=self.n_filters // 2,
                    nonlin=nonlin,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    res_block_type=self.res_block_type,
                    gated=self.gated,
                ))

        return nn.Sequential(*modules)

    def sample_from_q(self, x, masks=None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        img_size = x.size()[2:]

        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)
        bu_values_l1, bu_values_l2 = self.get_separate_bu_values(bu_values)

        sample1 = self._sample_from_q(bu_values_l1, top_down_layers=self.top_down_layers_l1,
                                      final_top_down_layer=self.final_top_down_l1, masks=masks)

        sample2 = self._sample_from_q(bu_values_l2, top_down_layers=self.top_down_layers_l2,
                                      final_top_down_layer=self.final_top_down_l2, masks=masks)
        return sample1, sample2

    @staticmethod
    def get_separate_bu_values(bu_values):
        """
        One bu_value list for each decoder
        """
        bu_values_l1 = []
        bu_values_l2 = []

        for one_level_bu in bu_values:
            bu_l1, bu_l2 = one_level_bu.chunk(2, dim=1)
            bu_values_l1.append(bu_l1)
            bu_values_l2.append(bu_l2)
        return bu_values_l1, bu_values_l2

    def forward(self, x):
        img_size = x.size()[2:]

        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x)
        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)
        bu_values_l1, bu_values_l2 = self.get_separate_bu_values(bu_values)

        # Top-down inference/generation
        out_l1, td_data_l1 = self.topdown_pass(
            bu_values_l1,
            top_down_layers=self.top_down_layers_l1,
            final_top_down_layer=self.final_top_down_l1,
        )
        out_l2, td_data_l2 = self.topdown_pass(
            bu_values_l2,
            top_down_layers=self.top_down_layers_l2,
            final_top_down_layer=self.final_top_down_l2,
        )

        # Restore original image size
        out_l1 = crop_img_tensor(out_l1, img_size)
        out_l2 = crop_img_tensor(out_l2, img_size)
        td_data = {
            'kl': [(td_data_l1['kl'][i] + td_data_l2['kl'][i]) / 2 for i in range(len(td_data_l1['kl']))],
            'z': [torch.cat([td_data_l1['z'][i], td_data_l2['z'][i]], dim=1) for i in range(len(td_data_l1['z']))],
        }
        return out_l1, out_l2, td_data

    def get_reconstruction_loss(self, reconstruction_l1, reconstruction_l2, target, return_predicted_img=False):
        # Log likelihood
        ll, like1_dict = self.likelihood_l1(reconstruction_l1, target[:, 0:1])
        recons_loss_l1 = -ll.mean()

        ll, like2_dict = self.likelihood_l2(reconstruction_l2, target[:, 1:])
        recons_loss_l2 = -ll.mean()
        recon_loss = (recons_loss_l1 + recons_loss_l2) / 2
        if return_predicted_img:
            rec_imgs = [like1_dict['params']['mean'], like2_dict['params']['mean']]
            return recon_loss, rec_imgs

        return recon_loss

    def compute_gradient_norm(self):
        grad_norm_bottom_up = self._compute_gradient_norm(self.bottom_up_layers)
        grad_norm_top_down = 0.5 * self._compute_gradient_norm(self.top_down_layers_l1)
        grad_norm_top_down += 0.5 * self._compute_gradient_norm(self.top_down_layers_l2)
        return grad_norm_bottom_up, grad_norm_top_down

    def training_step(self, batch, batch_idx):
        x, target = batch
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)

        out_l1, out_l2, td_data = self.forward(x_normalized)

        recons_loss = self.get_reconstruction_loss(out_l1, out_l2, target_normalized)
        kl_loss = self.get_kl_divergence_loss(td_data)

        net_loss = recons_loss + self.get_kl_weight() * kl_loss
        self.log('reconstruction_loss', recons_loss, on_epoch=True)
        self.log('kl_loss', kl_loss, on_epoch=True)
        self.log('training_loss', net_loss, on_epoch=True)
        self.log('lr', self.lr, on_epoch=True)
        self.log('grad_norm_bottom_up', self.grad_norm_bottom_up, on_epoch=True)
        self.log('grad_norm_top_down', self.grad_norm_top_down, on_epoch=True)
        output = {
            'loss': net_loss,
            'reconstruction_loss': recons_loss.detach(),
            'kl_loss': kl_loss.detach(),
        }
        return output

    def validation_step(self, batch, batch_idx):
        x, target = batch
        self.set_params_to_same_device_as(target)

        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)

        out_l1, out_l2, td_data = self.forward(x_normalized)

        recons_loss, recons_img_list = self.get_reconstruction_loss(out_l1, out_l2, target_normalized,
                                                                    return_predicted_img=True)
        self.label1_psnr.update(recons_img_list[0][:, 0], target_normalized[:, 0])
        self.label2_psnr.update(recons_img_list[1][:, 0], target_normalized[:, 1])

        kl_loss = self.get_kl_divergence_loss(td_data)

        net_loss = recons_loss + self.get_kl_weight() * kl_loss
        self.log('val_loss', recons_loss, on_epoch=True)
        if batch_idx == 0 and self.power_of_2(self.current_epoch):
            all_samples_l1 = []
            all_samples_l2 = []
            for i in range(20):
                sample_l1, sample_l2, _ = self(x_normalized[0:1, ...])
                sample_l1 = self.likelihood_l1.parameter_net(sample_l1)
                sample_l2 = self.likelihood_l2.parameter_net(sample_l2)
                all_samples_l1.append(sample_l1[None])
                all_samples_l2.append(sample_l2[None])

            all_samples_l1 = torch.cat(all_samples_l1, dim=0)
            all_samples_l1 = all_samples_l1 * self.data_std + self.data_mean
            all_samples_l1 = all_samples_l1.cpu()
            img_mmse_l1 = torch.mean(all_samples_l1, dim=0)[0]

            all_samples_l2 = torch.cat(all_samples_l2, dim=0)
            all_samples_l2 = all_samples_l2 * self.data_std + self.data_mean
            all_samples_l2 = all_samples_l2.cpu()
            img_mmse_l2 = torch.mean(all_samples_l2, dim=0)[0]

            self.log_images_for_tensorboard(all_samples_l1[:, 0, 0, ...], target[0, 0, ...], img_mmse_l1[0], 'label1')
            self.log_images_for_tensorboard(all_samples_l2[:, 0, 0, ...], target[0, 1, ...], img_mmse_l2[0], 'label2')

        return net_loss
