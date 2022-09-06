"""
Ladder VAE. Adapted from from https://github.com/juglab/HDN/blob/main/models/lvae.py
"""
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import wandb
from torch import nn
from torch.autograd import Variable

from disentangle.core.data_utils import Interpolate, crop_img_tensor, pad_img_tensor
from disentangle.core.likelihoods import GaussianLikelihood, NoiseModelLikelihood
from disentangle.core.loss_type import LossType
from disentangle.core.metric_monitor import MetricMonitor
from disentangle.core.psnr import RangeInvariantPsnr
from disentangle.losses import free_bits_kl
from disentangle.nets.lvae_layers import (BottomUpDeterministicResBlock, BottomUpLayer, TopDownDeterministicResBlock,
                                          TopDownLayer)

from disentangle.metrics.running_psnr import RunningPSNR


def torch_nanmean(inp):
    return torch.mean(inp[~inp.isnan()])


class LadderVAE(pl.LightningModule):
    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__()
        self.lr = config.training.lr
        self.lr_scheduler_patience = config.training.lr_scheduler_patience
        # grayscale input
        self.color_ch = 1

        # disentangling two grayscale images.
        self.target_ch = target_ch

        self.z_dims = config.model.z_dims
        self.blocks_per_layer = config.model.blocks_per_layer
        self.n_layers = len(self.z_dims)
        self.stochastic_skip = config.model.stochastic_skip
        self.batchnorm = config.model.batchnorm
        self.n_filters = config.model.n_filters
        self.dropout = config.model.dropout
        self.learn_top_prior = config.model.learn_top_prior
        self.img_shape = (config.data.image_size, config.data.image_size)
        self.res_block_type = config.model.res_block_type
        self.gated = config.model.gated
        self.data_mean = torch.Tensor(data_mean) if isinstance(data_mean, np.ndarray) else data_mean
        self.data_std = torch.Tensor(data_std) if isinstance(data_std, np.ndarray) else data_std

        self.noiseModel = None
        self.merge_type = config.model.merge_type
        self.analytical_kl = config.model.analytical_kl
        self.no_initial_downscaling = config.model.no_initial_downscaling
        self.mode_pred = config.model.mode_pred
        self.use_uncond_mode_at = use_uncond_mode_at
        self.nonlin = config.model.nonlin
        self.kl_start = config.loss.kl_start
        self.kl_annealing = config.loss.kl_annealing
        self.kl_annealtime = config.loss.kl_annealtime
        self.predict_logvar = config.model.predict_logvar
        self.logvar_lowerbound = config.model.logvar_lowerbound

        self._var_clip_max = config.model.var_clip_max
        # loss related
        self.loss_type = config.loss.loss_type
        self.kl_weight = config.loss.kl_weight
        self.free_bits = config.loss.free_bits

        # initialize the learning rate scheduler params.
        self.lr_scheduler_monitor = self.lr_scheduler_mode = None
        self._init_lr_scheduler_params(config)

        # vampprior
        self.vp_N = self.vp_enabled = self.vp_dummy_input = self.vp_means = self.vp_latent_ch = self.vp_hw = None
        self._init_vp(config)

        # enabling reconstruction loss on mixed input
        self.mixed_rec_w = 0
        self.enable_mixed_rec = False
        if self.loss_type == LossType.ElboMixedReconstruction:
            self.mixed_rec_w = config.loss.mixed_rec_weight
            self.enable_mixed_rec = True

        self._global_step = 0

        # normalized_input: If input is normalized, then we don't normalize the input.
        # We then just normalize the target. Otherwise, both input and target are normalized.
        self.normalized_input = config.data.normalized_input

        assert (self.data_std is not None)
        assert (self.data_mean is not None)
        if self.noiseModel is None:
            self.likelihood_form = "gaussian"
        else:
            self.likelihood_form = "noise_model"

        self.downsample = [1] * self.n_layers

        # Downsample by a factor of 2 at each downsampling operation
        self.overall_downscale_factor = np.power(2, sum(self.downsample))
        if not config.model.no_initial_downscaling:  # by default do another downscaling
            self.overall_downscale_factor *= 2

        assert max(self.downsample) <= self.blocks_per_layer
        assert len(self.downsample) == self.n_layers

        # Get class of nonlinear activation from string description
        nonlin = self.get_nonlin()

        # First bottom-up layer: change num channels + downsample by factor 2
        # unless we want to prevent this
        stride = 1 if config.model.no_initial_downscaling else 2
        self.first_bottom_up = self.create_first_bottom_up(stride)

        self.multiscale_retain_spatial_dims = config.model.multiscale_retain_spatial_dims
        self.lowres_first_bottom_ups = self._multiscale_count = None
        self._init_multires(config)

        # Init lists of layers
        self.top_down_layers = nn.ModuleList([])
        self.bottom_up_layers = nn.ModuleList([])

        enable_multiscale = self._multiscale_count is not None and self._multiscale_count > 1
        multiscale_lowres_size_factor = 1
        for i in range(self.n_layers):
            # Whether this is the top layer
            is_top = i == self.n_layers - 1
            layer_enable_multiscale = enable_multiscale and self._multiscale_count > i + 1
            # if multiscale is enabled, this is the factor by which the lowres tensor will be larger than
            multiscale_lowres_size_factor *= (1 + int(layer_enable_multiscale))
            # Add bottom-up deterministic layer at level i.
            # It's a sequence of residual blocks (BottomUpDeterministicResBlock)
            # possibly with downsampling between them.
            self.bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=self.n_filters,
                    downsampling_steps=self.downsample[i],
                    nonlin=nonlin,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    res_block_type=self.res_block_type,
                    gated=self.gated,
                    lowres_separate_branch=config.model.multiscale_lowres_separate_branch,
                    enable_multiscale=enable_multiscale,
                    multiscale_retain_spatial_dims=self.multiscale_retain_spatial_dims,
                    multiscale_lowres_size_factor=multiscale_lowres_size_factor,
                ))

            # Add top-down stochastic layer at level i.
            # The architecture when doing inference is roughly as follows:
            #    p_params = output of top-down layer above
            #    bu = inferred bottom-up value at this layer
            #    q_params = merge(bu, p_params)
            #    z = stochastic_layer(q_params)
            #    possibly get skip connection from previous top-down layer
            #    top-down deterministic ResNet
            #
            # When doing generation only, the value bu is not available, the
            # merge layer is not used, and z is sampled directly from p_params.
            #
            self.top_down_layers.append(
                TopDownLayer(
                    z_dim=self.z_dims[i],
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=self.n_filters,
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
                    vp_enabled=is_top and self.vp_enabled,
                ))

        # Final top-down layer
        modules = list()
        if not self.no_initial_downscaling:
            modules.append(Interpolate(scale=2))
        for i in range(self.blocks_per_layer):
            modules.append(
                TopDownDeterministicResBlock(
                    c_in=self.n_filters,
                    c_out=self.n_filters,
                    nonlin=nonlin,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    res_block_type=self.res_block_type,
                    gated=self.gated,
                ))
        self.final_top_down = nn.Sequential(*modules)

        # Define likelihood
        if self.likelihood_form == 'gaussian':
            self.likelihood = GaussianLikelihood(self.n_filters, self.target_ch,
                                                 predict_logvar=self.predict_logvar,
                                                 logvar_lowerbound=self.logvar_lowerbound)
        elif self.likelihood_form == 'noise_model':
            self.likelihood = NoiseModelLikelihood(self.n_filters, self.target_ch, data_mean, data_std, self.noiseModel)
        else:
            msg = "Unrecognized likelihood '{}'".format(self.likelihood_form)
            raise RuntimeError(msg)
        # gradient norms. updated while training. this is also logged.
        self.grad_norm_bottom_up = 0.0
        self.grad_norm_top_down = 0.0
        # PSNR computation on validation.
        self.label1_psnr = RunningPSNR()
        self.label2_psnr = RunningPSNR()

    def create_first_bottom_up(self, init_stride, num_blocks=1):
        nonlin = self.get_nonlin()
        modules = [nn.Conv2d(self.color_ch, self.n_filters, 5, padding=2, stride=init_stride), nonlin()]
        for _ in range(num_blocks):
            modules.append(BottomUpDeterministicResBlock(
                c_in=self.n_filters,
                c_out=self.n_filters,
                nonlin=nonlin,
                batchnorm=self.batchnorm,
                dropout=self.dropout,
                res_block_type=self.res_block_type,
            ))
        return nn.Sequential(*modules)

    def _init_multires(self, config):
        """
        Initialize everything related to multiresolution approach.
        """
        stride = 1 if config.model.no_initial_downscaling else 2
        nonlin = self.get_nonlin()
        self._multiscale_count = config.data.multiscale_lowres_count
        if self._multiscale_count is None:
            self._multiscale_count = 1

        msg = "Multiscale count({}) should not exceed the number of bottom up layers ({}) by more than 1"
        msg = msg.format(config.data.multiscale_lowres_count, len(config.model.z_dims))
        assert self._multiscale_count <= 1 or config.data.multiscale_lowres_count <= 1 + len(config.model.z_dims), msg

        msg = "if multiscale is enabled, then we are just working with monocrome images."
        assert self._multiscale_count == 1 or self.color_ch == 1, msg
        lowres_first_bottom_ups = []
        for _ in range(1, self._multiscale_count):
            first_bottom_up = nn.Sequential(
                nn.Conv2d(self.color_ch, self.n_filters, 5, padding=2, stride=stride), nonlin(),
                BottomUpDeterministicResBlock(
                    c_in=self.n_filters,
                    c_out=self.n_filters,
                    nonlin=nonlin,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    res_block_type=self.res_block_type,
                ))
            lowres_first_bottom_ups.append(first_bottom_up)

        self.lowres_first_bottom_ups = nn.ModuleList(lowres_first_bottom_ups) if len(lowres_first_bottom_ups) else None

    def _init_vp(self, config):
        """
        Initialize things related to Vampprior approach.
        """
        self.vp_enabled = config.model.use_vampprior
        if self.vp_enabled:
            self.vp_N = config.model.vampprior_N
            # create an idle input for calling pseudo-inputs
            self.vp_dummy_input = Variable(torch.eye(self.vp_N, self.vp_N), requires_grad=False)
            nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
            self.vp_means = nn.Sequential(nn.Linear(self.vp_N, int(np.prod(self.img_shape)), bias=False), nonlinearity)
            self.vp_latent_ch = config.model.z_dims[-1]
            self.vp_hw = self.img_shape[0]

    def get_nonlin(self):
        nonlin = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU,
            'elu': nn.ELU,
            'selu': nn.SELU,
        }
        return nonlin[self.nonlin]

    def increment_global_step(self):
        """Increments global step by 1."""
        self._global_step += 1

    @property
    def global_step(self) -> int:
        """Global step."""
        return self._global_step

    def _init_lr_scheduler_params(self, config):
        self.lr_scheduler_monitor = config.model.get('monitor', 'val_loss')
        self.lr_scheduler_mode = MetricMonitor(self.lr_scheduler_monitor).mode()

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.parameters(), lr=self.lr, weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         self.lr_scheduler_mode,
                                                         patience=self.lr_scheduler_patience,
                                                         factor=0.5,
                                                         min_lr=1e-12,
                                                         verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.lr_scheduler_monitor}

    def get_kl_weight(self):
        if (self.kl_annealing == True):
            # calculate relative weight
            kl_weight = (self.current_epoch - self.kl_start) * (1.0 / self.kl_annealtime)
            # clamp to [0,1]
            kl_weight = min(max(0.0, kl_weight), 1.0)

            # if the final weight is given, then apply that weight on top of it
            if self.kl_weight is not None:
                kl_weight = kl_weight * self.kl_weight

        elif self.kl_weight is not None:
            return self.kl_weight
        else:
            kl_weight = 1.0
        return kl_weight

    def get_reconstruction_loss(self, reconstruction, input, return_predicted_img=False):
        """
        Args:
            return_predicted_img: If set to True, the besides the loss, the reconstructed image is also returned.
        """
        # Log likelihood
        ll, like_dict = self.likelihood(reconstruction, input)
        recons_loss = -ll.mean()
        mixed_recons_loss = 0
        output = {'loss': recons_loss}
        if self.enable_mixed_rec:
            mixed_target = torch.mean(input, dim=1, keepdim=True)
            mixed_prediction = torch.mean(like_dict['params']['mean'], dim=1, keepdim=True)
            mixed_recons_ll = self.likelihood.log_likelihood(mixed_target, {'mean': mixed_prediction})
            output['mixed_loss'] = -1 * mixed_recons_ll.mean()

        if return_predicted_img:
            return output, like_dict['params']['mean']

        return output

    def get_kl_divergence_loss(self, topdown_layer_data_dict):
        # kl[i] for each i has length batch_size
        # resulting kl shape: (batch_size, layers)
        kl = torch.cat([kl_layer.unsqueeze(1) for kl_layer in topdown_layer_data_dict['kl']], dim=1)
        nlayers = kl.shape[1]
        for i in range(nlayers):
            kl[:, i] = kl[:, i] / np.prod(topdown_layer_data_dict['z'][i].shape[-3:])

        kl_loss = free_bits_kl(kl, self.free_bits).mean()
        return kl_loss

    def _compute_gradient_norm(self, network):
        max_norm = 0
        for p in network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                max_norm = max(max_norm, param_norm.item())
        return max_norm

    def compute_gradient_norm(self):
        grad_norm_bottom_up = self._compute_gradient_norm(self.bottom_up_layers)
        grad_norm_top_down = self._compute_gradient_norm(self.top_down_layers)
        return grad_norm_bottom_up, grad_norm_top_down

    def backward(self, loss, optimizer, optimizer_idx):
        """
        Overwriding the default function just to compute the gradient norm. it gets logged in trainin_step().
        Logging it here results in memory leak.
        """
        loss.backward(retain_graph=True)
        if optimizer_idx == 0:
            self.grad_norm_bottom_up, self.grad_norm_top_down = self.compute_gradient_norm()

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target = batch
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)

        out, td_data = self.forward(x_normalized)
        recons_loss_dict = self.get_reconstruction_loss(out, target_normalized)
        recons_loss = recons_loss_dict['loss']

        if self.loss_type == LossType.ElboMixedReconstruction:
            recons_loss += self.mixed_rec_w * recons_loss_dict['mixed_loss']
            if enable_logging:
                self.log('mixed_reconstruction_loss', recons_loss_dict['mixed_loss'], on_epoch=True)

        kl_loss = self.get_kl_divergence_loss(td_data)

        net_loss = recons_loss + self.get_kl_weight() * kl_loss
        if enable_logging:
            for i, x in enumerate(td_data['debug_qvar_max']):
                self.log(f'qvar_max:{i}', x.item(), on_epoch=True)

            self.log('reconstruction_loss', recons_loss_dict['loss'], on_epoch=True)
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
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output

    def normalize_input(self, x):
        if self.normalized_input:
            return x
        return (x - self.data_mean.mean()) / self.data_std.mean()

    def normalize_target(self, target):
        return (target - self.data_mean) / self.data_std

    def power_of_2(self, x):
        assert isinstance(x, int)
        if x == 1:
            return True
        if x == 0:
            # happens with validation
            return False
        if x % 2 == 1:
            return False
        return self.power_of_2(x // 2)

    def set_params_to_same_device_as(self, correct_device_tensor):
        if isinstance(self.data_mean, torch.Tensor):
            if self.data_mean.device != correct_device_tensor.device:
                self.data_mean = self.data_mean.to(correct_device_tensor.device)
                self.data_std = self.data_std.to(correct_device_tensor.device)

    def validation_step(self, batch, batch_idx):
        x, target = batch
        self.set_params_to_same_device_as(target)

        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)

        out, td_data = self.forward(x_normalized)
        recons_loss_dict, recons_img = self.get_reconstruction_loss(out, target_normalized, return_predicted_img=True)

        self.label1_psnr.update(recons_img[:, 0], target_normalized[:, 0])
        self.label2_psnr.update(recons_img[:, 1], target_normalized[:, 1])

        psnr_label1 = RangeInvariantPsnr(target_normalized[:, 0], recons_img[:, 0])
        psnr_label2 = RangeInvariantPsnr(target_normalized[:, 1], recons_img[:, 1])
        recons_loss = recons_loss_dict['loss']
        kl_loss = self.get_kl_divergence_loss(td_data)
        net_loss = recons_loss + self.get_kl_weight() * kl_loss
        self.log('val_loss', recons_loss, on_epoch=True)
        val_psnr_l1 = torch_nanmean(psnr_label1).item()
        val_psnr_l2 = torch_nanmean(psnr_label2).item()
        self.log('val_psnr_l1', val_psnr_l1, on_epoch=True)
        self.log('val_psnr_l2', val_psnr_l2, on_epoch=True)
        # self.log('val_psnr', (val_psnr_l1 + val_psnr_l2) / 2, on_epoch=True)

        if batch_idx == 0 and self.power_of_2(self.current_epoch):
            all_samples = []
            for i in range(20):
                sample, _ = self(x_normalized[0:1, ...])
                sample = self.likelihood.get_mean_lv(sample)[0]
                all_samples.append(sample[None])

            all_samples = torch.cat(all_samples, dim=0)
            all_samples = all_samples * self.data_std + self.data_mean
            all_samples = all_samples.cpu()
            img_mmse = torch.mean(all_samples, dim=0)[0]
            self.log_images_for_tensorboard(all_samples[:, 0, 0, ...], target[0, 0, ...], img_mmse[0], 'label1')
            self.log_images_for_tensorboard(all_samples[:, 0, 1, ...], target[0, 1, ...], img_mmse[1], 'label2')

        # return net_loss

    def on_validation_epoch_end(self):
        psnrl1 = self.label1_psnr.get()
        psnrl2 = self.label2_psnr.get()
        psnr = (psnrl1 + psnrl2) / 2
        self.log('val_psnr', psnr, on_epoch=True)
        self.label1_psnr.reset()
        self.label2_psnr.reset()

    def forward(self, x):
        img_size = x.size()[2:]

        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)
        vp_dist_params = None
        if self.vp_enabled:
            if self.vp_dummy_input.device != x_pad.device:
                self.vp_dummy_input = self.vp_dummy_input.to(x_pad.device)

            vp_dist_params = self._vp_compute_mu_logvar()

        # Top-down inference/generation
        out, td_data = self.topdown_pass(bu_values, vp_dist_params=vp_dist_params)
        # Restore original image size
        out = crop_img_tensor(out, img_size)

        return out, td_data

    def bottomup_pass(self, inp):
        # Bottom-up initial layer. The first channel is the original input, what we want to reconstruct.
        # later channels are simply to yield more context.
        x = self.first_bottom_up(inp[:, :1])

        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []

        for i in range(self.n_layers):
            lowres_x = None
            if self._multiscale_count > 1 and i + 1 < inp.shape[1]:
                lowres_x = self.lowres_first_bottom_ups[i](inp[:, i + 1:i + 2])

            x, bu_value = self.bottom_up_layers[i](x, lowres_x=lowres_x)
            bu_values.append(bu_value)

        return bu_values

    def _vp_compute_mu_logvar(self):
        X = self.vp_means(self.vp_dummy_input)  # 500*784, where 500 is the number of psudo inputs.
        X = X.view(-1, 1, self.vp_hw, self.vp_hw)  # 500*1*28*28
        # This is the mean and the var of the individual distribution of psudo inputs.
        # -1 ensures that we just pick the deepest layer. in future, we can simply learn an input of the shape of
        # a penultimate layer and then just use the last bottom_up layer
        p_mu_logvar = self.bottomup_pass(X)[-1]
        return p_mu_logvar

    def sample_from_q(self, x, masks=None):
        img_size = x.size()[2:]

        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)
        return self._sample_from_q(bu_values, masks=masks)

    def _sample_from_q(self, bu_values, top_down_layers=None, final_top_down_layer=None, masks=None):
        if top_down_layers is None:
            top_down_layers = self.top_down_layers
        if final_top_down_layer is None:
            final_top_down_layer = self.final_top_down
        if masks is None:
            masks = [None] * len(bu_values)

        msg = "Multiscale is not supported as of now. You need the output from the previous layers to do this."
        assert self.n_layers == 1, msg
        samples = []
        for i in reversed(range(self.n_layers)):
            bu_value = bu_values[i]

            # Note that the first argument can be set to None since we are just dealing with one level
            sample = top_down_layers[i].sample_from_q(None, bu_value, var_clip_max=self._var_clip_max,
                                                      mask=masks[i])
            samples.append(sample)

        return samples

    def topdown_pass(self,
                     bu_values=None,
                     n_img_prior=None,
                     mode_layers=None,
                     constant_layers=None,
                     forced_latent=None,
                     top_down_layers=None,
                     final_top_down_layer=None,
                     vp_dist_params=None
                     ):
        """
        Args:
            bu_values: Output of the bottom-up pass. It will have values from multiple layers of the ladder.
            n_img_prior: bu_values needs to be none for this. This generates n images from the prior. So, it does
                        not use bottom up pass at all.
            mode_layers: At these layers, sampling is disabled. Mean value is used directly.
            constant_layers: Here, a single instance's z is copied over the entire batch. Also, bottom-up path is not used.
                            So, only prior is used here.
            forced_latent: Here, latent vector is not sampled but taken from here.
            vp_dist_params: This is the vampprior's distribution params (mean and logvar concatenated)
        """
        if top_down_layers is None:
            top_down_layers = self.top_down_layers
        if final_top_down_layer is None:
            final_top_down_layer = self.final_top_down

        # Default: no layer is sampled from the distribution's mode
        if mode_layers is None:
            mode_layers = []
        if constant_layers is None:
            constant_layers = []
        prior_experiment = len(mode_layers) > 0 or len(constant_layers) > 0

        # If the bottom-up inference values are not given, don't do
        # inference, sample from prior instead
        inference_mode = bu_values is not None

        # Check consistency of arguments
        if inference_mode != (n_img_prior is None):
            msg = ("Number of images for top-down generation has to be given "
                   "if and only if we're not doing inference")
            raise RuntimeError(msg)
        if inference_mode and prior_experiment:
            msg = ("Prior experiments (e.g. sampling from mode) are not" " compatible with inference mode")
            raise RuntimeError(msg)

        # Sampled latent variables at each layer
        z = [None] * self.n_layers

        # KL divergence of each layer
        kl = [None] * self.n_layers

        # mean from which z is sampled.
        q_mu = [None] * self.n_layers
        # log(var) from which z is sampled.
        q_lv = [None] * self.n_layers

        # Spatial map of KL divergence for each layer
        kl_spatial = [None] * self.n_layers

        debug_qvar_max = [None] * self.n_layers

        kl_channelwise = [None] * self.n_layers
        if forced_latent is None:
            forced_latent = [None] * self.n_layers

        # log p(z) where z is the sample in the topdown pass
        # logprob_p = 0.

        # Top-down inference/generation loop
        out = out_pre_residual = None
        for i in reversed(range(self.n_layers)):

            # If available, get deterministic node from bottom-up inference
            try:
                bu_value = bu_values[i]
            except TypeError:
                bu_value = None

            # Whether the current layer should be sampled from the mode
            use_mode = i in mode_layers
            constant_out = i in constant_layers
            use_uncond_mode = i in self.use_uncond_mode_at

            # Input for skip connection
            skip_input = out  # TODO or out_pre_residual? or both?

            # Full top-down layer, including sampling and deterministic part
            out, out_pre_residual, aux = top_down_layers[i](out,
                                                            skip_connection_input=skip_input,
                                                            inference_mode=inference_mode,
                                                            bu_value=bu_value,
                                                            n_img_prior=n_img_prior,
                                                            use_mode=use_mode,
                                                            force_constant_output=constant_out,
                                                            forced_latent=forced_latent[i],
                                                            mode_pred=self.mode_pred,
                                                            use_uncond_mode=use_uncond_mode,
                                                            var_clip_max=self._var_clip_max,
                                                            vp_dist_params=vp_dist_params if i == self.n_layers - 1 else None)
            z[i] = aux['z']  # sampled variable at this layer (batch, ch, h, w)
            kl[i] = aux['kl_samplewise']  # (batch, )
            kl_spatial[i] = aux['kl_spatial']  # (batch, h, w)
            q_mu[i] = aux['q_mu']
            q_lv[i] = aux['q_lv']

            kl_channelwise[i] = aux['kl_channelwise']
            debug_qvar_max[i] = aux['qvar_max']
            # if self.mode_pred is False:
            #     logprob_p += aux['logprob_p'].mean()  # mean over batch
            # else:
            #     logprob_p = None
        # Final top-down layer
        out = final_top_down_layer(out)

        data = {
            'z': z,  # list of tensors with shape (batch, ch[i], h[i], w[i])
            'kl': kl,  # list of tensors with shape (batch, )
            'kl_spatial': kl_spatial,  # list of tensors w shape (batch, h[i], w[i])
            'kl_channelwise': kl_channelwise,  # list of tensors with shape (batch, ch[i])
            # 'logprob_p': logprob_p,  # scalar, mean over batch
            'q_mu': q_mu,
            'q_lv': q_lv,
            'debug_qvar_max': debug_qvar_max,
        }
        return out, data

    def pad_input(self, x):
        """
        Pads input x so that its sizes are powers of 2
        :param x:
        :return: Padded tensor
        """
        size = self.get_padded_size(x.size())
        x = pad_img_tensor(x, size)
        return x

    def get_padded_size(self, size):
        """
        Returns the smallest size (H, W) of the image with actual size given
        as input, such that H and W are powers of 2.
        :param size: input size, tuple either (N, C, H, w) or (H, W)
        :return: 2-tuple (H, W)
        """

        # Overall downscale factor from input to top layer (power of 2)
        dwnsc = self.overall_downscale_factor

        # Make size argument into (heigth, width)
        if len(size) == 4:
            size = size[2:]
        if len(size) != 2:
            msg = ("input size must be either (N, C, H, W) or (H, W), but it "
                   "has length {} (size={})".format(len(size), size))
            raise RuntimeError(msg)

        # Output smallest powers of 2 that are larger than current sizes
        padded_size = list(((s - 1) // dwnsc + 1) * dwnsc for s in size)

        return padded_size

    def sample_prior(self, n_imgs, mode_layers=None, constant_layers=None):

        # Generate from prior
        out, _ = self.topdown_pass(n_img_prior=n_imgs, mode_layers=mode_layers, constant_layers=constant_layers)
        out = crop_img_tensor(out, self.img_shape)

        # Log likelihood and other info (per data point)
        _, likelihood_data = self.likelihood(out, None)

        return likelihood_data['sample']

    def get_top_prior_param_shape(self, n_imgs=1):
        # TODO num channels depends on random variable we're using
        dwnsc = self.overall_downscale_factor
        sz = self.get_padded_size(self.img_shape)
        h = sz[0] // dwnsc
        w = sz[1] // dwnsc
        c = self.z_dims[-1] * 2  # mu and logvar
        top_layer_shape = (n_imgs, c, h, w)
        return top_layer_shape

    def log_images_for_tensorboard(self, pred, target, img_mmse, label):
        clamped_pred = torch.clamp((pred - pred.min()) / (pred.max() - pred.min()), 0, 1)
        clamped_mmse = torch.clamp((img_mmse - img_mmse.min()) / (img_mmse.max() - img_mmse.min()), 0, 1)
        if target is not None:
            clamped_input = torch.clamp((target - target.min()) / (target.max() - target.min()), 0, 1)
            img = wandb.Image(clamped_input[None].cpu().numpy())
            self.logger.experiment.log({f'target_for{label}': img})
            # self.trainer.logger.experiment.add_image(f'target_for{label}', clamped_input[None], self.current_epoch)
        for i in range(3):
            # self.trainer.logger.experiment.add_image(f'{label}/sample_{i}', clamped_pred[i:i + 1], self.current_epoch)
            img = wandb.Image(clamped_pred[i:i + 1].cpu().numpy())
            self.logger.experiment.log({f'{label}/sample_{i}': img})

        img = wandb.Image(clamped_mmse[None].cpu().numpy())
        self.trainer.logger.experiment.log({f'{label}/mmse (100 samples)': img})
