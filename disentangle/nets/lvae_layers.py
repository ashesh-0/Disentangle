"""
Taken from https://github.com/juglab/HDN/blob/main/models/lvae_layers.py
"""
from copy import deepcopy
from typing import Union

import torch
from torch import nn

from disentangle.core.data_utils import crop_img_tensor, pad_img_tensor
from disentangle.core.nn_submodules import ResidualBlock, ResidualGatedBlock
from disentangle.core.stochastic import NormalStochasticBlock2d


class TopDownLayer(nn.Module):
    """
    Top-down layer, including stochastic sampling, KL computation, and small
    deterministic ResNet with upsampling.
    The architecture when doing inference is roughly as follows:
       p_params = output of top-down layer above
       bu = inferred bottom-up value at this layer
       q_params = merge(bu, p_params)
       z = stochastic_layer(q_params)
       possibly get skip connection from previous top-down layer
       top-down deterministic ResNet
    When doing generation only, the value bu is not available, the
    merge layer is not used, and z is sampled directly from p_params.
    If this is the top layer, at inference time, the uppermost bottom-up value
    is used directly as q_params, and p_params are defined in this layer
    (while they are usually taken from the previous layer), and can be learned.
    """
    def __init__(self,
                 z_dim: int,
                 n_res_blocks: int,
                 n_filters: int,
                 is_top_layer: bool = False,
                 downsampling_steps: int = None,
                 nonlin=None,
                 merge_type: str = None,
                 batchnorm: bool = True,
                 dropout: Union[None, float] = None,
                 stochastic_skip: bool = False,
                 res_block_type=None,
                 res_block_kernel=None,
                 res_block_skip_padding=None,
                 gated=None,
                 learn_top_prior=False,
                 top_prior_param_shape=None,
                 analytical_kl=False,
                 vp_enabled=False):
        """
            Args:
                z_dim:          This is the dimension of the latent space.
                n_res_blocks:   Number of TopDownDeterministicResBlock blocks
                n_filters:      Number of channels which is present through out this layer.
                is_top_layer:   Whether it is top layer or not.
                downsampling_steps: How many times upsampling has to be done in this layer. This is typically 1.
                nonlin: What non linear activation is to be applied at various places in this module.
                merge_type: In Top down layer, one merges the information passed from q() and upper layers.
                            This specifies how to mix these two tensors.
                batchnorm: Whether to apply batch normalization at various places or not.
                dropout: Amount of dropout to be applied at various places.
                stochastic_skip: Previous layer's output is mixed with this layer's stochastic output. So, 
                                the previous layer's output has a way to reach this level without going
                                through the stochastic process. However, technically, this is not a skip as
                                both are merged together. 
                res_block_type: Example: 'bacdbac'. It has the constitution of the residual block.
                gated: This is also an argument for the residual block. At the end of residual block, whether 
                        there should be a gate or not.
                learn_top_prior: Whether we want to learn the top prior or not. If set to False, for the top-most
                                 layer, p will be N(0,1). Otherwise, we will still have a normal distribution. It is 
                                 just that the mean and the stdev will be different.
                top_prior_param_shape: This is the shape of the tensor which would contain the mean and the variance
                                        of the prior (which is normal distribution) for the top most layer.
                analytical_kl:  If True, typical KL divergence is calculated. Otherwise, an approximate of it is 
                            calculated.
                vp_enabled: If True, then the vampprior is enabled.
        """

        super().__init__()

        self.is_top_layer = is_top_layer
        self.z_dim = z_dim
        self.stochastic_skip = stochastic_skip
        self.learn_top_prior = learn_top_prior
        self.analytical_kl = analytical_kl
        self.vp_enabled = vp_enabled
        assert vp_enabled is False or (vp_enabled is True and is_top_layer is True)
        # Define top layer prior parameters, possibly learnable
        if is_top_layer:
            if self.vp_enabled:
                # if vamprior is enabled, the prior should come from the q() and not fixed here.
                pass
            else:
                self.top_prior_params = nn.Parameter(torch.zeros(top_prior_param_shape), requires_grad=learn_top_prior)

        # Downsampling steps left to do in this layer
        dws_left = downsampling_steps

        # Define deterministic top-down block: sequence of deterministic
        # residual blocks with downsampling when needed.
        block_list = []

        for _ in range(n_res_blocks):
            do_resample = False
            if dws_left > 0:
                do_resample = True
                dws_left -= 1
            block_list.append(
                TopDownDeterministicResBlock(
                    n_filters,
                    n_filters,
                    nonlin,
                    upsample=do_resample,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    res_block_kernel=res_block_kernel,
                    skip_padding=res_block_skip_padding,
                    gated=gated,
                ))
        self.deterministic_block = nn.Sequential(*block_list)

        # Define stochastic block with 2d convolutions
        self.stochastic = NormalStochasticBlock2d(
            c_in=n_filters,
            c_vars=z_dim,
            c_out=n_filters,
            transform_p_params=(not is_top_layer),
        )

        if not is_top_layer:

            # Merge layer, combine bottom-up inference with top-down
            # generative to give posterior parameters
            self.merge = MergeLayer(
                channels=n_filters,
                merge_type=merge_type,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                res_block_kernel=res_block_kernel,
            )

            # Skip connection that goes around the stochastic top-down layer
            if stochastic_skip:
                self.skip_connection_merger = SkipConnectionMerger(
                    channels=n_filters,
                    nonlin=nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    res_block_kernel=res_block_kernel,
                    res_block_skip_padding=res_block_skip_padding,
                )

    def forward_swapped(
        self,
        q_mu,
        q_lv,
        input_=None,
        skip_connection_input=None,
        n_img_prior=None,
    ):

        # Check consistency of arguments
        inputs_none = input_ is None and skip_connection_input is None
        if self.is_top_layer and not inputs_none:
            raise ValueError("In top layer, inputs should be None")

        # If top layer, define parameters of prior p(z_L)
        if self.is_top_layer:
            p_params = self.top_prior_params

            # Sample specific number of images by expanding the prior
            if n_img_prior is not None:
                p_params = p_params.expand(n_img_prior, -1, -1, -1)

        # Else the input from the layer above is the prior parameters
        else:
            p_params = input_

        x, data_stoch = self.stochastic.forward_swapped(p_params, q_mu, q_lv)
        # Skip connection from previous layer
        if self.stochastic_skip and not self.is_top_layer:
            x = self.skip_connection_merger(x, skip_connection_input)

        # Save activation before residual block: could be the skip
        # connection input in the next layer
        x_pre_residual = x

        # Last top-down block (sequence of residual blocks)
        x = self.deterministic_block(x)

        keys = ['z']
        data = {k: data_stoch[k] for k in keys}
        return x, x_pre_residual, data

    def sample_from_q(self, input_, bu_value, var_clip_max=None, mask=None):
        """
        We sample from q
        """
        if self.is_top_layer:
            q_params = bu_value
        else:
            # NOTE: Here the assumption is that the vampprior is only applied on the top layer.
            assert self.vp_enabled is False
            vp_dist_params = None
            n_img_prior = None
            p_params = self.get_p_params(input_, vp_dist_params, n_img_prior)
            q_params = self.merge(bu_value, p_params)

        sample = self.stochastic.sample_from_q(q_params, var_clip_max)
        if mask:
            return sample[mask]
        return sample

    def get_p_params(self, input_, vp_dist_params, n_img_prior):
        p_params = None
        # If top layer, define parameters of prior p(z_L)
        if self.is_top_layer:
            if self.vp_enabled is False:
                p_params = self.top_prior_params

                # Sample specific number of images by expanding the prior
                if n_img_prior is not None:
                    p_params = p_params.expand(n_img_prior, -1, -1, -1)
            else:
                p_params = vp_dist_params

        # Else the input from the layer above is the prior parameters
        else:
            p_params = input_

        return p_params

    def forward(self,
                input_: Union[None, torch.Tensor] = None,
                skip_connection_input=None,
                inference_mode=False,
                bu_value=None,
                n_img_prior=None,
                forced_latent: Union[None, torch.Tensor] = None,
                use_mode: bool = False,
                force_constant_output=False,
                mode_pred=False,
                use_uncond_mode=False,
                var_clip_max: Union[None, float] = None,
                vp_dist_params: Union[None, torch.Tensor] = None):
        """
        Args:
            input_: output from previous top_down layer.
            skip_connection_input: Currently, this is output from the previous top down layer. 
                                It is mixed with the output of the stochastic layer.
            inference_mode: In inference mode, q_params is not None. Otherwise it is. When q_params is None,
                            everything is generated from the p_params. So, the encoder is not used at all.
            bu_value: Output of the bottom-up pass layer of the same level as this top-down.
            n_img_prior: This affects just the top most top-down layer. This is only present if inference_mode=False.
            forced_latent: If this is a tensor, then in stochastic layer, we don't sample by using p() & q(). We simply 
                            use this as the latent space sampling.
            use_mode:      If it is true, we still don't sample from the q(). We simply 
                            use the mean of the distribution as the latent space.
            force_constant_output: This ensures that only the first sample of the batch is used. Typically used 
                                when infernce_mode is False
            mode_pred: If True, then only prediction happens. Otherwise, KL divergence loss also gets computed.
            use_uncond_mode: Used only when mode_pred=True
            var_clip_max: This is the maximum value the log of the variance of the latent vector for any layer can reach.
            vp_dist_params: This contains the vampprior distribution params. Essentially the mean+logvar concatenated
                            vector for each of the config.model.vampprior_N many trainable custom inputs.
        """
        # Check consistency of arguments
        inputs_none = input_ is None and skip_connection_input is None
        if self.is_top_layer and not inputs_none:
            raise ValueError("In top layer, inputs should be None")

        if vp_dist_params is not None and not self.is_top_layer:
            raise ValueError("VampPrior is implemented only in topmost layer right now")

        p_params = self.get_p_params(input_, vp_dist_params, n_img_prior)

        # In inference mode, get parameters of q from inference path,
        # merging with top-down path if it's not the top layer
        if inference_mode:
            if self.is_top_layer:
                q_params = bu_value
            else:
                if use_uncond_mode:
                    q_params = p_params
                else:
                    q_params = self.merge(bu_value, p_params)

        # In generative mode, q is not used
        else:
            q_params = None

        # Sample from either q(z_i | z_{i+1}, x) or p(z_i | z_{i+1})
        # depending on whether q_params is None
        x, data_stoch = self.stochastic(p_params=p_params,
                                        q_params=q_params,
                                        forced_latent=forced_latent,
                                        use_mode=use_mode,
                                        force_constant_output=force_constant_output,
                                        analytical_kl=self.analytical_kl,
                                        mode_pred=mode_pred,
                                        use_uncond_mode=use_uncond_mode,
                                        var_clip_max=var_clip_max,
                                        vp_enabled=vp_dist_params is not None)

        # Skip connection from previous layer
        if self.stochastic_skip and not self.is_top_layer:
            x = self.skip_connection_merger(x, skip_connection_input)

        # Save activation before residual block: could be the skip
        # connection input in the next layer
        x_pre_residual = x

        # Last top-down block (sequence of residual blocks)
        x = self.deterministic_block(x)

        keys = [
            'z',
            'kl_samplewise',
            'kl_spatial',
            'kl_channelwise',
            # 'logprob_p',
            'logprob_q',
            'qvar_max'
        ]
        data = {k: data_stoch.get(k, None) for k in keys}
        data['q_mu'] = None
        data['q_lv'] = None
        if data_stoch['q_params'] is not None:
            q_mu, q_lv = data_stoch['q_params']
            data['q_mu'] = q_mu
            data['q_lv'] = q_lv
        return x, x_pre_residual, data


class BottomUpLayer(nn.Module):
    """
    Bottom-up deterministic layer for inference, roughly the same as the
    small deterministic Resnet in top-down layers. Consists of a sequence of
    bottom-up deterministic residual blocks with downsampling.
    """
    def __init__(self,
                 n_res_blocks: int,
                 n_filters: int,
                 downsampling_steps: int = 0,
                 nonlin=None,
                 batchnorm: bool = True,
                 dropout: Union[None, float] = None,
                 res_block_type: str = None,
                 res_block_kernel: int = None,
                 res_block_skip_padding: bool = False,
                 gated: bool = None,
                 multiscale_lowres_size_factor: int = None,
                 enable_multiscale: bool = False,
                 lowres_separate_branch=False,
                 multiscale_retain_spatial_dims: bool = False):
        """
        Args:
            n_res_blocks: Number of BottomUpDeterministicResBlock blocks present in this layer.
            n_filters:      Number of channels which is present through out this layer.
            downsampling_steps: How many times downsampling has to be done in this layer. This is typically 1.
            nonlin: What non linear activation is to be applied at various places in this module.
            batchnorm: Whether to apply batch normalization at various places or not.
            dropout: Amount of dropout to be applied at various places.
            res_block_type: Example: 'bacdbac'. It has the constitution of the residual block.
            gated: This is also an argument for the residual block. At the end of residual block, whether 
            there should be a gate or not.
            res_block_kernel:int => kernel size for the residual blocks in the bottom up layer.
            multiscale_lowres_size_factor: How small is the bu_value when compared with low resolution tensor.
            enable_multiscale: Whether to enable multiscale or not.
            multiscale_retain_spatial_dims: typically the output of the bottom-up layer scales down spatially.
                                            However, with this set, we return the same spatially sized tensor.
        """
        super().__init__()
        self.enable_multiscale = enable_multiscale
        self.lowres_separate_branch = lowres_separate_branch
        self.multiscale_retain_spatial_dims = multiscale_retain_spatial_dims
        bu_blocks_downsized = []
        bu_blocks_samesize = []
        for _ in range(n_res_blocks):
            do_resample = False
            if downsampling_steps > 0:
                do_resample = True
                downsampling_steps -= 1
            block = BottomUpDeterministicResBlock(
                c_in=n_filters,
                c_out=n_filters,
                nonlin=nonlin,
                downsample=do_resample,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                res_block_kernel=res_block_kernel,
                skip_padding=res_block_skip_padding,
                gated=gated,
            )
            if do_resample:
                bu_blocks_downsized.append(block)
            else:
                bu_blocks_samesize.append(block)

        self.net_downsized = nn.Sequential(*bu_blocks_downsized)
        self.net = nn.Sequential(*bu_blocks_samesize)
        # using the same net for the lowresolution (and larger sized image)
        self.lowres_net = self.lowres_merge = self.multiscale_lowres_size_factor = None
        if self.enable_multiscale:
            self._init_multiscale(
                n_filters=n_filters,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                multiscale_retain_spatial_dims=multiscale_retain_spatial_dims,
                multiscale_lowres_size_factor=multiscale_lowres_size_factor,
            )

        msg = f'[{self.__class__.__name__}] McEnabled:{int(enable_multiscale)} '
        if enable_multiscale:
            msg += f'McParallelBeam:{int(multiscale_retain_spatial_dims)} McFactor{multiscale_lowres_size_factor}'
        print(msg)

    def _init_multiscale(
        self,
        n_filters=None,
        nonlin=None,
        batchnorm=None,
        dropout=None,
        res_block_type=None,
        multiscale_retain_spatial_dims=None,
        multiscale_lowres_size_factor=None,
    ):
        self.multiscale_lowres_size_factor = multiscale_lowres_size_factor
        self.lowres_net = self.net
        if self.lowres_separate_branch:
            self.lowres_net = deepcopy(self.net)

        self.lowres_merge = MergeLowRes(
            channels=n_filters,
            merge_type='residual',
            nonlin=nonlin,
            batchnorm=batchnorm,
            dropout=dropout,
            res_block_type=res_block_type,
            multiscale_retain_spatial_dims=multiscale_retain_spatial_dims,
            multiscale_lowres_size_factor=self.multiscale_lowres_size_factor,
        )

    def forward(self, x, lowres_x=None):
        primary_flow = self.net_downsized(x)
        primary_flow = self.net(primary_flow)

        if self.enable_multiscale is False:
            assert lowres_x is None
            return primary_flow, primary_flow

        if lowres_x is not None:
            lowres_flow = self.lowres_net(lowres_x)
            merged = self.lowres_merge(primary_flow, lowres_flow)
        else:
            merged = primary_flow

        if self.multiscale_retain_spatial_dims is False:
            return merged, merged

        fac = self.multiscale_lowres_size_factor
        expected_shape = (merged.shape[-2] // fac, merged.shape[-1] // fac)
        assert merged.shape[-2:] != expected_shape
        value_to_use_in_topdown = crop_img_tensor(merged, expected_shape)
        return merged, value_to_use_in_topdown


class ResBlockWithResampling(nn.Module):
    """
    Residual block that takes care of resampling steps (each by a factor of 2).
    The mode can be top-down or bottom-up, and the block does up- and
    down-sampling by a factor of 2, respectively. Resampling is performed at
    the beginning of the block, through strided convolution.
    The number of channels is adjusted at the beginning and end of the block,
    through convolutional layers with kernel size 1. The number of internal
    channels is by default the same as the number of output channels, but
    min_inner_channels overrides this behaviour.
    Other parameters: kernel size, nonlinearity, and groups of the internal
    residual block; whether batch normalization and dropout are performed;
    whether the residual path has a gate layer at the end. There are a few
    residual block structures to choose from.
    """
    def __init__(self,
                 mode,
                 c_in,
                 c_out,
                 nonlin=nn.LeakyReLU,
                 resample=False,
                 res_block_kernel=None,
                 groups=1,
                 batchnorm=True,
                 res_block_type=None,
                 dropout=None,
                 min_inner_channels=None,
                 gated=None,
                 lowres_input=False,
                 skip_padding=False):
        super().__init__()
        assert mode in ['top-down', 'bottom-up']
        if min_inner_channels is None:
            min_inner_channels = 0
        inner_filters = max(c_out, min_inner_channels)

        # Define first conv layer to change channels and/or up/downsample
        if resample:
            if mode == 'bottom-up':  # downsample
                self.pre_conv = nn.Conv2d(in_channels=c_in,
                                          out_channels=inner_filters,
                                          kernel_size=3,
                                          padding=1,
                                          stride=2,
                                          groups=groups)
            elif mode == 'top-down':  # upsample
                self.pre_conv = nn.ConvTranspose2d(in_channels=c_in,
                                                   out_channels=inner_filters,
                                                   kernel_size=3,
                                                   padding=1,
                                                   stride=2,
                                                   groups=groups,
                                                   output_padding=1)
        elif c_in != inner_filters:
            self.pre_conv = nn.Conv2d(c_in, inner_filters, 1, groups=groups)
        else:
            self.pre_conv = None

        # Residual block
        self.res = ResidualBlock(
            channels=inner_filters,
            nonlin=nonlin,
            kernel=res_block_kernel,
            groups=groups,
            batchnorm=batchnorm,
            dropout=dropout,
            gated=gated,
            block_type=res_block_type,
            skip_padding=skip_padding,
        )
        # self.lowres_input = lowres_input
        # self.lowres_pre_conv = self.lowres_body = self.lowres_merge = None
        # if self.lowres_input:
        #     self._init_lowres(c_in=c_in, inner_filters=inner_filters, nonlin=nonlin, res_block_kernel=res_block_kernel,
        #                       groups=groups, batchnorm=batchnorm, dropout=dropout, gated=gated,
        #                       res_block_type=res_block_type)

        # Define last conv layer to get correct num output channels
        if inner_filters != c_out:
            self.post_conv = nn.Conv2d(inner_filters, c_out, 1, groups=groups)
        else:
            self.post_conv = None

    # def _init_lowres(self, c_in=None, inner_filters=None, nonlin=None, res_block_kernel=None,
    #                  groups=None, batchnorm=None, dropout=None, gated=None,
    #                  res_block_type=None):
    #     self.lowres_pre_conv = nn.Conv2d(c_in, inner_filters, 1, groups=groups)
    #     self.lowres_body = ResidualBlock(
    #         channels=inner_filters,
    #         nonlin=nonlin,
    #         kernel=res_block_kernel,
    #         groups=groups,
    #         batchnorm=batchnorm,
    #         dropout=dropout,
    #         gated=gated,
    #         block_type=res_block_type,
    #     )
    #     self.lowres_merge = MergeLowRes(
    #         channels=inner_filters,
    #         merge_type='residual',
    #         nonlin=nonlin,
    #         batchnorm=batchnorm,
    #         dropout=dropout,
    #         res_block_type=res_block_type,
    #     )

    def forward(self, x):
        if self.pre_conv is not None:
            x = self.pre_conv(x)

        x = self.res(x)
        if self.post_conv is not None:
            x = self.post_conv(x)
        return x

    # def forward(self, x, lowres=None):
    #     if self.pre_conv is not None:
    #         x = self.pre_conv(x)
    #
    #     x = self.res(x)
    #     if lowres is not None:
    #         lowres = self.lowres_pre_conv(lowres)
    #         lowres = self.lowres_body(lowres)
    #         x = self.lowres_merge(x, lowres)
    #
    #     if self.post_conv is not None:
    #         x = self.post_conv(x)
    #     return x


class TopDownDeterministicResBlock(ResBlockWithResampling):
    def __init__(self, *args, upsample=False, **kwargs):
        kwargs['resample'] = upsample
        super().__init__('top-down', *args, **kwargs)


class BottomUpDeterministicResBlock(ResBlockWithResampling):
    def __init__(self, *args, downsample=False, **kwargs):
        kwargs['resample'] = downsample
        super().__init__('bottom-up', *args, **kwargs)


class MergeLayer(nn.Module):
    """
    Merge two/more than two 4D input tensors by concatenating along dim=1 and passing the
    result through 1) a convolutional 1x1 layer, or 2) a residual block
    """
    def __init__(self,
                 channels,
                 merge_type,
                 nonlin=nn.LeakyReLU,
                 batchnorm=True,
                 dropout=None,
                 res_block_type=None,
                 res_block_kernel=None,
                 res_block_skip_padding=False):
        super().__init__()
        try:
            iter(channels)
        except TypeError:  # it is not iterable
            channels = [channels] * 3
        else:  # it is iterable
            if len(channels) == 1:
                channels = [channels[0]] * 3

        # assert len(channels) == 3

        if merge_type == 'linear':
            self.layer = nn.Conv2d(sum(channels[:-1]), channels[-1], 1)
        elif merge_type == 'residual':
            self.layer = nn.Sequential(
                nn.Conv2d(sum(channels[:-1]), channels[-1], 1, padding=0),
                ResidualGatedBlock(
                    channels[-1],
                    nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    block_type=res_block_type,
                    kernel=res_block_kernel,
                    skip_padding=res_block_skip_padding,
                ),
            )

    def forward(self, *args):
        x = torch.cat(args, dim=1)
        return self.layer(x)


class MergeLowRes(MergeLayer):
    """
    Here, we merge the lowresolution input (which has higher size)
    """
    def __init__(self, *args, **kwargs):
        self.retain_spatial_dims = kwargs.pop('multiscale_retain_spatial_dims')
        self.multiscale_lowres_size_factor = kwargs.pop('multiscale_lowres_size_factor')
        super().__init__(*args, **kwargs)

    def forward(self, latent, lowres):
        if self.retain_spatial_dims:
            latent = pad_img_tensor(latent, lowres.shape[2:])
        else:
            lh, lw = lowres.shape[-2:]
            h = lh // self.multiscale_lowres_size_factor
            w = lw // self.multiscale_lowres_size_factor
            h_pad = (lh - h) // 2
            w_pad = (lw - w) // 2
            lowres = lowres[:, :, h_pad:-h_pad, w_pad:-w_pad]

        return super().forward(latent, lowres)


class SkipConnectionMerger(MergeLayer):
    """
    By default for now simply a merge layer.
    """

    merge_type = 'residual'

    def __init__(self,
                 channels,
                 nonlin,
                 batchnorm,
                 dropout,
                 res_block_type,
                 res_block_kernel=None,
                 res_block_skip_padding=False):
        super().__init__(channels,
                         self.merge_type,
                         nonlin,
                         batchnorm,
                         dropout=dropout,
                         res_block_type=res_block_type,
                         res_block_kernel=res_block_kernel,
                         res_block_skip_padding=res_block_skip_padding)
