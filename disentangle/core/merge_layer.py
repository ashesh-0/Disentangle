import torch
import torch.nn as nn

from disentangle.core.data_utils import pad_img_tensor
from disentangle.core.nn_submodules import ResidualBlock, ResidualGatedBlock


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
                 conv2d_bias=True,
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
            self.layer = nn.Conv2d(sum(channels[:-1]), channels[-1], 1, bias=conv2d_bias)
        elif merge_type == 'residual':
            self.layer = nn.Sequential(
                nn.Conv2d(sum(channels[:-1]), channels[-1], 1, padding=0, bias=conv2d_bias),
                ResidualGatedBlock(
                    channels[-1],
                    nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    block_type=res_block_type,
                    kernel=res_block_kernel,
                    conv2d_bias=conv2d_bias,
                    skip_padding=res_block_skip_padding,
                ),
            )
        elif merge_type == 'residual_ungated':
            self.layer = nn.Sequential(
                nn.Conv2d(sum(channels[:-1]), channels[-1], 1, padding=0, bias=conv2d_bias),
                ResidualBlock(
                    channels[-1],
                    nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    block_type=res_block_type,
                    kernel=res_block_kernel,
                    conv2d_bias=conv2d_bias,
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

    def __init__(self,
                 channels,
                 nonlin,
                 batchnorm,
                 dropout,
                 res_block_type,
                 merge_type='residual',
                 conv2d_bias: bool = True,
                 res_block_kernel=None,
                 res_block_skip_padding=False):
        super().__init__(channels,
                         merge_type,
                         nonlin,
                         batchnorm,
                         dropout=dropout,
                         res_block_type=res_block_type,
                         res_block_kernel=res_block_kernel,
                         conv2d_bias=conv2d_bias,
                         res_block_skip_padding=res_block_skip_padding)
