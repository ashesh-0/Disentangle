"""
Taken from https://github.com/juglab/HDN/blob/e30edf7ec2cd55c902e469b890d8fe44d15cbb7e/lib/nn.py
"""
import torch
import torchvision.transforms.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutional layers.
    Input, intermediate, and output channels are the same. Padding is always
    'same'. The 2 convolutional layers have the same groups. No stride allowed,
    and kernel sizes have to be odd.
    The result is:
        out = gate(f(x)) + x
    where an argument controls the presence of the gating mechanism, and f(x)
    has different structures depending on the argument block_type.
    block_type is a string specifying the structure of the block, where:
        a = activation
        b = batch norm
        c = conv layer
        d = dropout.
    For example, bacdbacd has 2x (batchnorm, activation, conv, dropout).
    """

    default_kernel_size = (3, 3)

    def __init__(self,
                 channels: int,
                 nonlin,
                 kernel=None,
                 mode_3D=False,
                 groups=1,
                 batchnorm: bool = True,
                 block_type: str = None,
                 dropout=None,
                 gated=None,
                 skip_padding=False,
                 conv2d_bias=True):
        super().__init__()
        if kernel is None:
            kernel = self.default_kernel_size
        elif isinstance(kernel, int):
            kernel = (kernel, kernel) if not mode_3D else (kernel, kernel, kernel)
        elif len(kernel) not in [2,3]:
            raise ValueError("kernel has to be None, int, or an iterable of length 2")
        assert all([k % 2 == 1 for k in kernel]), "kernel sizes have to be odd"
        kernel = list(kernel)
        self.skip_padding = skip_padding
        pad = [0] * len(kernel) if self.skip_padding else [k // 2 for k in kernel]
        print(kernel, pad)
        self.gated = gated
        modules = []

        if mode_3D:
            conv_cls = nn.Conv3d
            batchnorm_cls = nn.BatchNorm3d
            dropout_cls = nn.Dropout3d
        else:
            conv_cls = nn.Conv2d
            batchnorm_cls = nn.BatchNorm2d
            dropout_cls = nn.Dropout2d

        if block_type == 'cabdcabd':
            for i in range(2):
                conv = conv_cls(channels, channels, kernel[i], padding=pad[i], groups=groups, bias=conv2d_bias)
                modules.append(conv)
                modules.append(nonlin())
                if batchnorm:
                    modules.append(batchnorm_cls(channels))
                if dropout is not None:
                    modules.append(dropout_cls(dropout))

        elif block_type == 'bacdbac':
            for i in range(2):
                if batchnorm:
                    modules.append(batchnorm_cls(channels))
                modules.append(nonlin())
                conv = conv_cls(channels, channels, kernel[i], padding=pad[i], groups=groups, bias=conv2d_bias)
                modules.append(conv)
                if dropout is not None and i == 0:
                    modules.append(dropout_cls(dropout))

        elif block_type == 'bacdbacd':
            for i in range(2):
                if batchnorm:
                    modules.append(batchnorm_cls(channels))
                modules.append(nonlin())
                conv = conv_cls(channels, channels, kernel[i], padding=pad[i], groups=groups, bias=conv2d_bias)
                modules.append(conv)
                modules.append(dropout_cls(dropout))

        else:
            raise ValueError("unrecognized block type '{}'".format(block_type))

        if gated:
            modules.append(GateLayer(channels, 1, mode_3D=mode_3D, nonlin=nonlin))
        self.block = nn.Sequential(*modules)

    def forward(self, x):

        out = self.block(x)
        if out.shape != x.shape:
            return out + F.center_crop(x, out.shape[-2:])
        else:
            return out + x


class ResidualGatedBlock(ResidualBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, gated=True)


class GateLayer(nn.Module):
    """
    Double the number of channels through a convolutional layer, then use
    half the channels as gate for the other half.
    """
    def __init__(self, channels, kernel_size, mode_3D=False, nonlin=nn.LeakyReLU):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        if mode_3D:
            conv_cls = nn.Conv3d
        else:
            conv_cls = nn.Conv2d
        self.conv = conv_cls(channels, 2 * channels, kernel_size, padding=pad)
        self.nonlin = nonlin()

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = self.nonlin(x)  # TODO remove this?
        gate = torch.sigmoid(gate)
        return x * gate
