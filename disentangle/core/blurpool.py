"""
Code adapted from https://github.com/adobe/antialiased-cnns/blob/b27a34a26f3ab039113d44d83c54d0428598ac9c/antialiased_cnns/blurpool.py
"""
# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='zero', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        if self.filt_size == 1:
            a = np.array([1., ])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class UnAliasedMaxPool(BlurPool):
    """
    BlurPool() => Max(with stride 1)
    """

    def __init__(self, channels, pad_type='zero', filt_size=4, stride=2, pad_off=0):
        super(UnAliasedMaxPool, self).__init__(channels, pad_type=pad_type, filt_size=filt_size, stride=stride,
                                               pad_off=pad_off)
        self.max = nn.MaxPool2d(stride, stride=1)

    def forward(self, inp):
        out = self.max(inp)
        return super().forward(out)


class UnAliasedStridedConv(BlurPool):
    """
    BlurPool => ReLU => Conv(with no stride)
    """

    def __init__(self, in_channels, out_channels, kernel_size, nonlin=None, padding=None, bp_pad_type='zero',
                 bp_filt_size=1,
                 stride=2,
                 bp_pad_off=0,
                 groups=1):
        super(UnAliasedStridedConv, self).__init__(out_channels, pad_type=bp_pad_type, filt_size=bp_filt_size,
                                                   stride=stride,
                                                   pad_off=bp_pad_off)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, groups=groups, padding=padding)
        self.nonlin = nonlin()

    def forward(self, inp):
        out = self.conv(inp)
        if self.nonlin is not None:
            out = self.nonlin(out)
        return super().forward(out)


if __name__ == '__main__':
    tensor1 = torch.Tensor([[0, 1, 1, 0, 0, 1, 1, 0]] * 4)[None, None]
    tensor2 = torch.Tensor([[1, 1, 0, 0, 1, 1, 0, 0]] * 4)[None, None]
    bp = UnAliasedMaxPool(1, pad_type='zero', filt_size=2)
    out1 = bp(tensor1)
    out2 = bp(tensor2)

    nonlin = nn.LeakyReLU
    kernel = 3
    convlayer = UnAliasedStridedConv(1, 32, kernel, nonlin=nonlin, padding=kernel // 2, bp_filt_size=3)
    out3 = convlayer(tensor1)
