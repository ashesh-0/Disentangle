"""
This file defines the unet architecture.
"""

import numpy as np
import torch.nn as nn
import torch


def get_activation(activation_str):
    if activation_str == 'relu':
        return nn.ReLU()


def conv_block(in_channels, out_channels, kernel_size, strides, padding, activation, dropout, bn):
    modules = []
    modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=strides, padding=padding))
    modules.append(get_activation(activation))
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    modules.append(nn.Dropout(p=dropout))
    modules.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride=strides, padding=padding))
    modules.append(get_activation(activation))
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*modules)


def down_scale_path(num_kernels, kernel_size, strides, padding, activation, dropout, bn):
    """
    define bottom up layers
    """
    blocks = nn.ModuleList([])
    input_ch_N = 1
    for ch_N in num_kernels:
        blocks.append(conv_block(input_ch_N, ch_N, kernel_size, strides, padding, activation, dropout, bn))
        input_ch_N = ch_N
    return blocks


def convolution_layer(in_channels,
                      out_channels,
                      kernel_size,
                      stride=1,
                      padding=0,
                      activation=None,
                      dropout=0.0,
                      bn=True):
    branch = []
    branch.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))
    branch.append(get_activation(activation))
    if bn:
        branch.append(nn.BatchNorm2d(1))
    branch.append(nn.Dropout(p=dropout))

    return nn.Sequential(*branch)


def lowres_output_branches(num_kernels, final_activation, dropout):
    blocks = nn.ModuleList([])
    N = len(num_kernels)
    for i in range(N - 2):
        branch = convolution_layer(
            num_kernels[N - i],
            1,
            1,
            stride=1,
            padding=0,  #TODO: check
            activation=final_activation,
            dropout=dropout,
            bn=False)  #TODO: check this
        blocks.append(branch)
    return blocks


def up_scale_path(num_kernels, kernel_size, strides, padding, activation, dropout, bn):
    blocks = nn.ModuleList([])
    input_ch_N = num_kernels[-1]
    for i in range(len(num_kernels) - 1):
        blocks.append(
            conv_block(input_ch_N, num_kernels[len(num_kernels) - i - 2], kernel_size, strides, padding, activation,
                       dropout, bn))
    return blocks


class BraveNet(nn.Module):
    def __init__(self, num_kernels, kernel_size, strides, padding, activation, dropout, bn):
        super().__init__()
        self.num_kernels = num_kernels
        self.bottom_up_layers = down_scale_path(num_kernels, kernel_size, strides, padding, activation, dropout, bn)
        final_activation = None
        self.lowres_output_branches = lowres_output_branches(num_kernels, final_activation, dropout, bn)
        self.output_branch = convolution_layer(num_kernels[0],
                                               1,
                                               1,
                                               strides=1,
                                               activation=final_activation,
                                               dropout=dropout,
                                               padding=0,
                                               bn=False)
        self.num_kernels = num_kernels
        self.top_down_layers = up_scale_path(num_kernels, kernel_size, strides, padding, activation, dropout, bn)

    def bottom_up(self, input):
        residuals = {}
        conv_down = input
        for i, k in enumerate(self.num_kernels):
            # level i
            conv_down = self.bottom_up_layers[i](conv_down)
            residuals["conv_" + str(i)] = conv_down
            if i < len(self.num_kernels) - 1:
                conv_down = nn.MaxPool2d(2, stride=2)(conv_down)

        return conv_down, residuals

    def top_down(self, bu_output, residuals):
        outputs = []
        conv_up = bu_output
        for i in range(len(self.num_kernels) - 1):
            conv_up = nn.Upsample(scale_factor=2, mode='nearest')(conv_up)
            bu_tensor = residuals["conv_" + str(len(self.num_kernels) - i - 2)]
            conv_up = torch.cat([conv_up, bu_tensor], dim=1)
            conv_up = self.top_down_layers[i](conv_up)
            if i < len(self.num_kernels) - 2:
                temp_output = nn.Upsample(output_dim, mode='nearest')(conv_up)
                temp_output = self.lowres_output_branches[i](temp_output)
                outputs.append(temp_output)

        output = self.output_branch(conv_up)
        outputs.append(output)
        return outputs
