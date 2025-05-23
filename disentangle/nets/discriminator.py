# Taken from https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py
import torch
import torch.nn as nn


class Discriminator(torch.nn.Module):
    def __init__(self, channels, first_out_channel=128):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self._out_C = first_out_channel*4
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=first_out_channel, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(first_out_channel, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            # nn.Conv2d(in_channels=first_out_channel, out_channels=first_out_channel*2, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(first_out_channel*2, affine=True),
            # nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=first_out_channel, out_channels=self._out_C, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self._out_C, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=self._out_C, out_channels=1, kernel_size=2, stride=1, padding=0)
            # tanh
            # nn.Tanh()
            )
        print(f'{self.__class__.__name__} initialized with {self._out_C} channels')


    def forward(self, x):
        # print(x.shape, 'before discriminator')
        x = self.main_module(x)
        # print(x.shape, 'after discriminator')
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, self._out_C*4*4)


class LatentDiscriminator(nn.Module):
    def __init__(self, discriminator, embedding_network):
        super().__init__()
        self.embedding_network = embedding_network
        self.D = discriminator
    
    def forward(self, x):
        # print(x.shape, 'before embedding')
        x = self.embedding_network.get_embedding(x)
        # print(x.shape, 'after embedding')   
        return self.D(x)