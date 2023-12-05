import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, in_channels=3, init_ch_count=4, num_hierarchies=4, num_initial_downsamples=0):
        super(Discriminator, self).__init__()
        self.num_initial_downsamples = num_initial_downsamples

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers += discriminator_block(in_channels, init_ch_count, normalization=False)
        last_ch_count = init_ch_count
        for _ in range(num_hierarchies - 1):
            last_ch_count = 2 * last_ch_count
            layers += discriminator_block(last_ch_count // 2, last_ch_count)

        layers += [nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(last_ch_count, 1, 4, padding=1, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, img_input):
        for i in range(self.num_initial_downsamples):
            img_input = nn.functional.avg_pool2d(img_input, 2)
        return self.model(img_input)
