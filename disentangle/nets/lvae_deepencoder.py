from copy import deepcopy

import ml_collections
from disentangle.nets.lvae import LadderVAE
from disentangle.nets.lvae_twindecoder import LadderVAETwinDecoder


class LVAEWithDeepEncoder(LadderVAETwinDecoder):

    def __init__(self, data_mean, data_std, config):
        new_config = deepcopy(config)
        with new_config.unlocked():
            new_config.data.color_ch = config.model.encoder.n_filters
            new_config.data.multiscale_lowres_count = None  # multiscaleing is inside the extra encoder.
        super().__init__(data_mean, data_std, new_config)

        with config.unlocked():
            config.model.decoder.batchnorm = True
            config.model.encoder.batchnorm = True
            config.model.decoder.conv2d_bias = True
            config.model.non_stochastic_version = True
        self.extra_encoder = LadderVAE(data_mean, data_std, config, target_ch=config.encoder.n_filters)

    def forward(self, x):
        encoded, _ = self.extra_encoder(x)
        return super().forward(encoded)


if __name__ == '__main__':
    import torch

    from disentangle.configs.microscopy_multi_channel_lvae_config import get_config
    config = get_config()
    data_mean = torch.Tensor([0]).reshape(1, 1, 1, 1)
    data_std = torch.Tensor([1]).reshape(1, 1, 1, 1)
    model = LVAEWithDeepEncoder(data_mean, data_std, config)
    mc = 1 if config.data.multiscale_lowres_count is None else config.data.multiscale_lowres_count + 1
    inp = torch.rand((2, mc, 64, 64))
    out, _ = model(inp)
