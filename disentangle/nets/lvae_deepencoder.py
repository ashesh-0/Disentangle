from copy import deepcopy

import ml_collections
from disentangle.nets.lvae import LadderVAE
from disentangle.nets.lvae_twindecoder import LadderVAETwinDecoder


class LVAEWithDeepEncoder(LadderVAETwinDecoder):

    def __init__(self, data_mean, data_std, config):
        config = ml_collections.ConfigDict(config)
        new_config = deepcopy(config)
        with new_config.unlocked():
            new_config.data.color_ch = config.model.encoder.n_filters
            new_config.data.multiscale_lowres_count = None  # multiscaleing is inside the extra encoder.
            new_config.model.gated = False
            new_config.model.decoder.dropout = 0.0
            new_config.model.merge_type = 'residual_ungated'
        super().__init__(data_mean, data_std, new_config)

        with config.unlocked():
            config.model.decoder.batchnorm = True
            config.model.encoder.batchnorm = True
            config.model.decoder.conv2d_bias = True
            config.model.non_stochastic_version = True
        self.extra_encoder = LadderVAE(data_mean, data_std, config, target_ch=config.model.encoder.n_filters)

    def forward(self, x):
        encoded, _ = self.extra_encoder(x)
        return super().forward(encoded)


if __name__ == '__main__':
    import torch

    from disentangle.configs.deepencoder_lvae_config import get_config
    config = get_config()
    data_mean = torch.Tensor([0]).reshape(1, 1, 1, 1)
    data_std = torch.Tensor([1]).reshape(1, 1, 1, 1)
    model = LVAEWithDeepEncoder(data_mean, data_std, config)
    mc = 1 if config.data.multiscale_lowres_count is None else config.data.multiscale_lowres_count + 1
    inp = torch.rand((2, mc, 64, 64))
    out1, out2, td_data = model(inp)
    print(out1.shape, out2.shape)
    print(td_data)
    # decoder invariance.
    bu_values_l1 = []
    for i in range(1, config.data.multiscale_lowres_count + 1):
        isz = config.data.image_size
        z = config.model.encoder.n_filters
        pow = 2**(i)
        bu_values_l1.append(torch.rand(2, z // 2, isz // pow, isz // pow))

    out_l1_1x, _ = model.topdown_pass(
        bu_values_l1,
        top_down_layers=model.top_down_layers_l1,
        final_top_down_layer=model.final_top_down_l1,
    )

    out_l1_10x, _ = model.topdown_pass(
        [10 * x for x in bu_values_l1],
        top_down_layers=model.top_down_layers_l1,
        final_top_down_layer=model.final_top_down_l1,
    )
    # out_l1_1x = model.top_down_layers_l1[0](None, bu_value=bu_values_l1[0], inference_mode=True,use_mode=True)
    # out_l1_10x = model.top_down_layers_l1[0](None, bu_value=10*bu_values_l1[0], inference_mode=True,use_mode=True)

    max_diff = torch.abs(out_l1_1x * 10 - out_l1_10x).max().item()
    assert max_diff < 1e-5
