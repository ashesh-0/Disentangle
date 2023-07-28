import numpy as np
import torch
import torch.nn as nn

from disentangle.analysis.lvae_utils import get_img_from_forward_output
from disentangle.core.data_utils import crop_img_tensor
from disentangle.nets.lvae import LadderVAE
from disentangle.nets.lvae_layers import BottomUpLayer, MergeLayer


class AutoRegLadderVAE(LadderVAE):
    """
    In this variant, we feed the prediction of the upper patch into its prediction.  
    At this point, there is no extra loss which caters to smoothe prediction.
    """

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=target_ch)
        self._neighboring_encoder = None
        self._avg_pool_layers = nn.ModuleList(
            [nn.AvgPool2d(kernel_size=self.img_shape[0] // (np.power(2, i + 1))) for i in range(self.n_layers)])
        self._merge_layers = nn.ModuleList([
            MergeLayer(
                channels=config.model.encoder.n_filters,
                merge_type=config.model.merge_type,
                nonlin=self.get_nonlin(),
                batchnorm=config.model.encoder.batchnorm,
                dropout=config.model.encoder.dropout,
                res_block_type=config.model.res_block_type,
                res_block_kernel=config.model.encoder.res_block_kernel,
            ) for _ in range(self.n_layers)
        ])
        stride = 1 if config.model.no_initial_downscaling else 2
        self._nbr_first_bottom_up = self.create_first_bottom_up(stride, color_ch=2)
        self._nbr_bottom_up_layers = self.create_bottomup_layers()

    def create_bottomup_layers(self):
        nbr_bottom_up_layers = []
        for i in range(self.n_layers):
            nbr_bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.encoder_blocks_per_layer,
                    n_filters=self.encoder_n_filters,
                    downsampling_steps=self.downsample[i],
                    nonlin=self.get_nonlin(),
                    batchnorm=self.bottomup_batchnorm,
                    dropout=self.encoder_dropout,
                    res_block_type=self.res_block_type,
                    res_block_kernel=self.encoder_res_block_kernel,
                    res_block_skip_padding=self.encoder_res_block_skip_padding,
                    gated=self.gated,
                    enable_multiscale=False,
                ))
        return nn.ModuleList(nbr_bottom_up_layers)

    def forward(self, x):
        img_size = x.size()[2:]

        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x)

        # Now, process the neighboring patch.
        # with torch.no_grad():
        nbr_pred, *_ = super().forward(x_pad[:, 1:])

        # get the prediction for neighboring image.
        nbr_pred = get_img_from_forward_output(nbr_pred, self, unnormalized=False)
        # get some latent space encoding for the neighboring prediction.
        nbr_bu_values = self._bottomup_pass(nbr_pred, self._nbr_first_bottom_up, None, self._nbr_bottom_up_layers)
        nbr_bu_values = [nbr_bu_values[i].detach() for i in range(len(nbr_bu_values))]

        assert x_pad.shape[1] == 2, ' We must have exactly one neighbor'
        bu_values = self.bottomup_pass(x_pad[:, :1])

        merged_bu_values = []
        for idx in range(len(bu_values)):
            merged_bu_values.append(self._merge_layers[idx](bu_values[idx], nbr_bu_values[idx]))

        mode_layers = range(self.n_layers) if self.non_stochastic_version else None
        # Top-down inference/generation
        out, td_data = self.topdown_pass(merged_bu_values, mode_layers=mode_layers)

        if out.shape[-1] > img_size[-1]:
            # Restore original image size
            out = crop_img_tensor(out, img_size)

        return out, td_data


if __name__ == '__main__':
    import numpy as np
    import torch

    from disentangle.configs.twotiff_config import get_config

    config = get_config()
    data_mean = torch.Tensor([0]).reshape(1, 1, 1, 1)
    data_std = torch.Tensor([1]).reshape(1, 1, 1, 1)
    model = AutoRegLadderVAE(data_mean, data_std, config)
    inp = torch.rand((20, 2, config.data.image_size, config.data.image_size))
    out, td_data = model(inp)
    print(out.shape)
    batch = (
        torch.rand((16, 2, config.data.image_size, config.data.image_size)),
        torch.rand((16, 2, config.data.image_size, config.data.image_size)),
    )
    model.training_step(batch, 0)
    model.validation_step(batch, 0)

    ll = torch.ones((12, 2, 32, 32))
    ll_new = model._get_weighted_likelihood(ll)
    print(ll_new[:, 0].mean(), ll_new[:, 0].std())
    print(ll_new[:, 1].mean(), ll_new[:, 1].std())
    print('mar')
