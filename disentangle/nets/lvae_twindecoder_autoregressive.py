import numpy as np
import torch
import torch.nn as nn

from disentangle.analysis.lvae_utils import get_img_from_forward_output
from disentangle.core.data_utils import crop_img_tensor
from disentangle.nets.lvae_layers import BottomUpLayer, MergeLayer
from disentangle.nets.lvae_twindecoder import LadderVAETwinDecoder


class AutoRegTwinDecoderLadderVAE(LadderVAETwinDecoder):
    """
    In this variant, we feed the prediction of the upper patch into its prediction.  
    At this point, there is no extra loss which caters to smoothe prediction.
    """

    def __init__(self, data_mean, data_std, config):
        super().__init__(data_mean, data_std, config)

        self._merge_layers_c1 = nn.ModuleList([
            MergeLayer(
                channels=config.model.encoder.n_filters // 2,
                merge_type=config.model.merge_type,
                nonlin=self.get_nonlin(),
                batchnorm=config.model.encoder.batchnorm,
                dropout=config.model.encoder.dropout,
                res_block_type=config.model.res_block_type,
                res_block_kernel=config.model.encoder.res_block_kernel,
            ) for _ in range(self.n_layers)
        ])

        self._merge_layers_c2 = nn.ModuleList([
            MergeLayer(
                channels=config.model.encoder.n_filters // 2,
                merge_type=config.model.merge_type,
                nonlin=self.get_nonlin(),
                batchnorm=config.model.encoder.batchnorm,
                dropout=config.model.encoder.dropout,
                res_block_type=config.model.res_block_type,
                res_block_kernel=config.model.encoder.res_block_kernel,
            ) for _ in range(self.n_layers)
        ])

        stride = 1 if config.model.no_initial_downscaling else 2
        self._nbr_first_bottom_up_c1 = self.create_first_bottom_up(stride,
                                                                   color_ch=1,
                                                                   encoder_n_filters=self.encoder_n_filters // 2)
        self._nbr_bottom_up_layers_c1 = self.create_bottomup_layers()

        self._nbr_first_bottom_up_c2 = self.create_first_bottom_up(stride,
                                                                   color_ch=1,
                                                                   encoder_n_filters=self.encoder_n_filters // 2)
        self._nbr_bottom_up_layers_c2 = self.create_bottomup_layers()

    def create_bottomup_layers(self):
        nbr_bottom_up_layers = []
        for i in range(self.n_layers):
            nbr_bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.encoder_blocks_per_layer,
                    n_filters=self.encoder_n_filters // 2,
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

        #
        # Process the neighboring patch.
        #
        nbr_pred, *_ = super().forward(x_pad[:, 1:])
        # get the prediction for neighboring image.
        nbr_pred_c1 = get_img_from_forward_output(nbr_pred, self, likelihood_obj=self.likelihood_l1, unnormalized=False)
        nbr_pred_c2 = get_img_from_forward_output(nbr_pred, self, likelihood_obj=self.likelihood_l1, unnormalized=False)
        nbr_pred = torch.cat([nbr_pred_c1, nbr_pred_c2], dim=1)

        # get some latent space encoding for the neighboring prediction.
        nbr_bu_values_c1 = self._bottomup_pass(nbr_pred[:, :1], self._nbr_first_bottom_up_c1, None,
                                               self._nbr_bottom_up_layers_c1)
        nbr_bu_values_c1 = [nbr_bu_values_c1[i].detach() for i in range(len(nbr_bu_values_c1))]
        nbr_bu_values_c2 = self._bottomup_pass(nbr_pred[:, 1:], self._nbr_first_bottom_up_c2, None,
                                               self._nbr_bottom_up_layers_c2)
        nbr_bu_values_c2 = [nbr_bu_values_c2[i].detach() for i in range(len(nbr_bu_values_c2))]

        assert x_pad.shape[1] == 2, ' We must have exactly one neighbor'
        bu_values = self.bottomup_pass(x_pad[:, :1])
        bu_values_c1, bu_values_c2 = self.get_separate_bu_values(bu_values)

        merged_bu_values_c1 = []
        merged_bu_values_c2 = []
        for idx in range(len(bu_values)):
            merged_bu_values_c1.append(self._merge_layers_c1[idx](bu_values_c1[idx], nbr_bu_values_c1[idx]))
            merged_bu_values_c2.append(self._merge_layers_c2[idx](bu_values_c2[idx], nbr_bu_values_c2[idx]))

        # Top-down inference/generation
        out_l1, td_data_l1 = self.topdown_pass(
            merged_bu_values_c1,
            top_down_layers=self.top_down_layers_l1,
            final_top_down_layer=self.final_top_down_l1,
        )
        out_l2, td_data_l2 = self.topdown_pass(
            merged_bu_values_c2,
            top_down_layers=self.top_down_layers_l2,
            final_top_down_layer=self.final_top_down_l2,
        )

        # Restore original image size
        out_l1 = crop_img_tensor(out_l1, img_size)
        out_l2 = crop_img_tensor(out_l2, img_size)

        td_data = {
            'z': [torch.cat([td_data_l1['z'][i], td_data_l2['z'][i]], dim=1) for i in range(len(td_data_l1['z']))],
        }
        if td_data_l2['kl'][0] is not None:
            td_data['kl'] = [(td_data_l1['kl'][i] + td_data_l2['kl'][i]) / 2 for i in range(len(td_data_l1['kl']))]
        return out_l1, out_l2, td_data


if __name__ == '__main__':
    import numpy as np
    import torch

    from disentangle.configs.microscopy_multi_channel_lvae_config import get_config

    config = get_config()
    data_mean = torch.Tensor([0]).reshape(1, 1, 1, 1)
    data_std = torch.Tensor([1]).reshape(1, 1, 1, 1)
    model = AutoRegTwinDecoderLadderVAE(data_mean, data_std, config)
    inp = torch.rand((20, 2, config.data.image_size, config.data.image_size))
    out1, out2, td_data = model(inp)
    # print(out.shape)
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
