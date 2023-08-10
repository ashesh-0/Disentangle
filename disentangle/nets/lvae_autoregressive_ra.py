import numpy as np
import torch
import torch.nn as nn
import os

from disentangle.analysis.lvae_utils import get_img_from_forward_output
from disentangle.core.data_split_type import DataSplitType
from disentangle.core.data_utils import crop_img_tensor
from disentangle.core.model_type import ModelType
from disentangle.nets.lvae import LadderVAE
from disentangle.nets.lvae_layers import BottomUpLayer, MergeLayer
from disentangle.nets.solutionRA_manager import SolutionRAManager
from disentangle.data_loader.patch_index_manager import GridIndexManager


class AutoRegRALadderVAE(LadderVAE):
    """
    In this variant, we feed the prediction of the upper patch into its prediction.  
    At this point, there is no extra loss which caters to smoothe prediction.
    """

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=target_ch)
        self._neighboring_encoder = None
        self._avg_pool_layers = nn.ModuleList(
            [nn.AvgPool2d(kernel_size=self.img_shape[0] // (np.power(2, i + 1))) for i in range(self.n_layers)])

        # when creating the frame prediction, we want to skip boundary.
        innerpad_amount=GridIndexManager(get_val_instance=True).get_innerpad_amount()
        self._train_sol_manager = SolutionRAManager(DataSplitType.Train, innerpad_amount,
                                                    config.data.image_size, dump_img_dir=os.path.join(config.workdir,'train_imgs'))
        self._val_sol_manager = SolutionRAManager(DataSplitType.Val, innerpad_amount,
                                                  config.data.image_size, dump_img_dir=os.path.join(config.workdir,'val_imgs'))

        nbr_count = 4
        self._merge_layers = nn.ModuleList([
            MergeLayer(
                channels=[config.model.encoder.n_filters] * (nbr_count + 2),
                merge_type=config.model.merge_type,
                nonlin=self.get_nonlin(),
                batchnorm=config.model.encoder.batchnorm,
                dropout=config.model.encoder.dropout,
                res_block_type=config.model.res_block_type,
                res_block_kernel=config.model.encoder.res_block_kernel,
            ) for _ in range(self.n_layers)
        ])
        stride = 1 if config.model.no_initial_downscaling else 2
        self._nbr_first_bottom_up_list = nn.ModuleList(
            [self.create_first_bottom_up(stride, color_ch=2) for _ in range(nbr_count)])
        self._nbr_bottom_up_layers_list = nn.ModuleList([self.create_bottomup_layers() for _ in range(nbr_count)])

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

    def forward(self, x, nbr_pred):
        img_size = x.size()[2:]

        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x)

        nbr_bu_values_list = []
        for _ in range(len(self.bottom_up_layers)):
            nbr_bu_values_list.append([])

        # get some latent space encoding for the neighboring prediction.
        for idx in range(len(nbr_pred)):
            nbr_bu_values = self._bottomup_pass(nbr_pred[idx], self._nbr_first_bottom_up_list[idx], None,
                                                self._nbr_bottom_up_layers_list[idx])
            for i in range(len(nbr_bu_values)):
                nbr_bu_values_list[i].append(nbr_bu_values[i])

        bu_values = self.bottomup_pass(x_pad)

        merged_bu_values = []

        for idx in range(len(bu_values)):
            merged_bu_values.append(self._merge_layers[idx](bu_values[idx], *nbr_bu_values_list[idx]))

        mode_layers = range(self.n_layers) if self.non_stochastic_version else None
        # Top-down inference/generation
        out, td_data = self.topdown_pass(merged_bu_values, mode_layers=mode_layers)

        if out.shape[-1] > img_size[-1]:
            # Restore original image size
            out = crop_img_tensor(out, img_size)

        return out, td_data

    def get_output_from_batch(self, batch, sol_manager=None):
        if sol_manager is None:
            sol_manager = self._val_sol_manager

        x, target, indices, grid_sizes = batch
        self.set_params_to_same_device_as(target)
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)
        nbr_preds = []
        nbr_preds.append(sol_manager.get_top(indices, grid_sizes))
        nbr_preds.append(sol_manager.get_bottom(indices, grid_sizes))
        nbr_preds.append(sol_manager.get_left(indices, grid_sizes))
        nbr_preds.append(sol_manager.get_right(indices, grid_sizes))
        nbr_preds = [torch.Tensor(nbr_y).to(x.device) for nbr_y in nbr_preds]
        out, td_data = self.forward(x_normalized, nbr_preds)
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        return {
            'out': out,
            'input_normalized': x_normalized,
            'target_normalized': target_normalized,
            'td_data': td_data
        }

    def training_step(self, batch, batch_idx, enable_logging=True):
        output_dict = self.get_output_from_batch(batch, self._train_sol_manager)
        imgs = get_img_from_forward_output(output_dict['out'],self,unnormalized=False, likelihood_obj=self.likelihood)
        self._train_sol_manager.update(imgs.cpu().detach().numpy(), batch[2], batch[3])
        return self._training_step(batch, batch_idx, output_dict, enable_logging=enable_logging)

    def validation_step(self, batch, batch_idx):
        output_dict = self.get_output_from_batch(batch, self._val_sol_manager)
        imgs = get_img_from_forward_output(output_dict['out'],self,unnormalized=False, likelihood_obj=self.likelihood)
        self._val_sol_manager.update(imgs.cpu().detach().numpy(), batch[2], batch[3])
        return self._validation_step(batch, batch_idx, output_dict)

    def on_validation_epoch_end(self):
        self._val_sol_manager.dump_img(self.data_mean.cpu().numpy(), 
                                       self.data_std.cpu().numpy(), 
                                       t=0,
                                       downscale_factor=3,
                                       epoch=self.current_epoch)
        self._train_sol_manager.dump_img(self.data_mean.cpu().numpy(), 
                                       self.data_std.cpu().numpy(), 
                                       t=0,
                                       downscale_factor=3,
                                       epoch=self.current_epoch)
        
        super().on_validation_epoch_end()
    
if __name__ == '__main__':
    import numpy as np
    import torch

    from disentangle.configs.autoregressive_config import get_config
    from disentangle.data_loader.patch_index_manager import GridAlignement, GridIndexManager
    GridIndexManager((61, 2700, 2700, 2), 1, 64, GridAlignement.LeftTop, set_train_instance=True)
    GridIndexManager((6, 2700, 2700, 2), 1, 64, GridAlignement.LeftTop, set_val_instance=True)

    config = get_config()
    config.model.skip_boundary_pixelcount = 16
    data_mean = torch.Tensor([0]).reshape(1, 1, 1, 1)
    data_std = torch.Tensor([1]).reshape(1, 1, 1, 1)
    model = AutoRegRALadderVAE(data_mean, data_std, config)
    inp = torch.rand((20, 1, config.data.image_size, config.data.image_size))
    nbr = [torch.rand((20, 2, config.data.image_size, config.data.image_size))] * 4
    out, td_data = model(inp, nbr)
    batch = (torch.rand((16, 1, config.data.image_size, config.data.image_size)),
             torch.rand((16, 2, config.data.image_size, config.data.image_size)), torch.randint(0, 100, (16, )),
             torch.Tensor(np.array([config.data.image_size] * 16)).reshape(16, ).type(torch.int32))
    model.training_step(batch, 0)
    model.validation_step(batch, 0)
    model.on_validation_epoch_end()