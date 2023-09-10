import os

import numpy as np
import torch
import torch.nn as nn

from disentangle.analysis.lvae_utils import get_img_from_forward_output
from disentangle.core.data_split_type import DataSplitType
from disentangle.core.data_utils import crop_img_tensor
from disentangle.core.model_type import ModelType
from disentangle.core.psnr import RangeInvariantPsnr
from disentangle.data_loader.patch_index_manager import GridIndexManager
from disentangle.nets.lvae import LadderVAE
from disentangle.nets.lvae_layers import BottomUpLayer, MergeLayer
from disentangle.nets.solutionRA_manager import SolutionRAManager


class Neighbors:
    """
    It enables rotation of tensors (B,C,H,W)=> HW will be rotated.
    """

    def __init__(self, top, bottom, left, right) -> None:
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def rotate90anticlock(self, k):
        k = k % 4
        if k == 0:
            return
        arr = [self.top, self.left, self.bottom, self.right]
        nbr_preds = [torch.rot90(inp, k=k, dims=(2, 3)) for inp in arr]
        nbr_preds = nbr_preds[-k:] + nbr_preds[:4 - k]
        self.top, self.left, self.bottom, self.right = nbr_preds

    def get(self):
        return [self.top, self.bottom, self.left, self.right]


class AutoRegRALadderVAE(LadderVAE):
    """
    In this variant, we feed the prediction of the upper patch into its prediction.  
    At this point, there is no extra loss which caters to smoothe prediction.
    """

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=target_ch)
        self._neighboring_encoder = None
        self._enable_rotation = config.model.get('rotation_with_neighbors', False)
        self._untrained_nbr_branch = config.model.get('untrained_nbr_branch', False)
        self._avg_pool_layers = nn.ModuleList(
            [nn.AvgPool2d(kernel_size=self.img_shape[0] // (np.power(2, i + 1))) for i in range(self.n_layers)])

        self._nbr_share_weights = config.model.get('nbr_share_weights', False)

        # when creating the frame prediction, we want to skip boundary.
        innerpad_amount = GridIndexManager(get_val_instance=True).get_innerpad_amount()
        self._train_sol_manager = SolutionRAManager(DataSplitType.Train,
                                                    innerpad_amount,
                                                    config.data.image_size,
                                                    dump_img_dir=os.path.join(config.workdir, 'train_imgs'),
                                                    dropout=config.model.get('nbr_dropout', 0.0))
        self._val_sol_manager = SolutionRAManager(DataSplitType.Val,
                                                  innerpad_amount,
                                                  config.data.image_size,
                                                  dump_img_dir=os.path.join(config.workdir, 'val_imgs'))
        # save the groundtruth
        self._val_gt_manager = SolutionRAManager(DataSplitType.Val,
                                                 innerpad_amount,
                                                 config.data.image_size,
                                                 dump_img_dir=os.path.join(config.workdir, 'val_groundtruth'))

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
        if self._nbr_share_weights:
            self._nbr_first_bottom_up = self.create_first_bottom_up(stride, color_ch=2)
            self._nbr_bottom_up_layers = self.create_bottomup_layers()
            self._nbr_first_bottom_up_list = [self._nbr_first_bottom_up for _ in range(nbr_count)]
            self._nbr_bottom_up_layers_list = [self._nbr_bottom_up_layers for _ in range(nbr_count)]
        else:
            self._nbr_first_bottom_up_list = nn.ModuleList(
                [self.create_first_bottom_up(stride, color_ch=2) for _ in range(nbr_count)])
            self._nbr_bottom_up_layers_list = nn.ModuleList([self.create_bottomup_layers() for _ in range(nbr_count)])
        print(f'[{self.__class__.__name__}]Rotation:{self._enable_rotation} NbrSharedWeights:{self._nbr_share_weights}')

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

    @staticmethod
    def get_mask(spatial_dim, orientation: str, device):
        mask = torch.arange(0, spatial_dim) / (spatial_dim - 1)
        if orientation == 'top':
            mask = mask.reshape(1, 1, -1, 1).to(device)
        elif orientation == 'bottom':
            mask = torch.flip(mask, [0])
            mask = mask.reshape(1, 1, -1, 1).to(device)
        elif orientation == 'left':
            mask = mask.reshape(1, 1, 1, -1).to(device)
        elif orientation == 'right':
            mask = torch.flip(mask, [0])
            mask = mask.reshape(1, 1, 1, -1).to(device)
        else:
            raise ValueError(f"orientation {orientation} not recognized")

        mask[mask < 0.95] = 0
        return mask

    def forward(self, x, nbr_pred):
        img_size = x.size()[2:]

        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x)

        nbr_bu_values_list = []
        for _ in range(len(self.bottom_up_layers)):
            nbr_bu_values_list.append([])

        # get some latent space encoding for the neighboring prediction.
        # top, bottom, left, right
        assert len(nbr_pred) == 4

        nbr_bu_values_top = self._bottomup_pass(nbr_pred[0], self._nbr_first_bottom_up_list[0], None,
                                                self._nbr_bottom_up_layers_list[0])
        shapes = [x.shape[-2:] for x in nbr_bu_values_top]
        assert all([x[0] == x[1] for x in shapes])

        nbr_bu_values_top = [x[:, :, -1:] for x in nbr_bu_values_top]
        # import pdb;pdb.set_trace()
        nbr_bu_values_top = [x * self.get_mask(shapes[i][0], 'top', x.device) for i, x in enumerate(nbr_bu_values_top)]
        # nbr_bu_values_top = [x.repeat(1, 1, x.shape[3], 1) for x in nbr_bu_values_top]

        nbr_bu_values_bottom = self._bottomup_pass(nbr_pred[1], self._nbr_first_bottom_up_list[1], None,
                                                   self._nbr_bottom_up_layers_list[1])

        nbr_bu_values_bottom = [x[:, :, :1] for x in nbr_bu_values_bottom]
        nbr_bu_values_bottom = [
            x * self.get_mask(shapes[i][0], 'bottom', x.device) for i, x in enumerate(nbr_bu_values_bottom)
        ]

        nbr_bu_values_left = self._bottomup_pass(nbr_pred[2], self._nbr_first_bottom_up_list[2], None,
                                                 self._nbr_bottom_up_layers_list[2])
        nbr_bu_values_left = [x[..., -1:] for x in nbr_bu_values_left]
        nbr_bu_values_left = [
            x * self.get_mask(shapes[i][0], 'left', x.device) for i, x in enumerate(nbr_bu_values_left)
        ]

        nbr_bu_values_right = self._bottomup_pass(nbr_pred[3], self._nbr_first_bottom_up_list[3], None,
                                                  self._nbr_bottom_up_layers_list[3])
        nbr_bu_values_right = [x[..., :1] for x in nbr_bu_values_right]
        nbr_bu_values_right = [
            x * self.get_mask(shapes[i][0], 'right', x.device) for i, x in enumerate(nbr_bu_values_right)
        ]
        # nbr_bu_values_right = [x.repeat(1, 1, 1, x.shape[2]) for x in nbr_bu_values_right]

        nbr_bu_values_list = list(zip(nbr_bu_values_top, nbr_bu_values_bottom, nbr_bu_values_left, nbr_bu_values_right))

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

    def get_output_from_batch(self, batch, sol_manager=None, enable_rotation=False):
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
        nbrs = Neighbors(*nbr_preds)
        nbr_preds = nbrs.get()

        if enable_rotation:
            quadrant = np.random.randint(0, 4)
            if quadrant > 0:
                x_normalized = torch.rot90(x_normalized, k=quadrant, dims=(2, 3))
                target_normalized = torch.rot90(target_normalized, k=quadrant, dims=(2, 3))
                nbrs.rotate90anticlock(quadrant)
                nbr_preds = nbrs.get()

        out, td_data = self.forward(x_normalized, nbr_preds)
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        del nbrs
        return {
            'out': out,
            'input_normalized': x_normalized,
            'target_normalized': target_normalized,
            'td_data': td_data,
            'quadrant': quadrant if enable_rotation else None,
        }

    def training_step(self, batch, batch_idx, enable_logging=True):
        output_dict = self.get_output_from_batch(batch, self._train_sol_manager, enable_rotation=self._enable_rotation)
        imgs = get_img_from_forward_output(output_dict['out'], self, unnormalized=False, likelihood_obj=self.likelihood)

        if output_dict['quadrant'] is not None and output_dict['quadrant'] > 0:
            imgs = torch.rot90(imgs, k=-output_dict['quadrant'], dims=(2, 3))

        self._train_sol_manager.update(imgs.cpu().detach().numpy(), batch[2], batch[3])
        # in case of rotation, batch is invalid. since this is oly done in training,
        # None is being passed for batch in training and not in validation
        return self._training_step(None, batch_idx, output_dict, enable_logging=enable_logging)

    def validation_step(self, batch, batch_idx, return_output_dict=False):
        output_dict = self.get_output_from_batch(batch, self._val_sol_manager)
        imgs = get_img_from_forward_output(output_dict['out'], self, unnormalized=False, likelihood_obj=self.likelihood)
        self._val_sol_manager.update(imgs.cpu().detach().numpy(), batch[2], batch[3])
        self._val_gt_manager.update(output_dict['target_normalized'].cpu().detach().numpy(), batch[2], batch[3])
        val_out = self._validation_step(batch, batch_idx, output_dict)
        if return_output_dict:
            return val_out, output_dict

        return val_out

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
        if self.current_epoch == 0:
            self._val_gt_manager.dump_img(self.data_mean.cpu().numpy(),
                                          self.data_std.cpu().numpy(),
                                          t=0,
                                          downscale_factor=3,
                                          epoch=self.current_epoch)

        # PSNR
        psnr1 = RangeInvariantPsnr(self._val_gt_manager._data[:, 0], self._val_sol_manager._data[:, 0]).mean().item()
        psnr2 = RangeInvariantPsnr(self._val_gt_manager._data[:, 1], self._val_sol_manager._data[:, 1]).mean().item()
        psnr = (psnr1 + psnr2) / 2
        self.log('val_psnr', psnr, on_epoch=True)

        self.label1_psnr.reset()
        self.label2_psnr.reset()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    mask = AutoRegRALadderVAE.get_mask(64, 'left', 'cpu')[0, 0].numpy()
    mask = np.repeat(mask, 64, axis=0) if mask.shape[0] == 1 else np.repeat(mask, 64, axis=1)
    plt.imshow(mask)
    plt.show()
    # from disentangle.configs.autoregressive_config import get_config
    # from disentangle.data_loader.patch_index_manager import GridAlignement, GridIndexManager
    # GridIndexManager((61, 2700, 2700, 2), 1, 64, GridAlignement.LeftTop, set_train_instance=True)
    # GridIndexManager((6, 2700, 2700, 2), 1, 64, GridAlignement.LeftTop, set_val_instance=True)

    # config = get_config()
    # config.model.skip_boundary_pixelcount = 16
    # data_mean = torch.Tensor([0]).reshape(1, 1, 1, 1)
    # data_std = torch.Tensor([1]).reshape(1, 1, 1, 1)
    # model = AutoRegRALadderVAE(data_mean, data_std, config)
    # inp = torch.rand((20, 1, config.data.image_size, config.data.image_size))
    # nbr = [torch.rand((20, 2, config.data.image_size, config.data.image_size))] * 4
    # out, td_data = model(inp, nbr)
    # batch = (torch.rand((16, 1, config.data.image_size, config.data.image_size)),
    #          torch.rand((16, 2, config.data.image_size, config.data.image_size)), torch.randint(0, 100, (16, )),
    #          torch.Tensor(np.array([config.data.image_size] * 16)).reshape(16, ).type(torch.int32))
    # model.training_step(batch, 0)
    # model.validation_step(batch, 0)
    # model.on_validation_epoch_end()
