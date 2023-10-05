import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT

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

    def hflip(self):
        """
        Flip about vertical axis
        """
        left = self.right
        right = self.left
        self.left = left
        self.right = right
        self.top = torch.flip(self.top, dims=(3, ))
        self.bottom = torch.flip(self.bottom, dims=(3, ))
        self.left = torch.flip(self.left, dims=(3, ))
        self.right = torch.flip(self.right, dims=(3, ))

    def vflip(self):
        """
        Flip about horizontal axis
        """
        top = self.bottom
        bottom = self.top
        self.top = top
        self.bottom = bottom
        self.top = torch.flip(self.top, dims=(2, ))
        self.bottom = torch.flip(self.bottom, dims=(2, ))
        self.left = torch.flip(self.left, dims=(2, ))
        self.right = torch.flip(self.right, dims=(2, ))

    def flip(self, hflip, vflip):
        """
        hflip: bool, whether to flip about vertical axis
        vflip: bool, whether to flip about horizontal axis
        """
        assert isinstance(hflip, bool) and isinstance(vflip, bool), "hflip and vflip must be boolean"
        if hflip == False and vflip == False:
            return
        if hflip == True:
            self.hflip()
        if vflip == True:
            self.vflip()

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
        self._enable_flips = config.model.get('flips_with_neighbors', False)

        self._untrained_nbr_branch = config.model.get('untrained_nbr_branch', False)
        self._skip_nbr_in_bottomk_levels = config.model.get('skip_nbr_in_bottomk_levels', -1)
        self._avg_pool_layers = nn.ModuleList(
            [nn.AvgPool2d(kernel_size=self.img_shape[0] // (np.power(2, i + 1))) for i in range(self.n_layers)])

        self._nbr_share_weights = config.model.get('nbr_share_weights', False)
        self._nbr_disabled = config.model.get('nbr_disabled', False)
        self._enable_after_nepoch = config.model.get('enable_after_nepoch', -1)

        # when creating the frame prediction, we want to skip boundary.
        innerpad_amount = GridIndexManager(get_val_instance=True).get_innerpad_amount()
        self._train_sol_manager = SolutionRAManager(DataSplitType.Train,
                                                    innerpad_amount,
                                                    config.data.image_size,
                                                    dump_img_dir=os.path.join(config.workdir, 'train_imgs'),
                                                    dropout=config.model.get('nbr_dropout', 0.0),
                                                    enable_after_nepoch=self._enable_after_nepoch)
        self._val_sol_manager = SolutionRAManager(DataSplitType.Val,
                                                  innerpad_amount,
                                                  config.data.image_size,
                                                  dump_img_dir=os.path.join(config.workdir, 'val_imgs'),
                                                  enable_after_nepoch=self._enable_after_nepoch)
        # save the groundtruth
        self._val_gt_manager = SolutionRAManager(DataSplitType.Val,
                                                 innerpad_amount,
                                                 config.data.image_size,
                                                 dump_img_dir=os.path.join(config.workdir, 'val_groundtruth'))

        # self._train_gt_manager = SolutionRAManager(DataSplitType.Train,
        #                                            innerpad_amount,
        #                                            config.data.image_size,
        #                                            dump_img_dir=os.path.join(config.workdir, 'train_groundtruth'))

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

        # We don't need the merge layers for the bottomk levels where we skip the nbrs.
        for i in range(0, self._skip_nbr_in_bottomk_levels + 1):
            self._merge_layers[i] = nn.Identity()

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
        print(f'[{self.__class__.__name__}]Rotation:{self._enable_rotation} \
                Flips:{self._enable_flips} NbrSharedWeights:{self._nbr_share_weights} \
                SkipNbrBottomkLevels:{self._skip_nbr_in_bottomk_levels}')

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
        bu_values = self.bottomup_pass(x_pad)

        if self._nbr_disabled is False:
            nbr_bu_values_list = []
            for _ in range(len(self.bottom_up_layers)):
                nbr_bu_values_list.append([])

            # get some latent space encoding for the neighboring prediction.
            # top, bottom, left, right
            assert len(nbr_pred) == 4

            nbr_bu_values_top = self._bottomup_pass(nbr_pred[0],
                                                    self._nbr_first_bottom_up_list[0],
                                                    None,
                                                    self._nbr_bottom_up_layers_list[0],
                                                    disable_multiscale=True)
            shapes = [x.shape[-2:] for x in nbr_bu_values_top]
            assert all([x[0] == x[1] for x in shapes])

            nbr_bu_values_top = [x[:, :, -1:] for x in nbr_bu_values_top]
            # import pdb;pdb.set_trace()
            nbr_bu_values_top = [
                x * self.get_mask(shapes[i][0], 'top', x.device) for i, x in enumerate(nbr_bu_values_top)
            ]
            # nbr_bu_values_top = [x.repeat(1, 1, x.shape[3], 1) for x in nbr_bu_values_top]

            nbr_bu_values_bottom = self._bottomup_pass(nbr_pred[1],
                                                       self._nbr_first_bottom_up_list[1],
                                                       None,
                                                       self._nbr_bottom_up_layers_list[1],
                                                       disable_multiscale=True)

            nbr_bu_values_bottom = [x[:, :, :1] for x in nbr_bu_values_bottom]
            nbr_bu_values_bottom = [
                x * self.get_mask(shapes[i][0], 'bottom', x.device) for i, x in enumerate(nbr_bu_values_bottom)
            ]

            nbr_bu_values_left = self._bottomup_pass(nbr_pred[2],
                                                     self._nbr_first_bottom_up_list[2],
                                                     None,
                                                     self._nbr_bottom_up_layers_list[2],
                                                     disable_multiscale=True)

            nbr_bu_values_left = [x[..., -1:] for x in nbr_bu_values_left]
            nbr_bu_values_left = [
                x * self.get_mask(shapes[i][0], 'left', x.device) for i, x in enumerate(nbr_bu_values_left)
            ]

            nbr_bu_values_right = self._bottomup_pass(nbr_pred[3],
                                                      self._nbr_first_bottom_up_list[3],
                                                      None,
                                                      self._nbr_bottom_up_layers_list[3],
                                                      disable_multiscale=True)

            nbr_bu_values_right = [x[..., :1] for x in nbr_bu_values_right]
            nbr_bu_values_right = [
                x * self.get_mask(shapes[i][0], 'right', x.device) for i, x in enumerate(nbr_bu_values_right)
            ]

            nbr_bu_values_list = list(
                zip(nbr_bu_values_top, nbr_bu_values_bottom, nbr_bu_values_left, nbr_bu_values_right))

            multiscale_enabled = self._multiscale_count is not None and self._multiscale_count > 1
            if multiscale_enabled:
                upsampled_nbr_values = []
                # Upsample nbr_values to match the size of the bottom-up values
                for hierarchy_idx in range(len(nbr_bu_values_list)):
                    upsampled_one_hier = []
                    for nbr_value_h in nbr_bu_values_list[hierarchy_idx]:
                        pad = (bu_values[hierarchy_idx].shape[-1] - nbr_value_h.shape[-1]) // 2
                        upsampled_one_hier.append(F.pad(nbr_value_h, (pad, pad, pad, pad)))
                    upsampled_nbr_values.append(upsampled_one_hier)
                nbr_bu_values_list = upsampled_nbr_values

            merged_bu_values = []

            for idx in range(len(bu_values)):
                if idx > self._skip_nbr_in_bottomk_levels:
                    merged_bu_values.append(bu_values[idx] +
                                            self._merge_layers[idx](bu_values[idx], *nbr_bu_values_list[idx]))
                else:
                    merged_bu_values.append(bu_values[idx])
        else:
            merged_bu_values = bu_values

        mode_layers = range(self.n_layers) if self.non_stochastic_version else None
        # Top-down inference/generation
        out, td_data = self.topdown_pass(merged_bu_values, mode_layers=mode_layers)

        if out.shape[-1] > img_size[-1]:
            # Restore original image size
            out = crop_img_tensor(out, img_size)

        return out, td_data

    def get_output_from_batch(self, batch, sol_manager=None, enable_rotation=False, enable_flips=False, skip_nbr=False):
        if sol_manager is None:
            sol_manager = self._val_sol_manager

        x, target, indices, grid_sizes = batch
        self.set_params_to_same_device_as(target)
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)
        nbr_preds = sol_manager.get_nbrs(indices, grid_sizes, cur_epoch=self.current_epoch, skipdata=skip_nbr)
        if self._enable_after_nepoch > 0 and self.current_epoch < self._enable_after_nepoch:
            assert (nbr_preds[0] == 0).all()
            assert (nbr_preds[1] == 0).all()
            assert (nbr_preds[2] == 0).all()
            assert (nbr_preds[3] == 0).all()

        nbr_preds = [torch.Tensor(nbr_y).to(x.device) for nbr_y in nbr_preds]
        nbrs = Neighbors(*nbr_preds)
        nbr_preds = nbrs.get()
        # print('\n', 'LgStats', is_train, len(nbr_preds), f'{nbr_preds[0].shape}, {nbr_preds[0].max().item():.2f}, {nbr_preds[0].min().item():.2f}, {nbr_preds[0].mean().item():.2f}')

        if enable_rotation == True:
            quadrant = np.random.randint(0, 4)
            if quadrant > 0:
                x_normalized = torch.rot90(x_normalized, k=quadrant, dims=(2, 3))
                target_normalized = torch.rot90(target_normalized, k=quadrant, dims=(2, 3))
                nbrs.rotate90anticlock(quadrant)
                nbr_preds = nbrs.get()

        if enable_flips:
            hflip = bool(np.random.randint(0, 2))
            vflip = bool(np.random.randint(0, 2))
            if hflip:
                x_normalized = torch.flip(x_normalized, dims=(3, ))
                target_normalized = torch.flip(target_normalized, dims=(3, ))

            if vflip:
                x_normalized = torch.flip(x_normalized, dims=(2, ))
                target_normalized = torch.flip(target_normalized, dims=(2, ))

            nbrs.flip(hflip, vflip)
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
            'nbr_preds': nbr_preds,
            'hflip': hflip if enable_flips else None,
            'vflip': vflip if enable_flips else None,
        }

    def merge_loss_dicts(self, output_dicts):
        assert len(output_dicts) == 2
        output = {
            'loss': (output_dicts[0]['loss'] + output_dicts[1]['loss']) / 2,
            'reconstruction_loss':
            (output_dicts[0]['reconstruction_loss'] + output_dicts[1]['reconstruction_loss']) / 2,
            'kl_loss': (output_dicts[0]['kl_loss'] + output_dicts[1]['kl_loss']) / 2,
        }
        return output

    def training_step(self, batch, batch_idx, enable_logging=True):
        # the order of skip_nbr matters.
        output_dict = self.get_output_from_batch(batch,
                                                 self._train_sol_manager,
                                                 enable_rotation=self._enable_rotation,
                                                 enable_flips=self._enable_flips)
        imgs = get_img_from_forward_output(output_dict['out'], self, unnormalized=False, likelihood_obj=self.likelihood)

        if output_dict['quadrant'] is not None and output_dict['quadrant'] > 0:
            imgs = torch.rot90(imgs, k=-output_dict['quadrant'], dims=(2, 3))

        if output_dict['hflip'] is not None and output_dict['hflip']:
            imgs = torch.flip(imgs, dims=(3, ))
        if output_dict['vflip'] is not None and output_dict['vflip']:
            imgs = torch.flip(imgs, dims=(2, ))

        # if self.current_epoch % 10 == 0 and batch_idx < 2:
        #     nbrs = torch.cat([output_dict['target_normalized'], imgs] + output_dict['nbr_preds'], dim=1)
        #     nbrs = nbrs.detach().cpu().numpy()
        #     np.save(f'./nbrs_train_{self.current_epoch}_{batch_idx}.npy', nbrs)
        #     np.save(f'./nbrs_trainindices_{self.current_epoch}_{batch_idx}.npy', batch[2].detach().cpu().numpy())

        self._train_sol_manager.update(imgs.cpu().detach().numpy(), batch[2], batch[3])
        # in case of rotation, batch is invalid. since this is oly done in training,
        # None is being passed for batch in training and not in validation
        return self._training_step(None, batch_idx, output_dict, enable_logging=enable_logging)

    def test_step(self, batch, batch_idx, return_output_dict=False):
        self.validation_step(batch, batch_idx, return_output_dict=return_output_dict)

    def validation_step(self, batch, batch_idx, return_output_dict=False):
        output_dict = self.get_output_from_batch(batch, self._val_sol_manager)
        imgs = get_img_from_forward_output(output_dict['out'], self, unnormalized=False, likelihood_obj=self.likelihood)
        self._val_sol_manager.update(imgs.cpu().detach().numpy(), batch[2], batch[3])
        self._val_gt_manager.update(output_dict['target_normalized'].cpu().detach().numpy(), batch[2], batch[3])

        # if self.current_epoch % 10 == 0 and batch_idx < 2:
        #     nbrs = torch.cat([output_dict['target_normalized'], imgs] + output_dict['nbr_preds'], dim=1)
        #     nbrs = nbrs.detach().cpu().numpy()
        #     np.save(f'./nbrs_val_{self.current_epoch}_{batch_idx}.npy', nbrs)
        #     np.save(f'./nbrs_valindices_{self.current_epoch}_{batch_idx}.npy', batch[2].detach().cpu().numpy())

        val_out = self._validation_step(batch, batch_idx, output_dict)
        assert val_out is None

        if return_output_dict:
            return val_out, output_dict

        return val_out

    def on_validation_epoch_end(self):
        # if self.current_epoch % 10 == 0:
        #     np.save(f'val_{self.current_epoch}.npy', self._val_sol_manager._data)
        #     np.save(f'train_{self.current_epoch}.npy', self._train_sol_manager._data)
        #     np.save(f'train_gt_{self.current_epoch}.npy', self._train_gt_manager._data)
        #     np.save(f'val_gt_{self.current_epoch}.npy', self._val_gt_manager._data)

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
    # plt.imshow(mask)
    # plt.show()
    from disentangle.configs.autoregressive_config import get_config
    from disentangle.data_loader.patch_index_manager import GridAlignement, GridIndexManager
    GridIndexManager((61, 2700, 2700, 2), 1, 64, GridAlignement.LeftTop, set_train_instance=True)
    GridIndexManager((6, 2700, 2700, 2), 1, 64, GridAlignement.LeftTop, set_val_instance=True)
    config = get_config()
    config.model.skip_boundary_pixelcount = 16
    data_mean = torch.Tensor([0, 0]).reshape(1, 2, 1, 1)
    data_std = torch.Tensor([1, 1]).reshape(1, 2, 1, 1)
    model = AutoRegRALadderVAE(data_mean, data_std, config)
    mc = 1 if config.data.multiscale_lowres_count is None else config.data.multiscale_lowres_count + 1
    inp = torch.rand((20, mc, config.data.image_size, config.data.image_size))
    nbr = [torch.rand((20, 2, config.data.image_size, config.data.image_size))] * 4
    out, td_data = model(inp, nbr)
    batch = (torch.rand((16, mc, config.data.image_size, config.data.image_size)),
             torch.rand((16, 2, config.data.image_size, config.data.image_size)), torch.randint(0, 100, (16, )),
             torch.Tensor(np.array([config.data.image_size] * 16)).reshape(16, ).type(torch.int32))
    model.training_step(batch, 0)
    model.validation_step(batch, 0)
    # model.on_validation_epoch_end()
