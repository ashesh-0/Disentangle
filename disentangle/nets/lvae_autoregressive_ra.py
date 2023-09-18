import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from disentangle.analysis.lvae_utils import get_img_from_forward_output
from disentangle.core.data_split_type import DataSplitType
from disentangle.core.data_utils import crop_img_tensor
from disentangle.core.model_type import ModelType
from disentangle.core.psnr import RangeInvariantPsnr
from disentangle.data_loader.patch_index_manager import GridIndexManager
from disentangle.nets.lvae import LadderVAE
from disentangle.nets.lvae_layers import BottomUpLayer
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
        self._enable_rotation = config.model.get('rotation_with_neighbors', False)
        self._untrained_nbr_branch = config.model.get('untrained_nbr_branch', False)
        self._learnable_mask = config.model.get('nbr_learnable_mask', False)
        self._enable_seep_merge = config.model.get('nbr_enable_seep_merge', False)
        assert self._enable_seep_merge is False or self._learnable_mask is False
        self._nbr_disabled = config.model.get('nbr_disabled', False)
        super().__init__(data_mean, data_std, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=target_ch)
        self._neighboring_encoder = None

        latent_shapes = [self.img_shape[0] // (np.power(2, i + 1)) for i in range(self.n_layers)]
        self._avg_pool_layers = nn.ModuleList(
            [nn.AvgPool2d(kernel_size=latent_shapes[i]) for i in range(self.n_layers)])

        if self._learnable_mask:
            self._weight_masks = nn.ModuleList(
                [nn.Linear(1, latent_shapes[i], bias=False) for i in range(self.n_layers)])
            self._dummy_inp = torch.ones((1, 1, 1), requires_grad=False)
            self._dim_to_idx = {dim: idx for idx, dim in enumerate(latent_shapes)}

        self._nbr_share_weights = config.model.get('nbr_share_weights', False)

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
        stride = 1 if config.model.no_initial_downscaling else 2
        if self._nbr_share_weights:
            self._nbr_first_bottom_up = self.create_first_bottom_up(stride, color_ch=2)
            self._nbr_bottom_up_layers = self.create_bottom_up_layers(config.model.multiscale_lowres_separate_branch,
                                                                      enable_nbr_embedding=False)
            self._nbr_first_bottom_up_list = [self._nbr_first_bottom_up for _ in range(nbr_count)]
            self._nbr_bottom_up_layers_list = [self._nbr_bottom_up_layers for _ in range(nbr_count)]
        else:
            self._nbr_first_bottom_up_list = nn.ModuleList(
                [self.create_first_bottom_up(stride, color_ch=2) for _ in range(nbr_count)])
            self._nbr_bottom_up_layers_list = nn.ModuleList([
                self.create_bottom_up_layers(config.model.multiscale_lowres_separate_branch, enable_nbr_embedding=False)
                for _ in range(nbr_count)
            ])
        if self._nbr_disabled:
            print(f'[{self.__class__.__name__}]Rotation:{self._enable_rotation} NbrDisabled:{self._nbr_disabled}')

        else:
            print(
                f'[{self.__class__.__name__}]Rotation:{self._enable_rotation} NbrSharedWeights:{self._nbr_share_weights}\
               LearnableMask:{self._learnable_mask} UntrainedNbrBranch:{self._untrained_nbr_branch} Seep:{self._enable_seep_merge}'
            )

    def get_top_prior_param_shape(self, n_imgs=1):
        shape = super().get_top_prior_param_shape(n_imgs)
        if self._enable_seep_merge:
            # the way we are feeding the target (2 pixels on each side at first bottom-up layer nad 1 pixel on subsequent layers)
            return (*shape[:-2], shape[-2]+4, shape[-1]+4)
        else:
            return shape
            
    def create_top_down_layers(self):
        return super().create_top_down_layers(enable_nbr_embedding=True)
    
    def get_latent_spatialsize(self):
        """
        This assumes that a padding of 2 on each of the 4 sides is used for the first layer. 
        Every subsequent layer has a padding of 1 on each side.
        """
        latent_shapes = [(self.img_shape[0] // np.power(2, i + 1) + 2) for i in range(self.n_layers)]
        return latent_shapes

    def create_bottom_up_layers(self, lowres_separate_branch, enable_nbr_embedding=True):
        """
        For neighboring bottom up branch, we don't need nbr_embedding. We just need it for the main bottom up branch.
        """
        nbr_kwargs = {
            'nbr_enable_seep_merge': self._enable_seep_merge,
            'nbr_learnable_mask': self._learnable_mask,
            'nbr_merge_type': 'residual',
            'nbr_latent_spatialsize': None,
            'enable_nbr_embedding': enable_nbr_embedding,
        }
        enable_multiscale = self._multiscale_count is not None and self._multiscale_count > 1
        multiscale_lowres_size_factor = 1
        bottom_up_layers = []
        for i in range(self.n_layers):
            layer_enable_multiscale = enable_multiscale and self._multiscale_count > i + 1
            # if multiscale is enabled, this is the factor by which the lowres tensor will be larger than
            multiscale_lowres_size_factor *= (1 + int(layer_enable_multiscale))

            nbr_kwargs['nbr_latent_spatialsize'] = self.get_latent_spatialsize()[i]
            output_expected_shape = (self.img_shape[0] // 2**(i + 1),
                                     self.img_shape[1] // 2**(i + 1)) if self._multiscale_count > 1 else None

            bottom_up_layers.append(
                BottomUpLayer(n_res_blocks=self.encoder_blocks_per_layer,
                              n_filters=self.encoder_n_filters,
                              downsampling_steps=self.downsample[i],
                              nonlin=self.get_nonlin(),
                              batchnorm=self.bottomup_batchnorm,
                              dropout=self.encoder_dropout,
                              res_block_type=self.res_block_type,
                              res_block_kernel=self.encoder_res_block_kernel,
                              res_block_skip_padding=self.encoder_res_block_skip_padding,
                              lowres_separate_branch=lowres_separate_branch,
                              gated=self.gated,
                              enable_multiscale=False,
                              multiscale_retain_spatial_dims=self.multiscale_retain_spatial_dims,
                              multiscale_lowres_size_factor=multiscale_lowres_size_factor,
                              decoder_retain_spatial_dims=self.multiscale_decoder_retain_spatial_dims,
                              output_expected_shape=output_expected_shape,
                              **nbr_kwargs))
        return nn.ModuleList(bottom_up_layers)

    def _get_nbr_bu_values_one_side(self, nbr_predictions_one_side, nbr_first_bottom_up, nbr_bottom_up_layers):
        return self._bottomup_pass(nbr_predictions_one_side, nbr_first_bottom_up, None, nbr_bottom_up_layers)

    def _bottomup_pass(self, inp, first_bottom_up, lowres_first_bottom_ups, bottom_up_layers):
        if isinstance(inp, tuple):
            inp, nbr_embeddings = inp
        else:
            nbr_embeddings = [None] * self.n_layers

        if self._multiscale_count > 1:
            # Bottom-up initial layer. The first channel is the original input, what we want to reconstruct.
            # later channels are simply to yield more context.
            x = first_bottom_up(inp[:, :1])
        else:
            x = first_bottom_up(inp)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []
        for i in range(self.n_layers):
            lowres_x = None
            if self._multiscale_count > 1 and i + 1 < inp.shape[1]:
                lowres_x = lowres_first_bottom_ups[i](inp[:, i + 1:i + 2])

            x, bu_value = bottom_up_layers[i](x, lowres_x=lowres_x, nbr_embeddings=nbr_embeddings[i])
            bu_values.append(bu_value)

        return bu_values

    def _get_nbr_bu_values(self, nbr_predictions):
        nbr_bu_values_list = []
        for _ in range(len(self.bottom_up_layers)):
            nbr_bu_values_list.append([])

        # get some latent space encoding for the neighboring prediction.
        # top, bottom, left, right
        assert len(nbr_predictions) == 4
        nbr_bu_values_top = self._get_nbr_bu_values_one_side(nbr_predictions[0], self._nbr_first_bottom_up_list[0],
                                                             self._nbr_bottom_up_layers_list[0])

        nbr_bu_values_bottom = self._get_nbr_bu_values_one_side(nbr_predictions[1], self._nbr_first_bottom_up_list[1],
                                                                self._nbr_bottom_up_layers_list[1])

        nbr_bu_values_left = self._get_nbr_bu_values_one_side(nbr_predictions[2], self._nbr_first_bottom_up_list[2],
                                                              self._nbr_bottom_up_layers_list[2])

        nbr_bu_values_right = self._get_nbr_bu_values_one_side(nbr_predictions[3], self._nbr_first_bottom_up_list[3],
                                                               self._nbr_bottom_up_layers_list[3])

        nbr_bu_values_list = list(zip(nbr_bu_values_top, nbr_bu_values_bottom, nbr_bu_values_left, nbr_bu_values_right))
        return nbr_bu_values_list

    def forward(self, x, nbr_pred):
        img_size = x.size()[2:]

        nbr_bu_values = self._get_nbr_bu_values(nbr_pred)
        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x)
        bu_values = self.bottomup_pass((x_pad, nbr_bu_values))

        mode_layers = range(self.n_layers) if self.non_stochastic_version else None
        # Top-down inference/generation
        out, td_data = self.topdown_pass(bu_values, mode_layers=mode_layers)

        if out.shape[-1] > img_size[-1]:
            # Restore original image size
            out = crop_img_tensor(out, img_size)

        return out, td_data

    def get_output_from_batch(self, batch, sol_manager=None, enable_rotation=False):
        if sol_manager is None:
            sol_manager = self._val_sol_manager

        x, target, batch_locations, grid_sizes = batch

        self.set_params_to_same_device_as(target)
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)
        if self._nbr_disabled:
            nbr_preds = None
        else:
            nbr_preds = []

            nbr_preds.append(sol_manager.get_top(batch_locations, grid_sizes))
            nbr_preds.append(sol_manager.get_bottom(batch_locations, grid_sizes))
            nbr_preds.append(sol_manager.get_left(batch_locations, grid_sizes))
            nbr_preds.append(sol_manager.get_right(batch_locations, grid_sizes))
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

        # del nbrs
        return {
            'out': out,
            'input_normalized': x_normalized,
            'target_normalized': target_normalized,
            'td_data': td_data,
            'quadrant': quadrant if enable_rotation else None,
        }

    def _get_learned_mask(self, layer_idx):
        return self.bottom_up_layers[layer_idx].nbr_embedding_manager.get_learned_mask()

    def training_step(self, batch, batch_idx, enable_logging=True):
        # log
        if self._learnable_mask and not self._nbr_disabled:
            w = self._get_learned_mask(0).detach().cpu().numpy().squeeze()
            self.log('mask_w0', w[0], on_epoch=True)
            self.log(f'mask_w{len(w)-1}', w[-1], on_epoch=True)

        x, target, batch_locations, grid_sizes = batch
        batch_locations = batch_locations.cpu().numpy()
        batch = (x, target, batch_locations, grid_sizes)

        output_dict = self.get_output_from_batch(batch, self._train_sol_manager, enable_rotation=self._enable_rotation)
        imgs = get_img_from_forward_output(output_dict['out'], self, unnormalized=False, likelihood_obj=self.likelihood)

        if output_dict['quadrant'] is not None and output_dict['quadrant'] > 0:
            imgs = torch.rot90(imgs, k=-output_dict['quadrant'], dims=(2, 3))

        self._train_sol_manager.update(imgs.cpu().detach().numpy(), batch[2], batch[3])
        # in case of rotation, batch is invalid. since this is oly done in training,
        # None is being passed for batch in training and not in validation
        return self._training_step(None, batch_idx, output_dict, enable_logging=enable_logging)

    def validation_step(self, batch, batch_idx, return_output_dict=False):
        x, target, batch_locations, grid_sizes = batch
        batch_locations = batch_locations.cpu().numpy()
        batch = (x, target, batch_locations, grid_sizes)

        output_dict = self.get_output_from_batch(batch, self._val_sol_manager)
        imgs = get_img_from_forward_output(output_dict['out'], self, unnormalized=False, likelihood_obj=self.likelihood)
        if not self._nbr_disabled:
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

    from disentangle.configs.autoregressive_config import get_config
    from disentangle.data_loader.patch_index_manager import GridAlignement, GridIndexManager

    GridIndexManager((61, 2700, 2700, 2), 1, 64, GridAlignement.LeftTop, set_train_instance=True)
    GridIndexManager((20, 2700, 2700, 2), 1, 64, GridAlignement.LeftTop, set_val_instance=True)
    config = get_config()
    config.model.skip_boundary_pixelcount = 16
    data_mean = torch.Tensor([0]).reshape(1, 1, 1, 1)
    data_std = torch.Tensor([1]).reshape(1, 1, 1, 1)
    # with torch.no_grad():
    #     mask = AutoRegRALadderVAE(data_mean, data_std, config).get_mask(64, 'bottom', 'cpu')[0, 0].numpy()
    # mask = np.repeat(mask, 64, axis=0) if mask.shape[0] == 1 else np.repeat(mask, 64, axis=1)
    # plt.imshow(mask)
    # plt.show()

    model = AutoRegRALadderVAE(data_mean, data_std, config)
    inp = torch.rand((20, 1, config.data.image_size, config.data.image_size))
    nbr = [torch.rand((20, 2, config.data.image_size, config.data.image_size))] * 4
    out, td_data = model(inp, nbr)
    batch = (torch.rand(
        (16, 1, config.data.image_size,
         config.data.image_size)), torch.rand(
             (16, 2, config.data.image_size, config.data.image_size)), torch.randint(0, 10, (16, 3), dtype=torch.int32),
             torch.Tensor(np.array([config.data.image_size] * 16)).reshape(16, ).type(torch.int32))
    model.training_step(batch, 0)
    model.validation_step(batch, 0)
    model.on_validation_epoch_end()
