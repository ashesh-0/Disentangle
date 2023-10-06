import os
from copy import deepcopy

import torch

import ml_collections
from disentangle.core.data_split_type import DataSplitType
from disentangle.core.psnr import RangeInvariantPsnr
from disentangle.data_loader.patch_index_manager import GridIndexManager
from disentangle.nets.lvae import LadderVAE
from disentangle.nets.lvae_autoregressive_ra import AutoRegRALadderVAE
from disentangle.nets.solutionRA_manager import SolutionRAManager


class TwinAutoRegRALadderVAE(AutoRegRALadderVAE):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        new_config = deepcopy(ml_collections.ConfigDict(config))
        with new_config.unlocked():
            new_config.data.color_ch = 2
            new_config.data.multiscale_lowres_count = 1 if config.data.multiscale_lowres_count != None else None
            # make it lean.
            new_config.model.z_dims = [16, 16, 16, 16]
            new_config.model.encoder.n_filters = 8
            new_config.model.decoder.n_filters = 8
        super().__init__(data_mean, data_std, new_config, use_uncond_mode_at, target_ch)
        self._vae0 = LadderVAE(data_mean, data_std, config, use_uncond_mode_at, target_ch)
        innerpad_amount = GridIndexManager(get_val_instance=True).get_innerpad_amount()
        self._val_sol0_manager = SolutionRAManager(DataSplitType.Val,
                                                   innerpad_amount,
                                                   config.data.image_size,
                                                   dump_img_dir=os.path.join(config.workdir, 'val_imgs0'),
                                                   enable_after_nepoch=self._enable_after_nepoch)

    # def get_modelspecific_loss(self, model, batch):
    #     output_dict = model.get_output_from_batch(batch)
    #     out = output_dict['out']
    #     target_normalized = output_dict['target_normalized']
    #     recons_loss_dict, imgs = model.get_reconstruction_loss(out, target_normalized, return_predicted_img=True)
    #     return output_dict, recons_loss_dict, imgs

    def forward(self, x, nbr_pred):
        out0, td_data0 = self._vae0(x)
        img0 = self._vae0.likelihood.get_mean_lv(out0)[0]
        img0 = img0.detach()
        out, td_data = super().forward(img0, nbr_pred)
        td_data['img0'] = img0
        td_data['td_data0'] = td_data0
        td_data['out0'] = out0
        return out, td_data

    def training_step(self, batch, batch_idx, enable_logging=True):
        output_dict = self.get_output_from_batch(batch,
                                                 self._train_sol_manager,
                                                 enable_rotation=self._enable_rotation,
                                                 enable_flips=self._enable_flips)

        out0 = output_dict['td_data']['out0']
        td_data0 = output_dict['td_data']['td_data0']
        target_normalized = output_dict['target_normalized']
        recons_loss_dict0, imgs0 = self.get_reconstruction_loss(out0, target_normalized, return_predicted_img=True)

        out = output_dict['out']
        recons_loss_dict, imgs = self.get_reconstruction_loss(out, target_normalized, return_predicted_img=True)

        assert self.skip_nboundary_pixels_from_loss is None

        recons_loss = recons_loss_dict['loss'] + recons_loss_dict0['loss']

        if self.non_stochastic_version:
            kl_loss0 = torch.Tensor([0.0]).cuda()
            kl_loss1 = torch.Tensor([0.0]).cuda()
            kl_loss = kl_loss0 + kl_loss1
            net_loss = recons_loss
        else:
            kl_loss0 = self.get_kl_divergence_loss(output_dict['td_data'])
            kl_loss1 = self.get_kl_divergence_loss(td_data0)
            kl_loss = kl_loss0 + kl_loss1
            net_loss = recons_loss + self.get_kl_weight() * kl_loss

        if enable_logging:
            self.log('reconstruction_loss', recons_loss, on_epoch=True)
            self.log('reconstruction_loss0', recons_loss_dict0['loss'], on_epoch=True)
            self.log('reconstruction_loss1', recons_loss_dict['loss'], on_epoch=True)

            self.log('kl_loss', kl_loss, on_epoch=True)
            self.log('kl_loss0', kl_loss0, on_epoch=True)
            self.log('kl_loss1', kl_loss1, on_epoch=True)

            self.log('training_loss', net_loss, on_epoch=True)
            self.log('lr', self.lr, on_epoch=True)

        output = {
            'loss': net_loss,
            'reconstruction_loss': recons_loss.detach(),
            'kl_loss': kl_loss.detach(),
        }
        # TODO: check if making this imgs will be helpful
        self._train_sol_manager.update(imgs0.cpu().detach().numpy(), batch[2], batch[3])
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output

    def validation_step(self, batch, batch_idx):
        # dict_data = self.get_output_from_sequential_model(batch, return_new_batch=True)
        # output_dict0, recons_loss_dict0, imgs0 = dict_data['img0']
        # output_dict, recons_loss_dict, imgs = dict_data['img1']
        output_dict = self.get_output_from_batch(batch, self._val_sol_manager)

        out0 = output_dict['td_data']['out0']
        td_data0 = output_dict['td_data']['td_data0']
        target_normalized = output_dict['target_normalized']
        recons_loss_dict0, imgs0 = self.get_reconstruction_loss(out0, target_normalized, return_predicted_img=True)

        out = output_dict['out']
        recons_loss_dict, imgs = self.get_reconstruction_loss(out, target_normalized, return_predicted_img=True)

        self._val_sol_manager.update(imgs.cpu().detach().numpy(), batch[2], batch[3])
        self._val_sol0_manager.update(imgs0.cpu().detach().numpy(), batch[2], batch[3])

        self._val_gt_manager.update(output_dict['target_normalized'].cpu().detach().numpy(), batch[2], batch[3])
        # TODO: log the validation psnr 1 and 2.
        target_normalized = output_dict['target_normalized']
        self._vae0.label1_psnr.update(imgs0[:, 0], target_normalized[:, 0])
        self._vae0.label2_psnr.update(imgs0[:, 1], target_normalized[:, 1])

        self.label1_psnr.update(imgs[:, 0], target_normalized[:, 0])
        self.label2_psnr.update(imgs[:, 1], target_normalized[:, 1])

    def on_validation_epoch_end(self):
        # PSNR
        psnr1 = RangeInvariantPsnr(self._val_gt_manager._data[:, 0], self._val_sol_manager._data[:, 0]).mean().item()
        psnr2 = RangeInvariantPsnr(self._val_gt_manager._data[:, 1], self._val_sol_manager._data[:, 1]).mean().item()
        psnr = (psnr1 + psnr2) / 2
        self.log('val_psnr', psnr, on_epoch=True)

        psnr01 = RangeInvariantPsnr(self._val_gt_manager._data[:, 0], self._val_sol0_manager._data[:, 0]).mean().item()
        psnr02 = RangeInvariantPsnr(self._val_gt_manager._data[:, 1], self._val_sol0_manager._data[:, 1]).mean().item()
        psnr0 = (psnr01 + psnr02) / 2
        self.log('val0_psnr', psnr0, on_epoch=True)

        self.label1_psnr.reset()
        self.label2_psnr.reset()
        self._vae0.label1_psnr.reset()
        self._vae0.label2_psnr.reset()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    mask = TwinAutoRegRALadderVAE.get_mask(64, 'left', 'cpu')[0, 0].numpy()
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
    model = TwinAutoRegRALadderVAE(data_mean, data_std, config)
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
