import numbers
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from disentangle.core.loss_type import LossType
from disentangle.nets.learned_param_utils import get_params_of_type
from disentangle.nets.lvae import LadderVAE


class LadderVAEInterleavedOptimization(LadderVAE):
    """
    Few layers are optimized for reconstrution, few layers are optimized for splitting
    """

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=target_ch)
        self.automatic_optimization = False

    def configure_optimizers(self):
        reconstr_params = get_params_of_type(self, pytorch_type=nn.BatchNorm2d)
        reconstr_params += get_params_of_type(self, just_bias=True, except_pytorch_type=nn.BatchNorm2d)

        split_params = []
        for _, param in self.named_parameters():
            split_params.append(param)

        optimizer_split = optim.Adamax(split_params, lr=self.lr, weight_decay=0)
        optimizer_reconst = optim.Adamax(reconstr_params, lr=self.lr, weight_decay=0)

        scheduler0 = self.get_scheduler(optimizer_split)
        scheduler1 = self.get_scheduler(optimizer_reconst)

        return [optimizer_split, optimizer_reconst], [{
            'scheduler': scheduler,
            'monitor': self.lr_scheduler_monitor,
        } for scheduler in [scheduler0, scheduler1]]

    def get_scheduler(self, optimizer):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    self.lr_scheduler_mode,
                                                    patience=self.lr_scheduler_patience,
                                                    factor=0.5,
                                                    min_lr=1e-12,
                                                    verbose=True)

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target = batch[:2]
        x_normalized = self.normalize_input(x)
        optimizer_split, optimizer_reconst = self.optimizers()

        assert self.reconstruction_mode is False
        assert self.skip_nboundary_pixels_from_loss is None

        target_normalized = self.normalize_target(target)
        out_recons, _ = self.forward(x_normalized)
        recons_loss_dict = self.get_reconstruction_loss(out_recons,
                                                        target_normalized,
                                                        x_normalized,
                                                        None,
                                                        return_predicted_img=False)

        recons_loss = recons_loss_dict['mixed_loss']
        assert self.loss_type == LossType.ElboMixedReconstruction
        if enable_logging:
            self.log('mixed_reconstruction_loss', recons_loss_dict['mixed_loss'], on_epoch=True)

        net_loss = recons_loss
        optimizer_reconst.zero_grad()
        self.manual_backward(net_loss)
        optimizer_reconst.step()

        splitting_mask = ~((target == 0).reshape(len(target), -1).all(dim=1))
        x_normalized_split = x_normalized[splitting_mask]
        out_split, td_data_split = self.forward(x_normalized_split)
        target_normalized_split = target_normalized[splitting_mask]
        split_loss_dict, _ = self.get_reconstruction_loss(out_split,
                                                          target_normalized_split,
                                                          x_normalized_split,
                                                          None,
                                                          return_predicted_img=True)

        split_loss = split_loss_dict['loss']
        if self.non_stochastic_version:
            split_kl_loss = torch.Tensor([0.0]).cuda()
            net_loss = split_loss
        else:
            split_kl_loss = self.get_kl_divergence_loss(td_data_split)
            net_loss = split_loss + self.get_kl_weight() * split_kl_loss

        optimizer_split.zero_grad()
        self.manual_backward(net_loss)
        optimizer_split.step()

        kl_loss = split_kl_loss
        # print(f'rec:{split_loss_dict["loss"]:.3f} mix: {split_loss_dict.get("mixed_loss",0):.3f} KL: {kl_loss:.3f}')
        if enable_logging:
            self.log('reconstruction_loss', split_loss_dict['loss'], on_epoch=True)
            self.log('kl_loss', kl_loss, on_epoch=True)
            # self.log('training_loss', net_loss, on_epoch=True)
            # self.log('lr', self.lr, on_epoch=True)
            # self.log('grad_norm_bottom_up', self.grad_norm_bottom_up, on_epoch=True)
            # self.log('grad_norm_top_down', self.grad_norm_top_down, on_epoch=True)


if __name__ == '__main__':
    import numpy as np
    import torch

    from disentangle.configs.twotiff_config import get_config

    config = get_config()
    data_mean = torch.Tensor([0]).reshape(1, 1, 1, 1)
    data_std = torch.Tensor([1]).reshape(1, 1, 1, 1)
    model = LadderVAEInterleavedOptimization({
        'input': data_mean,
        'target': data_mean
    }, {
        'input': data_std,
        'target': data_std
    }, config)
    names = []
    for name, _ in model.named_parameters():
        names.append(name)
    model.configure_optimizers()
    print(set({name.split('.')[0] for name in names}))

    mc = 1 if config.data.multiscale_lowres_count is None else config.data.multiscale_lowres_count + 1
    inp = torch.rand((2, mc, config.data.image_size, config.data.image_size))
    out, td_data = model(inp)

    inp = torch.rand((16, mc, config.data.image_size, config.data.image_size))
    tar = torch.rand((16, 2, config.data.image_size, config.data.image_size))
    tar[:8] = 0
    batch = (inp, tar)

    model.training_step(batch, 0)
    model.validation_step(batch, 0)

    ll = torch.ones((12, 2, 32, 32))
    ll_new = model._get_weighted_likelihood(ll)
    print(ll_new[:, 0].mean(), ll_new[:, 0].std())
    print(ll_new[:, 1].mean(), ll_new[:, 1].std())
    print('mar')
