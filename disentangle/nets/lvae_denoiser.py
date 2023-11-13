import torch

from disentangle.nets.lvae import LadderVAE


class LadderVAEDenoiser(LadderVAE):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        # since input is the target, we don't need to normalize it at all.
        super().__init__(0, 1, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=2)

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target = batch[:2]
        new_target = torch.tile(x[:, :1], (1, 2, 1, 1))
        assert new_target.shape == target.shape
        batch = (x, new_target, *batch[2:])
        return super().training_step(batch, batch_idx, enable_logging)

    def validation_step(self, batch, batch_idx):
        x, target = batch[:2]
        new_target = torch.tile(x[:, :1], (1, 2, 1, 1))
        assert new_target.shape == target.shape
        batch = (x, new_target, *batch[2:])
        return super().validation_step(batch, batch_idx)
