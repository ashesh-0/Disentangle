import torch

from disentangle.nets.lvae import LadderVAE


class LadderVAEDenoiser(LadderVAE):

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target = batch[:2]
        new_target = torch.tile(x, (1, 2, 1, 1))
        assert new_target.shape == target.shape
        batch = (x, new_target) + batch[2:]
        return super().training_step(batch, batch_idx, enable_logging)

    def validation_step(self, batch, batch_idx):
        x, target = batch[:2]
        new_target = torch.tile(x, (1, 2, 1, 1))
        assert new_target.shape == target.shape
        batch = (x, new_target) + batch[2:]
        return super().validation_step(batch, batch_idx)
