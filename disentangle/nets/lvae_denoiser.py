import torch

from disentangle.nets.lvae import LadderVAE


class LadderVAEDenoiser(LadderVAE):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[]):
        # since input is the target, we don't need to normalize it at all.
        super().__init__(0, 1, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=1)

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


if __name__ == '__main__':
    import numpy as np
    import torch

    from disentangle.configs.microscopy_multi_channel_lvae_config import get_config

    config = get_config()
    data_mean = torch.Tensor([0]).reshape(1, 1, 1, 1)
    data_std = torch.Tensor([1]).reshape(1, 1, 1, 1)
    model = LadderVAEDenoiser(data_mean, data_std, config)
    mc = 1 if config.data.multiscale_lowres_count is None else config.data.multiscale_lowres_count + 1
    inp = torch.rand((2, mc, config.data.image_size, config.data.image_size))
    out, td_data = model(inp)
    print(out.shape)
    batch = (
        torch.rand((16, mc, config.data.image_size, config.data.image_size)),
        torch.rand((16, 2, config.data.image_size, config.data.image_size)),
    )
    model.training_step(batch, 0)
    model.validation_step(batch, 0)

    ll = torch.ones((12, 2, 32, 32))
    ll_new = model._get_weighted_likelihood(ll)
    print(ll_new[:, 0].mean(), ll_new[:, 0].std())
    print(ll_new[:, 1].mean(), ll_new[:, 1].std())
    print('mar')
