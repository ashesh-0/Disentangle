import torch

from disentangle.nets.lvae import LadderVAE


class LadderVAEDenoiser(LadderVAE):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[]):
        # since input is the target, we don't need to normalize it at all.
        super().__init__(data_mean, data_std, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=1)
        self._denoise_channel = config.model.denoise_channel
        assert self._denoise_channel in ['input', 'Ch1', 'Ch2']
        if self._denoise_channel == 'input':
            msg = 'For target, we expect it to be unnormalized. For such reasons, we expect same normalization for input and target.'
            assert len(self.data_mean['target'].squeeze()[:1]) == 2, msg
            assert self.data_mean['input'].squeeze() == self.data_mean['target'].squeeze()[:1], msg
            assert len(self.data_std['target'].squeeze()[:1]) == 2, msg
            assert self.data_std['input'].squeeze() == self.data_std['target'].squeeze()[:1], msg
        elif self._denoise_channel == 'Ch1':
            self.data_mean['target'] = self.data_mean['target'][:, :1]
            self.data_std['target'] = self.data_std['target'][:, :1]
        elif self._denoise_channel == 'Ch2':
            self.data_mean['target'] = self.data_mean['target'][:, 1:]
            self.data_std['target'] = self.data_std['target'][:, 1:]

    def get_new_input_target(self, batch):
        x, target = batch[:2]
        if self._denoise_channel == 'input':
            new_target = torch.tile(x[:, :1], (1, 2, 1, 1))
            # Input is normalized, but target is not. So we need to un-normalize it.
            new_target = new_target * self.data_std['input'] + self.data_mean['input']
        elif self._denoise_channel == 'Ch1':
            new_target = target[:, :1]
            # Input is normalized, but target is not. So we need to normalize it.
            x = self.normalize_target(new_target)

        elif self._denoise_channel == 'Ch2':
            new_target = target[:, 1:]
            # Input is normalized, but target is not. So we need to normalize it.
            x = self.normalize_target(new_target)
        return x, new_target

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, new_target = self.get_new_input_target(batch)
        batch = (x, new_target, *batch[2:])
        return super().training_step(batch, batch_idx, enable_logging)

    def validation_step(self, batch, batch_idx):
        self.set_params_to_same_device_as(batch[0])
        x, new_target = self.get_new_input_target(batch)
        batch = (x, new_target, *batch[2:])
        return super().validation_step(batch, batch_idx)


if __name__ == '__main__':
    import numpy as np
    import torch

    from disentangle.configs.denoiser_config import get_config

    config = get_config()
    data_mean = {'input': np.array([0]).reshape(1, 1, 1, 1), 'target': np.array([0, 0]).reshape(1, 2, 1, 1)}
    data_std = {'input': np.array([1]).reshape(1, 1, 1, 1), 'target': np.array([1, 1]).reshape(1, 2, 1, 1)}

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
