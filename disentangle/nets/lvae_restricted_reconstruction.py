from disentangle.core.loss_type import LossType
from disentangle.loss.restricted_reconstruction_loss import RestrictedReconstruction
from disentangle.nets.lvae import LadderVAE


class LadderVAERestrictedReconstruction(LadderVAE):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at, target_ch)
        self.automatic_optimization = False
        assert self.loss_type == LossType.ElboRestrictedReconstruction
        self.mixed_rec_w = config.loss.mixed_rec_weight
        self.grad_setter = RestrictedReconstruction(1, self.mixed_rec_w)

    def training_step(self, batch, batch_idx, enable_logging=True):
        if self.current_epoch == 0 and batch_idx == 0:
            self.log('val_psnr', 1.0, on_epoch=True)

        x, target = batch[:2]
        x_normalized = self.normalize_input(x)
        assert self.reconstruction_mode != True
        target_normalized = self.normalize_target(target)
        mask = ~((target == 0).reshape(len(target), -1).all(dim=1))
        out, td_data = self.forward(x_normalized)
        assert self.loss_type == LossType.ElboRestrictedReconstruction
        pred_x_normalized, _ = self.get_mixed_prediction(out, None, self.data_mean, self.data_std)
        optim = self.optimizers()
        optim.zero_grad()
        loss_dict = self.grad_setter.set_gradients(self.parameters(), x_normalized, target_normalized, out,
                                                   pred_x_normalized)
        optim.step()

        assert self.non_stochastic_version == True
        if enable_logging:
            self.log('training_loss', loss_dict['loss'], on_epoch=True)
            self.log('reconstruction_loss', loss_dict['split_loss'], on_epoch=True)
            self.log('input_reconstruction_loss', loss_dict['input_reconstruction_loss'], on_epoch=True)


if __name__ == '__main__':
    import numpy as np
    import torch

    from disentangle.configs.biosr_sparsely_supervised_config import get_config
    config = get_config()
    # config.loss.critic_loss_weight = 0.0
    data_mean = torch.Tensor([0]).reshape(1, 1, 1, 1)
    data_std = torch.Tensor([1]).reshape(1, 1, 1, 1)
    model = LadderVAERestrictedReconstruction({
        'input': data_mean,
        'target': data_mean.repeat(1, 2, 1, 1)
    }, {
        'input': data_std,
        'target': data_std.repeat(1, 2, 1, 1)
    }, config)
    model.configure_optimizers()
    mc = 1 if config.data.multiscale_lowres_count is None else config.data.multiscale_lowres_count
    inp = torch.rand((2, mc, config.data.image_size, config.data.image_size))
    out, td_data = model(inp)
    batch = (
        torch.rand((16, mc, config.data.image_size, config.data.image_size)),
        torch.rand((16, 2, config.data.image_size, config.data.image_size)),
    )
    batch[1][::2] = 0 * batch[1][::2]

    model.validation_step(batch, 0)
    model.training_step(batch, 0)

    ll = torch.ones((12, 2, 32, 32))
    ll_new = model._get_weighted_likelihood(ll)
    print(ll_new[:, 0].mean(), ll_new[:, 0].std())
    print(ll_new[:, 1].mean(), ll_new[:, 1].std())
    print('mar')
