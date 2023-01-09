from disentangle.nets.lvae import LadderVAE
import torch
from disentangle.core.loss_type import LossType


class LadderVAEWithMixedRecons(LadderVAE):
    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=target_ch)

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target, mixed_recons_flag = batch[:2]
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)

        out, td_data = self.forward(x_normalized)
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict, imgs = self.get_reconstruction_loss(out[~mixed_recons_flag],
                                                              target_normalized[~mixed_recons_flag],
                                                              return_predicted_img=True)

        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        recons_loss = recons_loss_dict['loss']

        recons_loss_dict2, _ = self.get_reconstruction_loss(out[mixed_recons_flag],
                                                            target_normalized[mixed_recons_flag],
                                                            return_predicted_img=True)

        assert self.loss_type == LossType.ElboMixedReconstruction
        recons_loss += self.mixed_rec_w * recons_loss_dict2['mixed_loss']
        if enable_logging:
            self.log('mixed_reconstruction_loss', recons_loss_dict2['mixed_loss'], on_epoch=True)

        kl_loss = self.get_kl_divergence_loss(td_data)
        net_loss = recons_loss + self.get_kl_weight() * kl_loss

        if enable_logging:
            for i, x in enumerate(td_data['debug_qvar_max']):
                self.log(f'qvar_max:{i}', x.item(), on_epoch=True)

            self.log('reconstruction_loss', recons_loss_dict['loss'], on_epoch=True)
            self.log('kl_loss', kl_loss, on_epoch=True)
            self.log('training_loss', net_loss, on_epoch=True)
            self.log('lr', self.lr, on_epoch=True)
            self.log('grad_norm_bottom_up', self.grad_norm_bottom_up, on_epoch=True)
            self.log('grad_norm_top_down', self.grad_norm_top_down, on_epoch=True)

        output = {
            'loss': net_loss,
            'reconstruction_loss': recons_loss.detach(),
            'kl_loss': kl_loss.detach(),
        }

        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output

    def validation_step(self, batch, batch_idx):
        batch = batch[:2]
        return super().validation_step(batch, batch_idx)