from disentangle.nets.lvae import LadderVAE
import torch.nn as nn
import torch

class LadderVAEMultiTarget(LadderVAE):
    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super(LadderVAEMultiTarget, self).__init__(data_mean,data_std,config,use_uncond_mode_at=use_uncond_mode_at,
                                                   target_ch=target_ch)
        self._lowres_supervision = True
        self.final_top_down_lowres = None

        self.final_top_down_lowres = nn.ModuleList()
        for _ in range(self._multiscale_count):
            self.final_top_down_lowres.append(self.create_final_topdown_layer(False))
        self._lowres_likelihoods = None
        self._lowres_likelihoods = nn.ModuleList()
        for _ in range(self._multiscale_count):
            self._lowres_likelihoods.append(self.create_likelihood())

    def validation_step(self, batch, batch_idx):
        x, target = batch
        return super().validation_step((x,target[:,0]), batch_idx)

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target = batch
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)

        out, td_data = self.forward(x_normalized)
        recons_loss = 0
        N = target_normalized.shape[1]
        for ith_res in range(N):
            if ith_res==0:
                recons_loss_dict = self.get_reconstruction_loss(out, target_normalized[:,0])
            else:
                recons_loss_dict = self.get_reconstruction_loss(td_data['out_lowres'][ith_res],
                                                                target_normalized[:,ith_res])
            recons_loss += recons_loss_dict['loss']/N



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