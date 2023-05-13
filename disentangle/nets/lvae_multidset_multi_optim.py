import torch.optim as optim

from disentangle.core.loss_type import LossType
from disentangle.nets.lvae_multidset_multi_input_branches import LadderVaeMultiDatasetMultiBranch


class LadderVaeMultiDatasetMultiOptim(LadderVaeMultiDatasetMultiBranch):
    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at, target_ch)
        self.automatic_optimization=False

    def get_encoder_params(self):
        encoder_params = list(self._first_bottom_up_subdset1.parameters()) + list(self.bottom_up_layers.parameters())
        if self.lowres_first_bottom_ups is not None:
            encoder_params.append(self.lowres_first_bottom_ups.parameters())
        return encoder_params

    def get_decoder_params(self):
        decoder_params = list(self.top_down_layers.parameters()) + list(self.final_top_down.parameters()) + list(
            self.likelihood.parameters())
        return decoder_params

    def get_mixrecons_extra_params(self):
        params = list(self._first_bottom_up_subdset0.parameters())
        if self._interchannel_weights is not None:
            params = params + [self._interchannel_weights]
        return params

    def get_scheduler(self, optimizer):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    self.lr_scheduler_mode,
                                                    patience=self.lr_scheduler_patience,
                                                    factor=0.5,
                                                    min_lr=1e-12,
                                                    verbose=True)

    def configure_optimizers(self):

        encoder_params = self.get_encoder_params()
        decoder_params = self.get_decoder_params()
        # channel 1 params
        ch2_pathway = encoder_params + decoder_params
        optimizer0 = optim.Adamax(ch2_pathway, lr=self.lr, weight_decay=0)

        optimizer1 = optim.Adamax(self.get_mixrecons_extra_params(), lr=self.lr, weight_decay=0)

        scheduler0 = self.get_scheduler(optimizer0)
        scheduler1 = self.get_scheduler(optimizer1)

        return [optimizer0, optimizer1], [{
            'scheduler': scheduler,
            'monitor': self.lr_scheduler_monitor,
        } for scheduler in [scheduler0, scheduler1]]

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target, dset_idx, loss_idx = batch
        ch2_opt, mix_opt = self.optimizers()
        mask_ch2 = loss_idx == LossType.Elbo
        mask_mix = loss_idx == LossType.ElboMixedReconstruction
        assert mask_ch2.sum() + mask_mix.sum() == len(x)
        loss_dict = None
        
        if mask_ch2.sum() > 0:
            batch = (x[mask_ch2], target[mask_ch2], dset_idx[mask_ch2], loss_idx[mask_ch2])
            loss_dict = super().training_step(batch, batch_idx, enable_logging=enable_logging)
            if loss_dict is not None:
                ch2_opt.zero_grad()
                self.manual_backward(loss_dict['loss'])
                ch2_opt.step()
        
        if mask_mix.sum() > 0:
            batch = (x[mask_mix], target[mask_mix], dset_idx[mask_mix], loss_idx[mask_mix])
            mix_loss_dict = super().training_step(batch, batch_idx, enable_logging=enable_logging)
            if loss_dict is not None:
                mix_opt.zero_grad()
                self.manual_backward(mix_loss_dict['loss'])
                mix_opt.step()
        
        if loss_dict is not None:
            self.log_dict({"loss": loss_dict['loss'].item()}, prog_bar=True)
