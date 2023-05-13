import torch.optim as optim

from disentangle.core.loss_type import LossType
from disentangle.nets.lvae_multidset_multi_input_branches import LadderVaeMultiDatasetMultiBranch


class LadderVaeMultiDatasetMultiOptim(LadderVaeMultiDatasetMultiBranch):

    def get_encoder_params(self):
        encoder_params = list(self._first_bottom_up_subdset0.parameters()) + list(self.bottom_up_layers.parameters())
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
            params = params + self._interchannel_weights.parameters()
        return params

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

    def training_step(self, batch, batch_idx, optimizer_idx, enable_logging=True):
        x, target, dset_idx, loss_idx = batch
        if optimizer_idx == 0:
            mask = loss_idx == LossType.Elbo
        elif optimizer_idx == 1:
            mask = loss_idx == LossType.ElboMixedReconstruction

        if mask.sum() > 0:
            batch = (x[mask], target[mask], dset_idx[mask], loss_idx[mask])
            return super().training_step(batch, batch_idx, enable_logging=enable_logging)
        else:
            print('There is no element for optimizer', optimizer_idx)
