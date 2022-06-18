import torch

from disentangle.nets.lvae import LadderVAE
import torch.optim as optim


class LadderVaeSepVampprior(LadderVAE):
    def __init__(self, *args, **kwargs):
        super(LadderVaeSepVampprior, self).__init__(*args, **kwargs)
        # this will be used to disable/enable gradient flow in the decoder when the learnable vamprior inputs are the
        # input for the first validation call, the value is set to 0. 1 would be just as fine.
        self._optimizer_idx = 0

    def configure_optimizers(self):
        params1 = list(self.first_bottom_up.parameters()) + list(self.bottom_up_layers.parameters()) + list(
            self.top_down_layers.parameters()) + list(self.final_top_down.parameters()) + list(
            self.likelihood.parameters())

        optimizer1 = optim.Adamax(params1, lr=self.lr, weight_decay=0)
        optimizer2 = optim.Adamax(self.vp_means.parameters(), lr=self.lr, weight_decay=0)

        scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1,
                                                          'min',
                                                          patience=self.lr_scheduler_patience,
                                                          factor=0.5,
                                                          min_lr=1e-12,
                                                          verbose=True)
        scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2,
                                                          'min',
                                                          patience=self.lr_scheduler_patience,
                                                          factor=0.5,
                                                          min_lr=1e-12,
                                                          verbose=True)

        return [optimizer1, optimizer2], [{
            'scheduler': scheduler1,
            'monitor': 'val_loss'
        }, {
            'scheduler': scheduler2,
            'monitor': 'val_loss'
        }]

    def _vp_compute_mu_logvar(self):
        """
        the Idea here is that decoder weights should not get any gradient due to us passing the trainable input through them.
        So, in one optimizer, the mu and sigma of prior is taken to be as non-trainable. Note that the last part of the
        decoder happens in stochastic.py where a transform is applied on the decoder's output to access the mu and logvar
        So, ideally, that portion should also be put under torch.no_grad. However, this should give us some indication of
        whether our idea is correct or not.
        Also, note that the second optimizer does not update the decoder weights. So, the second optimizer will just
        update the inputs.
        """
        if self._optimizer_idx == 0:
            with torch.no_grad():
                output = super()._vp_compute_mu_logvar()
        elif self._optimizer_idx == 1:
            output = super()._vp_compute_mu_logvar()
        else:
            raise ValueError(f"Invalid self._optimizer_idx:{self._optimizer_idx}")
        return output

    def training_step(self, batch, batch_idx, optimizer_idx: int):
        self._optimizer_idx = optimizer_idx
        if optimizer_idx == 0:
            return super().training_step(batch, batch_idx, enable_logging=True)
        elif optimizer_idx == 1:
            return super().training_step(batch, batch_idx, enable_logging=False)
