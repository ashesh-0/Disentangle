"""
Model with combines VAE with critic. Critic is used to enfore a prior on the generated images.
"""
import torch
import torch.optim as optim
from torch import nn

from disentangle.nets.lvae import LadderVAE
from disentangle.nets.texture_classifier import TextureEncoder
from disentangle.core.loss_type import LossType

class LadderVAETexDiscrim(LadderVAE):
    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=target_ch)
        self.D1 = TextureEncoder(with_sigmoid=False)
        self.D2 = TextureEncoder(with_sigmoid=False)
        self.automatic_optimization = False


        self.critic_loss_weight = config.loss.critic_loss_weight
        self.critic_loss_fn = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        params1 = list(self.first_bottom_up.parameters()) + list(self.bottom_up_layers.parameters()) + list(
            self.top_down_layers.parameters()) + list(self.final_top_down.parameters()) + list(
            self.likelihood.parameters())

        optimizer1 = optim.Adamax(params1, lr=self.lr, weight_decay=0)
        params2 = list(self.D1.parameters()) + list(self.D2.parameters())
        optimizer2 = optim.Adamax(params2, lr=self.lr, weight_decay=0)

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

    def get_critic_loss_stats(self, pred_normalized: torch.Tensor, target_normalized: torch.Tensor) -> dict:
        """
        This function takes as input one batch of predicted image (both labels) and target images and returns the 
        crossentropy loss.
        Args:
            pred_normalized: The predicted (normalized) images. Note that this is not the output of the forward().
                            Likelihood module is also applied on top of it to produce the image.
            target_normalized: This is the normalized target images.
        """
        pred1, pred2 = pred_normalized.chunk(2, dim=1)
        tar1, tar2 = target_normalized.chunk(2, dim=1)
        loss1, avg_pred_dict1 = self.get_critic_loss(pred1, tar1, self.D1)
        loss2, avg_pred_dict2 = self.get_critic_loss(pred2, tar2, self.D2)
        return {
            'loss': (loss1 + loss2) / 2,
            'loss_Label1': loss1,
            'loss_Label2': loss2,
            'avg_Label1': avg_pred_dict1,
            'avg_Label2': avg_pred_dict2,
        }

    def get_critic_loss(self, pred: torch.Tensor, tar: torch.Tensor, D):
        """
        Given a predicted image and a target image, here we return a binary crossentropy loss.
        discriminator is trained to predict 1 for target image and 0 for the predicted image.
        Args:
            pred: predicted image
            tar: target image
            D: discriminator model
        """
        pred_label = D(pred)
        tar_label = D(tar)
        loss_0 = self.critic_loss_fn(pred_label, torch.zeros_like(pred_label))
        loss_1 = self.critic_loss_fn(tar_label, torch.ones_like(tar_label))
        loss = loss_0 + loss_1
        return loss, {'generated': torch.sigmoid(pred_label).mean(), 'actual': torch.sigmoid(tar_label).mean()}

    def training_step(self, batch: tuple, batch_idx: int):
        optimizer_g, optimizer_d = self.optimizers()

        x, target = batch
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)
        mask = ~((target == 0).reshape(len(target), -1).all(dim=1))

        optimizer_g.zero_grad()
        out, td_data = self.forward(x_normalized)
        recons_loss_dict, pred_nimg = self.get_reconstruction_loss(out, target_normalized,
                                                                   x_normalized, 
                                                                   splitting_mask=mask, return_predicted_img=True)
        recons_loss = recons_loss_dict['loss']
        if self.loss_type == LossType.ElboMixedReconstruction:
            recons_loss += self.mixed_rec_w * recons_loss_dict['mixed_loss']


        if self.non_stochastic_version:
            kl_loss = torch.Tensor([0.0]).cuda()
        else:
            kl_loss = self.get_kl_divergence_loss(td_data)

        net_loss = recons_loss + self.get_kl_weight() * kl_loss
        
        if mask.sum() > 0:
            critic_dict = self.get_critic_loss_stats(pred_nimg, target_normalized[mask])
            D_loss = critic_dict['loss']

            # Note the negative here. It will aim to maximize the discriminator loss.
            net_loss += -1 * self.critic_loss_weight * D_loss

        self.manual_backward(net_loss)
        optimizer_g.step()

        # for i, x in enumerate(td_data['debug_qvar_max']):
        #     self.log(f'qvar_max:{i}', x.item(), on_epoch=True)

        self.log('reconstruction_loss', recons_loss, on_epoch=True)
        # self.log('kl_loss', kl_loss, on_epoch=True)
        self.log('training_loss', net_loss, on_epoch=True)
        self.log('D_loss', D_loss, on_epoch=True)
        self.log('L1_generated_probab', critic_dict['avg_Label1']['generated'], on_epoch=True)
        self.log('L1_actual_probab', critic_dict['avg_Label1']['actual'], on_epoch=True)
        self.log('L2_generated_probab', critic_dict['avg_Label2']['generated'], on_epoch=True)
        self.log('L2_actual_probab', critic_dict['avg_Label2']['actual'], on_epoch=True)
        if mask.sum() > 0:
            optimizer_d.zero_grad()
            D_loss = self.critic_loss_weight * self.get_critic_loss_stats(pred_nimg.detach(), target_normalized[mask])['loss']
            self.manual_backward(D_loss)
            optimizer_d.step()
