"""
Iterative update methodology. 
"""
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from disentangle.core.loss_type import LossType
from disentangle.nets.lvae import LadderVAE
from disentangle.nets.lvae_with_texturediscriminator import LadderVAETexDiscrim


class CrutchModel(pl.LightningModule):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__()
        self.automatic_optimization = False
        self.student = LadderVAETexDiscrim(data_mean, data_std, config)
        self.teacher = LadderVAETexDiscrim(data_mean, data_std, config)
        self._workdir = config.workdir
        # TODO: compute the best psnr from the checkpoint
        self.best_psnr = None
        self.last_psnr = None
        self._divergence_loss_w = config.loss.divergence_loss_w

        for param in self.teacher.parameters():
            param.requires_grad = False

        ckpt_fpath = config.model.pretrained_weights_path
        checkpoint = torch.load(ckpt_fpath)
        _ = self.student.load_state_dict(checkpoint['state_dict'], strict=False)
        checkpoint = torch.load(ckpt_fpath)
        _ = self.teacher.load_state_dict(checkpoint['state_dict'], strict=False)

    def update_teacher(self):
        state_dict = self.student.state_dict()
        _ = self.teacher.load_state_dict(state_dict, strict=False)

    def configure_optimizers(self):
        return self.student.configure_optimizers()

    def set_params_to_same_device_as(self, correct_device_tensor):
        self.student.set_params_to_same_device_as(correct_device_tensor)
        self.teacher.set_params_to_same_device_as(correct_device_tensor)

    def reset_for_different_output_size(self, output_size):
        self.student.reset_for_different_output_size(output_size)
        self.teacher.reset_for_different_output_size(output_size)

    def on_validation_epoch_end(self):
        self.last_psnr = self.student.on_validation_epoch_end(log_fn=self.log)
        print('\nPSNR', self.current_epoch, self.last_psnr)
        if self.best_psnr is None:
            self.best_psnr = self.last_psnr
            self.last_psnr = self.best_psnr

        # self.log('val_psnr', self.last_psnr, on_epoch=True)
        # if self.best_psnr is None:
        #     self.best_psnr = self.last_psnr
        #     for param in self.student.parameters():
        #         assert param.requires_grad == False
        #         param.requires_grad = True

    def get_divergence_loss(self, x_normalized, out_student, td_data_student):
        out_teacher, td_data_teacher = self.teacher(x_normalized)

        diff = torch.nn.MSELoss()(out_teacher, out_student)
        n = 1
        for i in range(len(td_data_teacher['bu_values'])):
            diff += torch.nn.MSELoss()(td_data_teacher['bu_values'][i], td_data_student['bu_values'][i])
            n += 1

        for i in range(len(td_data_teacher['z'])):
            diff += torch.nn.MSELoss()(td_data_teacher['z'][i], td_data_student['z'][i])
            n += 1

        diff /= n
        return diff

    def training_step(self, batch, batch_idx):
        if self.last_psnr is not None and self.best_psnr is not None and self.last_psnr > self.best_psnr:
            self.update_teacher()
            print('Updating teacher', self.last_psnr, self.best_psnr)
            self.best_psnr = self.last_psnr

        optimizer_g, optimizer_d = self.optimizers()

        x, target = batch
        x_normalized = self.student.normalize_input(x)
        target_normalized = self.student.normalize_target(target)
        mask = ~((target == 0).reshape(len(target), -1).all(dim=1))

        optimizer_g.zero_grad()
        out, td_data_student = self.student.forward(x_normalized)
        recons_loss_dict, pred_nimg = self.student.get_reconstruction_loss(out,
                                                                           target_normalized,
                                                                           x_normalized,
                                                                           splitting_mask=mask,
                                                                           return_predicted_img=True)
        recons_loss = recons_loss_dict['loss']
        if self.student.loss_type == LossType.ElboMixedReconstruction:
            recons_loss += self.student.mixed_rec_w * recons_loss_dict['mixed_loss']

        if self.student.non_stochastic_version:
            kl_loss = torch.Tensor([0.0]).cuda()
        else:
            kl_loss = self.student.get_kl_divergence_loss(td_data_student)

        div_loss = self.get_divergence_loss(x_normalized, out, td_data_student)
        self.log('divergence_loss', div_loss, on_epoch=True)

        net_loss = recons_loss + self.student.get_kl_weight() * kl_loss + self._divergence_loss_w * div_loss

        if mask.sum() > 0:
            critic_dict = self.student.get_critic_loss_stats(pred_nimg, target_normalized[mask])
            D_loss = critic_dict['loss']

            # Note the negative here. It will aim to maximize the discriminator loss.
            net_loss += -1 * self.student.critic_loss_weight * D_loss

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
            D_loss = self.student.critic_loss_weight * self.student.get_critic_loss_stats(
                pred_nimg.detach(), target_normalized[mask])['loss']
            self.manual_backward(D_loss)
            optimizer_d.step()

    def validation_step(self, batch, batch_idx):
        return self.student.validation_step(batch, batch_idx, log_fn=self.log)
