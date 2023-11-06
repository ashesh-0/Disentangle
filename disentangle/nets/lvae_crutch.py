"""
Iterative update methodology. 
"""
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from disentangle.nets.lvae import LadderVAE


class CrutchModel(pl.LightningModule):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__()
        self.student = LadderVAE(data_mean, data_std, config)
        self.teacher = LadderVAE(data_mean, data_std, config)
        self._workdir = config.workdir
        # TODO: compute the best psnr from the checkpoint
        self.best_psnr = None
        self.last_psnr = None
        self._divergence_loss_w = config.loss.divergence_loss_w

        for param in self.teacher.parameters():
            param.requires_grad = False

        ckpt_fpath = config.model.pretrained_weights_path
        checkpoint = torch.load(ckpt_fpath)
        _ = self.student.load_state_dict(checkpoint['state_dict'], strict=True)
        checkpoint = torch.load(ckpt_fpath)
        _ = self.teacher.load_state_dict(checkpoint['state_dict'], strict=True)

    def update_teacher(self):
        state_dict = self.student.state_dict()
        _ = self.teacher.load_state_dict(state_dict, strict=True)

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.student.parameters(), lr=self.student.lr, weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         self.student.lr_scheduler_mode,
                                                         patience=self.student.lr_scheduler_patience,
                                                         factor=0.5,
                                                         min_lr=1e-12,
                                                         verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.student.lr_scheduler_monitor}

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
            self.best_psnr = self.last_psnr / 2
            self.last_psnr = self.best_psnr

        # self.log('val_psnr', self.last_psnr, on_epoch=True)
        # if self.best_psnr is None:
        #     self.best_psnr = self.last_psnr
        #     for param in self.student.parameters():
        #         assert param.requires_grad == False
        #         param.requires_grad = True

    def training_step(self, batch, batch_idx):
        if self.last_psnr is not None and self.best_psnr is not None and self.last_psnr > self.best_psnr:
            self.update_teacher()
            print('Updating teacher', self.last_psnr, self.best_psnr)
            import pdb
            pdb.set_trace()
            self.best_psnr = self.last_psnr

        student_dict = self.student.training_step(batch, batch_idx, log_fn=self.log)
        td_data_student = student_dict['td_data']
        # compute the divergence loss between the student and the teacher
        out_teacher, td_data_teacher = self.teacher(student_dict['x_normalized'])

        diff = torch.nn.MSELoss()(out_teacher, student_dict['out'])
        n = 1
        for i in range(len(td_data_teacher['bu_values'])):
            diff += torch.nn.MSELoss()(td_data_teacher['bu_values'][i], td_data_student['bu_values'][i])
            n += 1

        for i in range(len(td_data_teacher['z'])):
            diff += torch.nn.MSELoss()(td_data_teacher['z'][i], td_data_student['z'][i])
            n += 1

        diff /= n
        self.log('divergence_loss', diff, on_epoch=True)
        student_dict['loss'] += self._divergence_loss_w * diff

        return {'loss': student_dict['loss']}

    def validation_step(self, batch, batch_idx):
        return self.student.validation_step(batch, batch_idx, log_fn=self.log)
