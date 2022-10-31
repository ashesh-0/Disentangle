""" 
Adapted from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""
import wandb
from disentangle.nets.unet_parts import *
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import numpy as np
from disentangle.core.metric_monitor import MetricMonitor
from disentangle.metrics.running_psnr import RunningPSNR


class UNet(pl.LightningModule):

    def __init__(self, data_mean, data_std, config):
        super(UNet, self).__init__()
        bilinear = True
        self.bilinear = bilinear
        self.lr = config.training.lr
        self.n_levels = config.model.n_levels
        self.lr_scheduler_patience = config.training.lr_scheduler_patience
        self.lr_scheduler_monitor = config.model.get('monitor', 'val_loss')
        self.lr_scheduler_mode = MetricMonitor(self.lr_scheduler_monitor).mode()

        self.inc = DoubleConv(1, 64)
        ch = 64
        for i in range(1, self.n_levels):
            setattr(self, f'down{i}', Down(ch,2*ch))
            ch = 2*ch

        factor = 2 if bilinear else 1
        setattr(self, f'down{self.n_levels}', Down(ch, 2*ch // factor))
        ch = 2*ch
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 1024 // factor)
        for i in range(1,self.n_levels):
            setattr(self,f'up{i}',Up(ch, (ch//2) // factor, bilinear))
            ch = ch//2
        
        setattr(self, f'up{self.n_levels}', Up(ch, ch//2, bilinear))
        ch = ch//2
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(ch, 2)
        self.normalized_input = config.data.normalized_input
        self.data_mean = torch.Tensor(data_mean) if isinstance(data_mean, np.ndarray) else data_mean
        self.data_std = torch.Tensor(data_std) if isinstance(data_std, np.ndarray) else data_std
        self.label1_psnr = RunningPSNR()
        self.label2_psnr = RunningPSNR()

    def forward(self, x):
        x1 = self.inc(x)
        latents = []
        x_end = x1
        for i in range(1,self.n_levels+1):
            latents.append(x_end)
            x_end = getattr(self, f'down{i}')(x_end)

        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)

        for i in range(1,self.n_levels+1):
           x_end = getattr(self, f'up{i}')(x_end,latents[-1*i])

        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # pred = self.outc(x)
        pred = self.outc(x_end)
        return pred

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.parameters(), lr=self.lr, weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         self.lr_scheduler_mode,
                                                         patience=self.lr_scheduler_patience,
                                                         factor=0.5,
                                                         min_lr=1e-12,
                                                         verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.lr_scheduler_monitor}

    def normalize_input(self, x):
        if self.normalized_input:
            return x
        return (x - self.data_mean.mean()) / self.data_std.mean()

    def normalize_target(self, target):
        return (target - self.data_mean) / self.data_std

    def power_of_2(self, x):
        assert isinstance(x, int)
        if x == 1:
            return True
        if x == 0:
            # happens with validation
            return False
        if x % 2 == 1:
            return False
        return self.power_of_2(x // 2)

    def set_params_to_same_device_as(self, correct_device_tensor):
        if isinstance(self.data_mean, torch.Tensor):
            if self.data_mean.device != correct_device_tensor.device:
                self.data_mean = self.data_mean.to(correct_device_tensor.device)
                self.data_std = self.data_std.to(correct_device_tensor.device)

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target = batch
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)

        out = self.forward(x_normalized)
        net_loss = self.get_reconstruction_loss(out, target_normalized)

        self.log('reconstruction_loss', net_loss, on_epoch=True)

        output = {
            'loss': net_loss,
            'reconstruction_loss': net_loss,
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output

    def get_reconstruction_loss(self, reconstruction, input):
        loss_fn = nn.MSELoss()
        return loss_fn(reconstruction, input)

    def validation_step(self, batch, batch_idx):
        x, target = batch
        self.set_params_to_same_device_as(target)

        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)

        out = self.forward(x_normalized)
        recons_img = out
        recons_loss = self.get_reconstruction_loss(out, target_normalized)

        self.log('val_loss', recons_loss, on_epoch=True)
        self.label1_psnr.update(recons_img[:, 0], target_normalized[:, 0])
        self.label2_psnr.update(recons_img[:, 1], target_normalized[:, 1])

        if batch_idx == 0 and self.power_of_2(self.current_epoch):
            sample = self(x_normalized[0:1, ...])

            sample = sample * self.data_std + self.data_mean
            sample = sample.cpu()
            self.log_images_for_tensorboard(sample[:, 0, ...], target[0, 0, ...], 'label1')
            self.log_images_for_tensorboard(sample[:, 1, ...], target[0, 1, ...], 'label2')

    def log_images_for_tensorboard(self, pred, target, label):
        clamped_pred = torch.clamp((pred - pred.min()) / (pred.max() - pred.min()), 0, 1)
        if target is not None:
            clamped_input = torch.clamp((target - target.min()) / (target.max() - target.min()), 0, 1)
            img = wandb.Image(clamped_input[None].cpu().numpy())
            self.logger.experiment.log({f'target_for{label}': img})
            # self.trainer.logger.experiment.add_image(f'target_for{label}', clamped_input[None], self.current_epoch)

        img = wandb.Image(clamped_pred.cpu().numpy())
        self.logger.experiment.log({f'{label}/sample_0': img})

    def on_validation_epoch_end(self):
        psnrl1 = self.label1_psnr.get()
        psnrl2 = self.label2_psnr.get()
        psnr = (psnrl1 + psnrl2) / 2
        self.log('val_psnr', psnr, on_epoch=True)
        self.label1_psnr.reset()
        self.label2_psnr.reset()
