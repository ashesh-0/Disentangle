import pytorch_lightning as pl
from disentangle.nets.lvae import LadderVAE
from disentangle.nets.texture_classifier import TextureEncoder
import torchvision.transforms.functional as F
from disentangle.core.loss_type import LossType

import torch

class LadderVAEWithClassifier(LadderVAE):
    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2)
        self.classifier = TextureEncoder()
        self.classifier.load_state_dict(torch.load(config.model.classifier_fpath))        
        print('Loaded classifier from {}'.format(config.model.classifier_fpath))
        for param in self.classifier.parameters():
            param.requires_grad = False

        self._classifier_loss_weight = config.model.classifier_loss_weight



    def training_step(self, batch, batch_idx, enable_logging=True):
        if self.current_epoch == 0 and batch_idx == 0:
            self.log('val_psnr', 1.0, on_epoch=True)

        x, target = batch[:2]
        x_normalized = self.normalize_input(x)
        if self.reconstruction_mode:
            target_normalized = x_normalized[:, :1].repeat(1, 2, 1, 1)
            target = None
            mask = None
        else:
            target_normalized = self.normalize_target(target)
            mask = ~((target == 0).reshape(len(target), -1).all(dim=1))

        out, td_data = self.forward(x_normalized)
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict, imgs = self.get_reconstruction_loss(out,
                                                              target_normalized,
                                                              x_normalized,
                                                              mask,
                                                              return_predicted_img=True)

        

        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        recons_loss = recons_loss_dict['loss']
        if torch.isnan(recons_loss).any():
            recons_loss = 0.0

        if self.loss_type == LossType.ElboMixedReconstruction:
            recons_loss += self.mixed_rec_w * recons_loss_dict['mixed_loss']

            if enable_logging:
                self.log('mixed_reconstruction_loss', recons_loss_dict['mixed_loss'], on_epoch=True)

        # classifier loss. 
        ch0_class_pred = self.classifier(imgs[:, :1])
        ch1_class_pred = self.classifier(imgs[:, 1:])
        # remove all frames where one of the predictions was totally empty.
        ch0_mask = torch.logical_and(ch0_class_pred > 0.4, ch0_class_pred < 0.6).float()
        ch1_mask = torch.logical_and(ch1_class_pred > 0.4, ch1_class_pred < 0.6).float()

        ch0_mask = torch.mean(ch0_mask, dim=(1, 2, 3))  
        ch1_mask = torch.mean(ch1_mask, dim=(1, 2, 3))
        ch0_class_pred = ch0_class_pred[ch0_mask < 0.9]
        ch1_class_pred = ch1_class_pred[ch1_mask < 0.9]

        classifier_loss = torch.mean(ch0_class_pred) + (1 - torch.mean(ch1_class_pred))
        recons_loss += self._classifier_loss_weight * classifier_loss
        if enable_logging:
            self.log('classifier_loss', classifier_loss, on_epoch=True)


        if self.non_stochastic_version:
            kl_loss = torch.Tensor([0.0]).cuda()
            net_loss = recons_loss
        else:
            kl_loss = self.get_kl_divergence_loss(td_data)
            net_loss = recons_loss + self.get_kl_weight() * kl_loss

        # print(f'rec:{recons_loss_dict["loss"]:.3f} mix: {recons_loss_dict.get("mixed_loss",0):.3f} KL: {kl_loss:.3f}')
        if enable_logging:
            for i, x in enumerate(td_data['debug_qvar_max']):
                self.log(f'qvar_max:{i}', x.item(), on_epoch=True)

            self.log('reconstruction_loss', recons_loss_dict['loss'], on_epoch=True)
            self.log('kl_loss', kl_loss, on_epoch=True)
            self.log('training_loss', net_loss, on_epoch=True)
            self.log('lr', self.lr, on_epoch=True)
            # self.log('grad_norm_bottom_up', self.grad_norm_bottom_up, on_epoch=True)
            # self.log('grad_norm_top_down', self.grad_norm_top_down, on_epoch=True)

        output = {
            'loss': net_loss,
            'reconstruction_loss': recons_loss.detach() if isinstance(recons_loss, torch.Tensor) else recons_loss,
            'kl_loss': kl_loss.detach(),
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output