"""
Multi dataset based setup.
"""
import torch

from disentangle.core.loss_type import LossType
from disentangle.core.psnr import RangeInvariantPsnr
from disentangle.nets.lvae import LadderVAE, compute_batch_mean, torch_nanmean


class LadderVaeMultiDataset(LadderVAE):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at, target_ch)
        for dloader_key in self.data_mean.keys():
            assert dloader_key in ['subdset_0', 'subdset_1']
            for data_key in self.data_mean[dloader_key].keys():
                assert data_key in ['target', 'input']
                self.data_mean[dloader_key][data_key] = torch.Tensor(data_mean[dloader_key][data_key])
                self.data_std[dloader_key][data_key] = torch.Tensor(data_std[dloader_key][data_key])

            self.data_mean[dloader_key]['input'] = self.data_mean[dloader_key]['input'].reshape(1, 1, 1, 1)
            self.data_std[dloader_key]['input'] = self.data_std[dloader_key]['input'].reshape(1, 1, 1, 1)

    def get_reconstruction_loss(self,
                                reconstruction,
                                input,
                                dset_idx,
                                loss_type_idx,
                                return_predicted_img=False,
                                likelihood_obj=None):
        output = self._get_reconstruction_loss_vector(reconstruction,
                                                      input,
                                                      dset_idx,
                                                      return_predicted_img=return_predicted_img,
                                                      likelihood_obj=likelihood_obj)
        loss_dict = output[0] if return_predicted_img else output
        individual_ch_loss_mask = loss_type_idx == LossType.Elbo
        mixed_reconstruction_mask = loss_type_idx == LossType.ElboMixedReconstruction
        print(torch.sum(individual_ch_loss_mask), torch.sum(mixed_reconstruction_mask))
        
        if torch.sum(individual_ch_loss_mask) > 0:
            loss_dict['loss'] = torch.mean(loss_dict['loss'][individual_ch_loss_mask])
            loss_dict['ch1_loss'] = torch.mean(loss_dict['ch1_loss'][individual_ch_loss_mask])
            loss_dict['ch2_loss'] = torch.mean(loss_dict['ch2_loss'][individual_ch_loss_mask])
        else:
            loss_dict['loss'] = 0.0
            loss_dict['ch1_loss'] = 0.0
            loss_dict['ch2_loss'] = 0.0

        if torch.sum(mixed_reconstruction_mask) > 0:
            loss_dict['mixed_loss'] = torch.mean(loss_dict['mixed_loss'][mixed_reconstruction_mask])
        else:
            loss_dict['mixed_loss'] = 0.0

        if return_predicted_img:
            assert len(output) == 2
            return loss_dict, output[1]
        else:
            return loss_dict

    def normalize_target(self, target, dataset_index):
        dataset_index = dataset_index[:, None, None, None]
        mean = self.data_mean['subdset_0']['target'] * (
            1 - dataset_index) + self.data_mean['subdset_1']['target'] * dataset_index
        std = self.data_std['subdset_0']['target'] * (
            1 - dataset_index) + self.data_std['subdset_1']['target'] * dataset_index
        return (target - mean) / std

    def _get_reconstruction_loss_vector(self,
                                        reconstruction,
                                        input,
                                        dset_idx,
                                        return_predicted_img=False,
                                        likelihood_obj=None):
        """
        Args:
            return_predicted_img: If set to True, the besides the loss, the reconstructed image is also returned.
        """

        if likelihood_obj is None:
            likelihood_obj = self.likelihood
        # Log likelihood
        ll, like_dict = likelihood_obj(reconstruction, input)
        if self.skip_nboundary_pixels_from_loss is not None and self.skip_nboundary_pixels_from_loss > 0:
            pad = self.skip_nboundary_pixels_from_loss
            ll = ll[:, :, pad:-pad, pad:-pad]
            like_dict['params']['mean'] = like_dict['params']['mean'][:, :, pad:-pad, pad:-pad]

        assert ll.shape[1] == 2, f"Change the code below to handle >2 channels first. ll.shape {ll.shape}"
        output = {
            'loss': compute_batch_mean(-1 * ll),
            'ch1_loss': compute_batch_mean(-ll[:, 0]),
            'ch2_loss': compute_batch_mean(-ll[:, 1]),
        }
        if self.channel_1_w is not None or self.channel_2_w is not None:
            output['loss'] = (self.channel_1_w * output['ch1_loss'] +
                              self.channel_2_w * output['ch2_loss']) / (self.channel_1_w + self.channel_2_w)

        if self.enable_mixed_rec:
            data_mean, data_std = self.get_mean_std_for_one_batch(dset_idx, self.data_mean, self.data_std)
            mixed_pred, mixed_logvar = self.get_mixed_prediction(like_dict['params']['mean'],
                                                                 like_dict['params']['logvar'], data_mean, data_std)
            mixed_target = input
            mixed_recons_ll = self.likelihood.log_likelihood(mixed_target, {'mean': mixed_pred, 'logvar': mixed_logvar})
            output['mixed_loss'] = compute_batch_mean(-1 * mixed_recons_ll)

        if return_predicted_img:
            return output, like_dict['params']['mean']

        return output

    @staticmethod
    def get_mean_std_for_one_batch(dset_idx, data_mean, data_std):
        """
        For each element in the batch, pick the relevant mean and stdev on the basis of which dataset it is coming from.
        """
        # to make it work as an index
        dset_idx = dset_idx.type(torch.long)
        batch_data_mean = {}
        batch_data_std = {}
        for key in data_mean['subdset_0'].keys():
            assert key in ['target', 'input']
            combined = torch.cat([data_mean['subdset_0'][key], data_mean['subdset_1'][key]], dim=0)
            batch_values = combined[dset_idx]
            batch_data_mean[key] = batch_values
            combined = torch.cat([data_std['subdset_0'][key], data_std['subdset_1'][key]], dim=0)
            batch_values = combined[dset_idx]
            batch_data_std[key] = batch_values

        return batch_data_mean, batch_data_std

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target, dset_idx, loss_idx = batch
        assert self.normalized_input == True
        x_normalized = x
        target_normalized = self.normalize_target(target, dset_idx)

        out, td_data = self.forward(x_normalized)

        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict = self.get_reconstruction_loss(out,
                                                        target_normalized,
                                                        dset_idx,
                                                        loss_idx,
                                                        return_predicted_img=False)

        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        recons_loss = recons_loss_dict['loss']
        if self.loss_type == LossType.ElboMixedReconstruction:
            recons_loss += self.mixed_rec_w * recons_loss_dict['mixed_loss']

            if enable_logging:
                self.log('mixed_reconstruction_loss', recons_loss_dict['mixed_loss'], on_epoch=True)

        if self.non_stochastic_version:
            kl_loss = torch.Tensor([0.0]).cuda()
            net_loss = recons_loss
        else:
            kl_loss = self.get_kl_divergence_loss(td_data)
            net_loss = recons_loss + self.get_kl_weight() * kl_loss

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
            'reconstruction_loss': recons_loss.detach(),
            'kl_loss': kl_loss.detach(),
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output

    def set_params_to_same_device_as(self, correct_device_tensor):
        for dataset_index in [0, 1]:
            str_idx = f'subdset_{dataset_index}'
            if str_idx in self.data_mean and isinstance(self.data_mean[str_idx]['target'], torch.Tensor):
                if self.data_mean[str_idx]['target'].device != correct_device_tensor.device:
                    self.data_mean[str_idx]['target'] = self.data_mean[str_idx]['target'].to(
                        correct_device_tensor.device)
                    self.data_std[str_idx]['target'] = self.data_std[str_idx]['target'].to(correct_device_tensor.device)

                    self.data_mean[str_idx]['input'] = self.data_mean[str_idx]['input'].to(correct_device_tensor.device)
                    self.data_std[str_idx]['input'] = self.data_std[str_idx]['input'].to(correct_device_tensor.device)

                    self.likelihood.set_params_to_same_device_as(correct_device_tensor)
                else:
                    return

    def validation_step(self, batch, batch_idx):
        x, target, dset_idx, loss_idx = batch
        self.set_params_to_same_device_as(target)

        x_normalized = x
        target_normalized = self.normalize_target(target, dset_idx)

        out, td_data = self.forward(x_normalized)
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict, recons_img = self.get_reconstruction_loss(out,
                                                                    target_normalized,
                                                                    dset_idx,
                                                                    loss_idx,
                                                                    return_predicted_img=True)
        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        self.label1_psnr.update(recons_img[:, 0], target_normalized[:, 0])
        self.label2_psnr.update(recons_img[:, 1], target_normalized[:, 1])

        psnr_label1 = RangeInvariantPsnr(target_normalized[:, 0].clone(), recons_img[:, 0].clone())
        psnr_label2 = RangeInvariantPsnr(target_normalized[:, 1].clone(), recons_img[:, 1].clone())
        recons_loss = recons_loss_dict['loss']
        # kl_loss = self.get_kl_divergence_loss(td_data)
        # net_loss = recons_loss + self.get_kl_weight() * kl_loss
        self.log('val_loss', recons_loss, on_epoch=True)
        val_psnr_l1 = torch_nanmean(psnr_label1).item()
        val_psnr_l2 = torch_nanmean(psnr_label2).item()
        self.log('val_psnr_l1', val_psnr_l1, on_epoch=True)
        self.log('val_psnr_l2', val_psnr_l2, on_epoch=True)
        # self.log('val_psnr', (val_psnr_l1 + val_psnr_l2) / 2, on_epoch=True)

        if batch_idx == 0 and self.power_of_2(self.current_epoch):
            all_samples = []
            for i in range(20):
                sample, _ = self(x_normalized[0:1, ...])
                sample = self.likelihood.get_mean_lv(sample)[0]
                all_samples.append(sample[None])

            all_samples = torch.cat(all_samples, dim=0)
            data_mean, data_std = self.get_mean_std_for_one_batch(dset_idx, self.data_mean, self.data_std)
            all_samples = all_samples * data_std['target'] + data_mean['target']
            all_samples = all_samples.cpu()
            img_mmse = torch.mean(all_samples, dim=0)[0]
            self.log_images_for_tensorboard(all_samples[:, 0, 0, ...], target[0, 0, ...], img_mmse[0], 'label1')
            self.log_images_for_tensorboard(all_samples[:, 0, 1, ...], target[0, 1, ...], img_mmse[1], 'label2')


if __name__ == '__main__':
    data_mean = {
        'subdset_0': {
            'target': torch.Tensor([1.1, 3.2]).reshape((1, 2, 1, 1)),
            'input': torch.Tensor([1366]).reshape((1, 1, 1, 1))
        },
        'subdset_1': {
            'target': torch.Tensor([15, 30]).reshape((1, 2, 1, 1)),
            'input': torch.Tensor([10]).reshape((1, 1, 1, 1))
        }
    }

    data_std = {
        'subdset_0': {
            'target': torch.Tensor([21, 45]).reshape((1, 2, 1, 1)),
            'input': torch.Tensor([955]).reshape((1, 1, 1, 1))
        },
        'subdset_1': {
            'target': torch.Tensor([90, 2]).reshape((1, 2, 1, 1)),
            'input': torch.Tensor([121]).reshape((1, 1, 1, 1))
        }
    }

    dset_idx = torch.Tensor([0, 0, 0, 1, 1, 0])

    mean, std = LadderVaeMultiDataset.get_mean_std_for_one_batch(dset_idx, data_mean, data_std)
