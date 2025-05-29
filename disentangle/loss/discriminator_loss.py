# Adapted from https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py#L296
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable

from disentangle.nets.discriminator import Discriminator, LatentDiscriminator


class RealData:
    def __init__(self, data):
        self.data = data

    def get(self, count):
        idx = torch.randint(0, len(self.data), (count,))
        return torch.Tensor(self.data[idx])

class DiscriminatorLoss(nn.Module):
    def __init__(self, num_channels=2, gradient_penalty_lambda=0.1, dense_discriminator=False, loss_mode='wgan', realimg_key='gt', fakeimg_key='pred_FP1',
                 loss_scalar:Union[float,Tuple]=1.0, 
                 train_G_on_both_real_and_fake=False,
                 only_one_channel_idx=None, 
                 embedding_network=None):
        super().__init__()
        assert num_channels ==1 or only_one_channel_idx is None, "If num_channels > 1, only_one_channel_idx must be None"
        self.discriminator_network = Discriminator(num_channels, dense=dense_discriminator)
        if embedding_network is None:
            self.D = self.discriminator_network
        else:
            self.D = LatentDiscriminator(self.discriminator_network, embedding_network)

        self.gp_lambda = gradient_penalty_lambda
        self.loss_mode = loss_mode

        self.realkey = realimg_key
        self.fakekey = fakeimg_key
        if isinstance(loss_scalar, tuple) or isinstance(loss_scalar, list):
            self.loss_scalar_D = loss_scalar[0]
            self.loss_scalar_G = loss_scalar[1]
        else:
            self.loss_scalar_D = loss_scalar
            self.loss_scalar_G = loss_scalar

        self._ch_idx = only_one_channel_idx
        self._train_G_on_both_real_and_fake = train_G_on_both_real_and_fake
        
        # groundtruth, prediction at first forward pass, prediction at second forward pass
        assert self.realkey in ['inp','gt', 'pred_FP1', 'pred_FP2','predInp1', 'pred_FP1_aug','inv_inp2'], f"Invalid discriminator real image key: {self.realkey}. Must be 'gt', 'pred_FP1' or 'pred_FP2'."
        assert self.fakekey in ['inp','gt', 'pred_FP1', 'pred_FP2','predInp1', 'pred_FP1_aug','inv_inp2'], f"Invalid discriminator fake image key: {self.fakekey}. Must be 'gt', 'pred_FP1' or 'pred_FP2'."
        print(f'{self.__class__.__name__} RKey: {self.realkey}, FKey: {self.fakekey} Ch: {self._ch_idx} GP: {self.gp_lambda} LossMode: {self.loss_mode}, TrainGBoth: {self._train_G_on_both_real_and_fake} w_D: {self.loss_scalar_D} w_G: {self.loss_scalar_G}')
    
    def update_gradients_with_generator_loss(self, fake_images, real_images=None, return_loss_without_update=False):
        return update_gradients_with_generator_loss(self.D, fake_images, mode=self.loss_mode, real_images=real_images, loss_scalar=self.loss_scalar_G, return_loss_without_update=return_loss_without_update)
    
    def update_gradients_with_discriminator_loss(self, real_imgs, fake_imgs, return_loss_without_update=False):
        return update_gradients_with_discriminator_loss(self.D, real_imgs, fake_imgs, lambda_term=self.gp_lambda, mode=self.loss_mode, enable_gradient_penalty=self.gp_lambda > 0, loss_scalar=self.loss_scalar_D, return_loss_without_update=return_loss_without_update)
    
    def G_loss(self, data_dict, return_loss_without_update=False):
        fakedata = data_dict[self.fakekey]
        realdata = data_dict[self.realkey] if self._train_G_on_both_real_and_fake else None

        if self._ch_idx is not None:
            fakedata = fakedata[:,self._ch_idx:self._ch_idx+1]
            realdata = realdata[:,self._ch_idx:self._ch_idx+1] if realdata is not None else None

        return self.update_gradients_with_generator_loss(fakedata, real_images=realdata, return_loss_without_update=return_loss_without_update)
    
    def D_loss(self, data_dict, return_loss_without_update=False):
        real_img = data_dict[self.realkey].detach() if self.realkey != 'inp' else None
        fake_img = data_dict[self.fakekey].detach()
        if self._ch_idx is not None:
            real_img = real_img[:,self._ch_idx:self._ch_idx+1]
            fake_img = fake_img[:,self._ch_idx:self._ch_idx+1]
        return self.update_gradients_with_discriminator_loss(real_img, fake_img, return_loss_without_update=return_loss_without_update)



class DiscriminatorLossWithExistingData(DiscriminatorLoss):
    def __init__(self, external_data:Union[None, np.ndarray], use_external_data_probability=1.0,**kwargs):
        super().__init__(**kwargs)
        if external_data is None:
            assert use_external_data_probability == 0.0, "If external_data is None, use_external_data_probability must be 0.0"
        
        self.external_data = RealData(external_data)
        self.p = use_external_data_probability

    def get_external_data(self, count):
        return self.external_data.get(count)
    
    def update_gradients_with_discriminator_loss(self, real_imgs, fake_imgs, return_loss_without_update=False):
        if torch.rand(1).item() < self.p:
            # print('replacing real images with external data')
            real_imgs = self.get_external_data(len(fake_imgs)).to(fake_imgs.device)
            if self._ch_idx is not None:
                real_imgs = real_imgs[:,self._ch_idx:self._ch_idx+1]

        return update_gradients_with_discriminator_loss(self.D, real_imgs, fake_imgs, lambda_term=self.gp_lambda, mode=self.loss_mode, enable_gradient_penalty=self.gp_lambda > 0, loss_scalar=self.loss_scalar_D, return_loss_without_update=return_loss_without_update)

    

def calculate_gradient_penalty(real_images, fake_images, discriminator, lambda_term):
    batch_size = len(real_images)
    eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3)).to(fake_images.device)

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).to(fake_images.device),
                            create_graph=True, retain_graph=True)[0]

    # flatten the gradients to it calculates norm batchwise
    gradients = gradients.view(gradients.size(0), -1)

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty

def increase_value(x, mode, loss_scalar=1.0,retain_graph=False, verbose=False, verbose_name=''):
    if mode == 'wgan':
        x.backward(gradient=-1*loss_scalar*(torch.ones_like(x)), retain_graph=retain_graph)
    elif mode == '-1_1':
        loss = torch.nn.MSELoss()(x, torch.ones_like(x))*loss_scalar
        if verbose:
            print(f'[{verbose_name}] increasing value loss term', loss.item())
        loss.backward(retain_graph=retain_graph)
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes are 'wgan' and '-1_1'.")

def decrease_value(x, mode, loss_scalar=1.0,retain_graph=False):
    if mode == 'wgan':
        x.backward(gradient=loss_scalar*torch.ones_like(x), retain_graph=retain_graph)
    elif mode == '-1_1':
        loss = torch.nn.MSELoss()(x, -torch.ones_like(x))*loss_scalar
        loss.backward(retain_graph=retain_graph)

def update_gradients_with_generator_loss(discriminator, fake_images, real_images=None, mode='wgan', loss_scalar=1.0, return_loss_without_update=False):
    d_pred_fake = discriminator(fake_images)
    d_pred_fake = d_pred_fake.mean()
    if not return_loss_without_update:
        increase_value(d_pred_fake, mode, loss_scalar=loss_scalar, retain_graph=real_images is not None)#, verbose=True, verbose_name='G_loss')

    if real_images is not None:
        d_pred_real = discriminator(real_images)
        d_pred_real = d_pred_real.mean()
        if not return_loss_without_update:
            decrease_value(d_pred_real, mode, loss_scalar=loss_scalar)

    return {'g_loss':d_pred_fake.item(), 'g_loss_real':d_pred_real.item() if real_images is not None else None}

def update_gradients_with_discriminator_loss(discriminator, real_images, fake_images, lambda_term, 
                                             loss_scalar=1.0,mode='wgan', enable_gradient_penalty=True, return_loss_without_update=False):
    d_pred_real = discriminator(real_images)
    d_pred_real = d_pred_real.mean()
    if not return_loss_without_update:
        increase_value(d_pred_real, mode, loss_scalar=loss_scalar, retain_graph=True)

    d_pred_fake = discriminator(fake_images)
    d_pred_fake = d_pred_fake.mean()
    if not return_loss_without_update:
        decrease_value(d_pred_fake, mode, loss_scalar=loss_scalar, retain_graph=True)

    # Gradient penalty
    d_loss_gradient_penalty= torch.Tensor([0.0])
    if enable_gradient_penalty:
        d_loss_gradient_penalty = calculate_gradient_penalty(real_images, fake_images, discriminator, lambda_term) * loss_scalar
        d_loss_gradient_penalty.backward()

    return {'d_pred_real': d_pred_real.item(), 'd_pred_fake': d_pred_fake.item(), 'd_loss_gradient_penalty': d_loss_gradient_penalty.item()}