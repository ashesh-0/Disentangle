# Adapted from https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py#L296
import torch
from torch import autograd
from torch.autograd import Variable


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

def increase_value(x, retain_graph=False):
    x.backward(gradient=-1*(torch.ones_like(x)), retain_graph=retain_graph)

def decrease_value(x, retain_graph=False):
    x.backward(gradient=torch.ones_like(x), retain_graph=retain_graph)

def update_gradients_with_generator_loss(discriminator, fake_images):
    d_pred_fake = discriminator(fake_images)
    d_pred_fake = d_pred_fake.mean()
    increase_value(d_pred_fake)
    return {'g_loss':d_pred_fake.item()}

def update_gradients_with_discriminator_loss(discriminator, real_images, fake_images, lambda_term, enable_gradient_penalty=True):
    d_pred_real = discriminator(real_images)
    d_pred_real = d_pred_real.mean()
    increase_value(d_pred_real, retain_graph=True)

    d_pred_fake = discriminator(fake_images)
    d_pred_fake = d_pred_fake.mean()
    decrease_value(d_pred_fake, retain_graph=True)

    # Gradient penalty
    d_loss_gradient_penalty= torch.Tensor([0.0])
    if enable_gradient_penalty:
        d_loss_gradient_penalty = calculate_gradient_penalty(real_images, fake_images, discriminator, lambda_term)
        d_loss_gradient_penalty.backward()

    return {'d_pred_real': d_pred_real.item(), 'd_pred_fake': d_pred_fake.item(), 'd_loss_gradient_penalty': d_loss_gradient_penalty.item()}