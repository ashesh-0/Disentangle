from turtle import right
import torch
import torch.nn as nn


class NeighborConsistencyLoss:
    def __init__(self) -> None:
        self.loss_metric = nn.MSELoss()

    def on_boundary_lgrad(self, imgs):
        nD = len(imgs.shape)
        return torch.diff(imgs[..., :, :2], dim=nD - 1)

    def on_boundary_rgrad(self, imgs):
        nD = len(imgs.shape)
        return torch.diff(imgs[..., :, -2:], dim=nD - 1)

    def on_boundary_ugrad(self, imgs):
        nD = len(imgs.shape)
        return torch.diff(imgs[..., :2, :], dim=nD - 2)

    def on_boundary_dgrad(self, imgs):
        nD = len(imgs.shape)
        return torch.diff(imgs[..., -2:, :], dim=nD - 2)

    def across_boundary_horizontal_grad(self, left_img, right_img):
        return right_img[..., :, :1] - left_img[..., :, -1:]

    def across_boundary_vertical_grad(self, top_img, bottom_img):
        return bottom_img[..., :1, :] - top_img[..., -1:, :]

    def get_left_loss(self, imgs):
        # center-left
        ref_lgrad = self.on_boundary_lgrad(imgs[0])
        left_rgrad = self.on_boundary_rgrad(imgs[1])
        across_horizontal_grad = self.across_boundary_horizontal_grad(imgs[1], imgs[0])
        return self.loss_metric(across_horizontal_grad, (left_rgrad + ref_lgrad) / 2)

    def get_right_loss(self, imgs):
        ref_rgrad = self.on_boundary_rgrad(imgs[0])
        left_lgrad = self.on_boundary_lgrad(imgs[2])
        across_horizontal_grad = self.across_boundary_horizontal_grad(imgs[0], imgs[2])
        return self.loss_metric(across_horizontal_grad, (left_lgrad + ref_rgrad) / 2)

    def get_top_loss(self, imgs):
        ref_ugrad = self.on_boundary_ugrad(imgs[0])
        up_dgrad = self.on_boundary_dgrad(imgs[3])
        across_vertical_grad = self.across_boundary_vertical_grad(imgs[3], imgs[0])
        return self.loss_metric(across_vertical_grad, (up_dgrad + ref_ugrad) / 2)

    def get_bottom_loss(self, imgs):
        ref_dgrad = self.on_boundary_dgrad(imgs[0])
        down_ugrad = self.on_boundary_ugrad(imgs[4])
        across_vertical_grad = self.across_boundary_vertical_grad(imgs[0], imgs[4])
        return self.loss_metric(across_vertical_grad, (ref_dgrad + down_ugrad) / 2)

    def get(self, imgs):
        relevant_imgs = 5 * (len(imgs) // 5)
        imgs = imgs[:relevant_imgs]
        imgs = imgs.view(5, relevant_imgs // 5, *imgs.shape[1:])
        loss = 0
        loss += self.get_left_loss(imgs)
        loss += self.get_right_loss(imgs)
        loss += self.get_top_loss(imgs)
        loss += self.get_bottom_loss(imgs)
        return loss / 4


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    loss = NeighborConsistencyLoss()
    center = torch.Tensor(np.arange(32)[None, None, None]).repeat(1, 2, 32, 1)
    left = torch.Tensor(np.arange(-32, 0)[None, None, None]).repeat(1, 2, 32, 1)
    right = torch.Tensor(np.arange(40, 72)[None, None, None]).repeat(1, 2, 32, 1)
    top = torch.Tensor(np.arange(32)[None, None, :, None]).repeat(1, 2, 1, 32)
    bottom = torch.Tensor(np.arange(32)[None, None, None]).repeat(1, 2, 32, 1)

    imgs = torch.cat([center, left, right, top, bottom], dim=0)
    _, ax = plt.subplots(figsize=(9, 9), ncols=3, nrows=3)
    ax[0, 1].imshow(top[0, 0], vmin=-32, vmax=72)
    ax[1, 1].imshow(center[0, 0], vmin=-32, vmax=72)
    ax[1, 0].imshow(left[0, 0], vmin=-32, vmax=72)
    ax[1, 2].imshow(right[0, 0], vmin=-32, vmax=72)
    ax[2, 1].imshow(bottom[0, 0], vmin=-32, vmax=72)

    out = loss.get(imgs)