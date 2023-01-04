from turtle import right
import torch
import torch.nn as nn
import numpy as np


class NeighborConsistencyLoss:
    def __init__(self, grid_size, nbr_set_count=None) -> None:
        self.loss_metric = nn.MSELoss()
        self._default_grid_size = grid_size
        self._nbr_set_count = nbr_set_count
        print(f'[{self.__class__.__name__}] DefGrid:{self._default_grid_size} NbrSet:{self._nbr_set_count}')

    def use_default_grid(self, grid_size):
        return grid_size is None or grid_size < 0

    def on_boundary_lgrad(self, imgs, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        nD = len(imgs.shape)
        assert imgs.shape[-1] == imgs.shape[-2]
        pad = (imgs.shape[-1] - grid_size) // 2
        return torch.diff(imgs[..., pad:-pad, pad:pad + 2], dim=nD - 1)

    def on_boundary_rgrad(self, imgs, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        nD = len(imgs.shape)
        assert imgs.shape[-1] == imgs.shape[-2]
        pad = (imgs.shape[-1] - grid_size) // 2

        return torch.diff(imgs[..., pad:-pad, -(pad + 2):-pad], dim=nD - 1)

    def on_boundary_ugrad(self, imgs, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        nD = len(imgs.shape)
        assert imgs.shape[-1] == imgs.shape[-2]
        pad = (imgs.shape[-1] - grid_size) // 2

        return torch.diff(imgs[..., pad:pad + 2, pad:-pad], dim=nD - 2)

    def on_boundary_dgrad(self, imgs, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        nD = len(imgs.shape)
        assert imgs.shape[-1] == imgs.shape[-2]
        pad = (imgs.shape[-1] - grid_size) // 2
        return torch.diff(imgs[..., -(pad + 2):-pad, pad:-pad], dim=nD - 2)

    def across_boundary_horizontal_grad(self, left_img, right_img, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        pad = (left_img.shape[-1] - grid_size) // 2
        return right_img[..., pad:-pad, pad:pad + 1] - left_img[..., pad:-pad, -(pad + 1):-pad]

    def across_boundary_vertical_grad(self, top_img, bottom_img, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        pad = (top_img.shape[-1] - grid_size) // 2
        return bottom_img[..., pad:(pad + 1), pad:-pad] - top_img[..., -(pad + 1):-pad, pad:-pad]

    def get_left_loss(self, imgs, grid_size=None):
        # center-left
        ref_lgrad = self.on_boundary_lgrad(imgs[0], grid_size=grid_size)
        left_rgrad = self.on_boundary_rgrad(imgs[1], grid_size=grid_size)
        across_horizontal_grad = self.across_boundary_horizontal_grad(imgs[1], imgs[0], grid_size=grid_size)
        return self.loss_metric(across_horizontal_grad, (left_rgrad + ref_lgrad) / 2)

    def get_right_loss(self, imgs, grid_size=None):
        ref_rgrad = self.on_boundary_rgrad(imgs[0], grid_size=grid_size)
        left_lgrad = self.on_boundary_lgrad(imgs[2], grid_size=grid_size)
        across_horizontal_grad = self.across_boundary_horizontal_grad(imgs[0], imgs[2], grid_size=grid_size)
        return self.loss_metric(across_horizontal_grad, (left_lgrad + ref_rgrad) / 2)

    def get_top_loss(self, imgs, grid_size=None):
        ref_ugrad = self.on_boundary_ugrad(imgs[0], grid_size=grid_size)
        up_dgrad = self.on_boundary_dgrad(imgs[3], grid_size=grid_size)
        across_vertical_grad = self.across_boundary_vertical_grad(imgs[3], imgs[0], grid_size=grid_size)
        return self.loss_metric(across_vertical_grad, (up_dgrad + ref_ugrad) / 2)

    def get_bottom_loss(self, imgs, grid_size=None):
        ref_dgrad = self.on_boundary_dgrad(imgs[0], grid_size=grid_size)
        down_ugrad = self.on_boundary_ugrad(imgs[4], grid_size=grid_size)
        across_vertical_grad = self.across_boundary_vertical_grad(imgs[0], imgs[4], grid_size=grid_size)
        return self.loss_metric(across_vertical_grad, (ref_dgrad + down_ugrad) / 2)

    def get(self, imgs, grid_sizes=None):
        if grid_sizes is not None:
            grid_sizes = grid_sizes.detach().cpu().numpy()
        else:
            grid_sizes = np.ones(len(imgs)) * self._default_grid_size

        relevant_imgs = 5 * (len(imgs) // 5)
        if self._nbr_set_count is not None:
            relevant_imgs = min(relevant_imgs, 5 * self._nbr_set_count)

        imgs = imgs[:relevant_imgs]
        if len(imgs) == 0:
            return None

        imgs = imgs.view(5, relevant_imgs // 5, *imgs.shape[1:])
        loss = 0
        for idx in range(0, relevant_imgs // 5):
            grid_size = np.unique(grid_sizes[5 * idx:5 * idx + 5])
            assert len(grid_size) == 1
            grid_size = grid_size[0]
            loss += self.get_left_loss(imgs[:, idx:idx + 1], grid_size=grid_size)
            loss += self.get_right_loss(imgs[:, idx:idx + 1], grid_size=grid_size)
            loss += self.get_top_loss(imgs[:, idx:idx + 1], grid_size=grid_size)
            loss += self.get_bottom_loss(imgs[:, idx:idx + 1], grid_size=grid_size)

        return loss / (4 * relevant_imgs / 5)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    grid_size = 20
    loss = NeighborConsistencyLoss(grid_size)
    center = torch.Tensor(np.arange(grid_size)[None, None, None]).repeat(1, 2, grid_size, 1)
    left = torch.Tensor(np.arange(-grid_size - 10, -10)[None, None, None]).repeat(1, 2, grid_size, 1)
    right = torch.Tensor(np.arange(grid_size, 2 * grid_size)[None, None, None]).repeat(1, 2, grid_size, 1)
    bottom = torch.Tensor(np.arange(grid_size)[None, None, :, None]).repeat(1, 2, 1, grid_size)
    top = torch.Tensor(np.arange(grid_size)[None, None, None]).repeat(1, 2, grid_size, 1)
    _, ax = plt.subplots(figsize=(9, 9), ncols=3, nrows=3)
    ax[0, 1].imshow(top[0, 0], vmin=-20, vmax=49)
    ax[1, 1].imshow(center[0, 0], vmin=-20, vmax=49)
    ax[1, 0].imshow(left[0, 0], vmin=-20, vmax=49)
    ax[1, 2].imshow(right[0, 0], vmin=-20, vmax=49)
    ax[2, 1].imshow(bottom[0, 0], vmin=-20, vmax=49)

    center = torch.Tensor(np.pad(center, ((0, 0), (0, 0), (6, 6), (6, 6)), mode='linear_ramp'))
    left = torch.Tensor(np.pad(left, ((0, 0), (0, 0), (6, 6), (6, 6)), mode='linear_ramp'))
    right = torch.Tensor(np.pad(right, ((0, 0), (0, 0), (6, 6), (6, 6)), mode='linear_ramp'))
    bottom = torch.Tensor(np.pad(bottom, ((0, 0), (0, 0), (6, 6), (6, 6)), mode='linear_ramp'))
    top = torch.Tensor(np.pad(top, ((0, 0), (0, 0), (6, 6), (6, 6)), mode='linear_ramp'))

    imgs = torch.cat([center, left, right, top, bottom], dim=0)
    _, ax = plt.subplots(figsize=(9, 9), ncols=3, nrows=3)
    ax[0, 1].imshow(top[0, 0], vmin=-20, vmax=49)
    ax[1, 1].imshow(center[0, 0], vmin=-20, vmax=49)
    ax[1, 0].imshow(left[0, 0], vmin=-20, vmax=49)
    ax[1, 2].imshow(right[0, 0], vmin=-20, vmax=49)
    ax[2, 1].imshow(bottom[0, 0], vmin=-20, vmax=49)
    grid_sizes = torch.Tensor(np.repeat([16, 18, 20, 22], repeats=5)).type(torch.int32)
    out = loss.get(imgs, grid_sizes=grid_sizes)
