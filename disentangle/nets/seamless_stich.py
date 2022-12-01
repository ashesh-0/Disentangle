"""
Do seamless stitching
"""
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.optim as optim
from disentangle.core.seamless_stitch_base import SeamlessStitchBase


class Model(nn.Module):
    def __init__(self, N):
        super().__init__()
        self._N = N
        self.params = nn.Parameter(torch.zeros(self._N, self._N))
        self.shape = self.params.shape

    def __getitem__(self, pos):
        i, j = pos
        return self.params[i, j]


class SeamlessStitch(SeamlessStitchBase):
    def __init__(self, grid_size, stitched_frame, learning_rate):
        super().__init__(grid_size, stitched_frame)
        self.params = Model(self._N)
        self.opt = torch.optim.SGD(self.params.parameters(), lr=learning_rate)
        self.loss_metric = nn.L1Loss(reduction='sum')

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt,
                                                                 'min',
                                                                 patience=10,
                                                                 factor=0.5,
                                                                 threshold_mode='abs',
                                                                 min_lr=1e-12,
                                                                 verbose=True)

    def get_ch0_offset(self, row_idx, col_idx):
        return self.params[row_idx, col_idx].item()

    def _compute_loss_on_boundaries(self, boundary1, boundary2, boundary1_param):
        ch0_loss = self.loss_metric(boundary1[0] + boundary1_param, boundary2[0])
        ch1_loss = self.loss_metric(boundary1[1] - boundary1_param, boundary2[1])
        return (ch0_loss + ch1_loss) / 2

    def _compute_left_loss(self, row_idx, col_idx):
        if col_idx == 0:
            return 0.0
        p = self.params[row_idx, col_idx]

        left_p_boundary = self.get_lboundary(row_idx, col_idx)
        right_p_boundary = self.get_rboundary(row_idx, col_idx - 1)
        return self._compute_loss_on_boundaries(left_p_boundary, right_p_boundary, p)

    def _compute_right_loss(self, row_idx, col_idx):
        if col_idx == self.params.shape[1] - 1:
            return 0.0
        p = self.params[row_idx, col_idx]

        left_p_boundary = self.get_lboundary(row_idx, col_idx + 1)
        right_p_boundary = self.get_rboundary(row_idx, col_idx)
        return self._compute_loss_on_boundaries(right_p_boundary, left_p_boundary, p)

    def _compute_top_loss(self, row_idx, col_idx):
        if row_idx == 0:
            return 0.0
        p = self.params[row_idx, col_idx]

        top_p_boundary = self.get_tboundary(row_idx, col_idx)
        bottom_p_boundary = self.get_bboundary(row_idx - 1, col_idx)
        return self._compute_loss_on_boundaries(top_p_boundary, bottom_p_boundary, p)

    def _compute_bottom_loss(self, row_idx, col_idx):
        if row_idx == self.params.shape[1] - 1:
            return 0.0
        p = self.params[row_idx, col_idx]

        top_p_boundary = self.get_tboundary(row_idx + 1, col_idx)
        bottom_p_boundary = self.get_bboundary(row_idx, col_idx)
        return self._compute_loss_on_boundaries(bottom_p_boundary, top_p_boundary, p)

    def _compute_loss(self, row_idx, col_idx):
        left_loss = self._compute_left_loss(row_idx, col_idx)
        right_loss = self._compute_right_loss(row_idx, col_idx)

        top_loss = self._compute_top_loss(row_idx, col_idx)
        bottom_loss = self._compute_bottom_loss(row_idx, col_idx)
        return left_loss + right_loss + top_loss + bottom_loss

    def compute_loss(self):
        loss = 0.0
        for row_idx in range(self._N):
            for col_idx in range(self._N):
                loss += self._compute_loss(row_idx, col_idx) / (2 * ((self._N - 1)**2))
        return loss

    def fit(self, steps=100):
        loss_arr = []
        steps_iter = tqdm(range(steps))
        for _ in steps_iter:
            self.params.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            self.opt.step()

            loss_arr.append(loss.item())
            steps_iter.set_description(f'Loss: {loss_arr[-1]:.3f}')
            self.lr_scheduler.step(loss)
        return loss_arr
