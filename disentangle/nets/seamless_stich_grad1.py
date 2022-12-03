from disentangle.nets.seamless_stich import SeamlessStitch
import numpy as np
import torch


class SeamlessStitchGrad1(SeamlessStitch):
    """
    here, we simply return the derivative
            Top
        ------------
        |
    Left|
        |
        |
        ------------
            Bottom
    """

    # computing loss now.

    def _compute_loss_on_boundaries(self, boundary1, boundary2, boundary1_offset):
        ch0_loss = self.loss_metric(boundary1[0] + boundary1_offset, boundary2[0])
        ch1_loss = self.loss_metric(boundary1[1] - boundary1_offset, boundary2[1])
        return (ch0_loss + ch1_loss) / 2

    def _compute_left_loss(self, row_idx, col_idx):
        if col_idx == 0:
            return 0.0
        p = self.params[row_idx, col_idx]
        nbr_p = self.params[row_idx, col_idx - 1].item()

        left_p_gradient = self.get_lgradient(row_idx, col_idx)
        right_p_gradient = self.get_rgradient(row_idx, col_idx - 1)
        avg_gradient = (left_p_gradient + right_p_gradient) / 2
        boundary_gradient = self.get_lneighbor_gradient(row_idx, col_idx)
        return self._compute_loss_on_boundaries(boundary_gradient, avg_gradient, p - nbr_p)

    def _compute_right_loss(self, row_idx, col_idx):
        if col_idx == self.params.shape[1] - 1:
            return 0.0
        p = self.params[row_idx, col_idx]
        nbr_p = self.params[row_idx, col_idx + 1].item()

        left_p_gradient = self.get_lgradient(row_idx, col_idx + 1)
        right_p_gradient = self.get_rgradient(row_idx, col_idx)
        avg_gradient = (left_p_gradient + right_p_gradient) / 2
        boundary_gradient = self.get_rneighbor_gradient(row_idx, col_idx)
        return self._compute_loss_on_boundaries(boundary_gradient, avg_gradient, nbr_p - p)

    def _compute_top_loss(self, row_idx, col_idx):
        if row_idx == 0:
            return 0.0
        p = self.params[row_idx, col_idx]
        nbr_p = self.params[row_idx - 1, col_idx].item()

        top_p_gradient = self.get_tgradient(row_idx, col_idx)
        bottom_p_gradient = self.get_bgradient(row_idx - 1, col_idx)
        avg_gradient = (top_p_gradient + bottom_p_gradient) / 2
        boundary_gradient = self.get_tneighbor_gradient(row_idx, col_idx)
        return self._compute_loss_on_boundaries(boundary_gradient, avg_gradient, p - nbr_p)

    def _compute_bottom_loss(self, row_idx, col_idx):
        if row_idx == self.params.shape[1] - 1:
            return 0.0
        p = self.params[row_idx, col_idx]
        nbr_p = self.params[row_idx + 1, col_idx].item()

        top_p_gradient = self.get_tgradient(row_idx + 1, col_idx)
        bottom_p_gradient = self.get_bgradient(row_idx, col_idx)
        avg_gradient = (top_p_gradient + bottom_p_gradient) / 2
        boundary_gradient = self.get_bneighbor_gradient(row_idx, col_idx)
        return self._compute_loss_on_boundaries(boundary_gradient, avg_gradient, nbr_p - p)
