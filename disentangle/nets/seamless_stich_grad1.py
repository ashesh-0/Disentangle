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
        nbr_p = self.params[row_idx, col_idx - 1]

        left_p_gradient = self.get_lgradient(row_idx, col_idx)
        right_p_gradient = self.get_rgradient(row_idx, col_idx - 1)
        avg_gradient = (left_p_gradient + right_p_gradient) / 2
        boundary_gradient = self.get_lneighbor_gradient(row_idx, col_idx)
        return self._compute_loss_on_boundaries(boundary_gradient, avg_gradient, p - nbr_p)

    def _compute_right_loss(self, row_idx, col_idx):
        if col_idx == self.params.shape[1] - 1:
            return 0.0
        p = self.params[row_idx, col_idx]
        nbr_p = self.params[row_idx, col_idx + 1]

        left_p_gradient = self.get_lgradient(row_idx, col_idx + 1)
        right_p_gradient = self.get_rgradient(row_idx, col_idx)
        avg_gradient = (left_p_gradient + right_p_gradient) / 2
        boundary_gradient = self.get_rneighbor_gradient(row_idx, col_idx)
        return self._compute_loss_on_boundaries(boundary_gradient, avg_gradient, nbr_p - p)

    def _compute_top_loss(self, row_idx, col_idx):
        if row_idx == 0:
            return 0.0
        p = self.params[row_idx, col_idx]
        nbr_p = self.params[row_idx - 1, col_idx]

        top_p_gradient = self.get_tgradient(row_idx, col_idx)
        bottom_p_gradient = self.get_bgradient(row_idx - 1, col_idx)
        avg_gradient = (top_p_gradient + bottom_p_gradient) / 2
        boundary_gradient = self.get_tneighbor_gradient(row_idx, col_idx)
        return self._compute_loss_on_boundaries(boundary_gradient, avg_gradient, p - nbr_p)

    def _compute_bottom_loss(self, row_idx, col_idx):
        if row_idx == self.params.shape[1] - 1:
            return 0.0
        p = self.params[row_idx, col_idx]
        nbr_p = self.params[row_idx + 1, col_idx]

        top_p_gradient = self.get_tgradient(row_idx + 1, col_idx)
        bottom_p_gradient = self.get_bgradient(row_idx, col_idx)
        avg_gradient = (top_p_gradient + bottom_p_gradient) / 2
        boundary_gradient = self.get_bneighbor_gradient(row_idx, col_idx)
        return self._compute_loss_on_boundaries(boundary_gradient, avg_gradient, nbr_p - p)


if __name__ == '__main__':

    from disentangle.core.tiff_reader import load_tiff
    import numpy as np
    import torch

    pref = '2211-D3M3S0-31_P64_G64_M1_Sk32'
    data0 = load_tiff(f'paper_tifs/{pref}_C0.tif')
    data1 = load_tiff(f'paper_tifs/{pref}_C1.tif')

    pred0 = data0[0]
    tar0 = data0[1]

    pred1 = data1[0]
    tar1 = data1[1]

    pred_np = np.concatenate([pred0[None], pred1[None]], axis=0)
    tar_np = np.concatenate([tar0[None], tar1[None]], axis=0)
    pred = torch.Tensor(pred_np).cuda()

    grid_size = 64
    learning_rate = 200
    lr_patience = 5
    # 4347.534
    # model = SeamlessStitch(grid_size, pred, learning_rate)
    model = SeamlessStitchGrad1(grid_size, pred, learning_rate, lr_patience=lr_patience)