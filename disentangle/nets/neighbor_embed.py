"""
Neighboring embedding manager.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from disentangle.core.merge_layer import MergeLayer


class NeighborEmbedManager(nn.Module):
    """
    Tasks which this class performs:
        1. Merge the embedding of neighboring prediction with the primary flow embeddings.

    Assumption: the embedding of neighboring predictions have the same shape as primary flow embeddings would have when there are no neighboring predictions. Sp, while the embeddings of neighboring predictions are expected to be (B,C, 64, 64), (B,C,32,32), (B,C,16,16)... the primary flow embeddings are expected to be (B,C, 64, 64), (B,C,34,34), (B,C, 18, 18)... 
    This assumption is used to decide where to put the boundary information on top of the primary flow. 
    """

    def __init__(self,
                 latent_spatialsize: int,
                 enable_seep_merge: bool,
                 learnable_mask: bool,
                 n_filters=None,
                 merge_type=None,
                 nonlin=None,
                 batchnorm=None,
                 dropout=None,
                 res_block_type=None,
                 res_block_kernel=None):
        super().__init__()
        self._enable_seep_merge = enable_seep_merge
        self._learnable_mask = learnable_mask
        nbr_count = 4
        if self._enable_seep_merge:
            self._merge_layer = MergeLayer(
                channels=[n_filters] * (2 + 1),
                merge_type=merge_type,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                res_block_kernel=res_block_kernel,
            )
        else:
            self._merge_layer = MergeLayer(
                channels=[n_filters] * (nbr_count + 2),
                merge_type=merge_type,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                res_block_kernel=res_block_kernel,
            )

            if self._learnable_mask:
                self._weight_mask = nn.Linear(1, latent_spatialsize, bias=False)
                self._dummy_inp = torch.ones((1, 1, 1), requires_grad=False)

    def get_learned_mask(self):
        return nn.Sigmoid()(self._weight_mask(self._dummy_inp))

    def get_mask(self, spatial_dim, orientation: str, device):
        # disable mask when seep merge is enabled
        if self._enable_seep_merge:
            return 1

        if self._learnable_mask:
            idx = self._dim_to_idx[spatial_dim]
            if self._dummy_inp.device != device:
                self._dummy_inp = self._dummy_inp.to(device)

            mask = self.get_learned_mask(idx)
        else:
            mask = torch.arange(0, spatial_dim) / (spatial_dim - 1)

        if orientation == 'top':
            mask = mask.reshape(1, 1, -1, 1).to(device)
        elif orientation == 'bottom':
            mask = torch.flip(mask, [0])
            mask = mask.reshape(1, 1, -1, 1).to(device)
        elif orientation == 'left':
            mask = mask.reshape(1, 1, 1, -1).to(device)
        elif orientation == 'right':
            mask = torch.flip(mask, [0])
            mask = mask.reshape(1, 1, 1, -1).to(device)
        else:
            raise ValueError(f"orientation {orientation} not recognized")

        if not self._learnable_mask:
            mask[mask < 0.95] = 0
        return mask

    def _process_nbr_bu_value(self, side: str, nbr_bu_value: List[torch.Tensor], pix_n: int = 1):
        """
        Tasks:
            1. Only keep the boundary region of the neighboring predictions.
            2. Optionally apply a mask to the boundary region.
        """
        shape = nbr_bu_value.shape[-2:]
        assert shape[0] == shape[1]

        if side == 'top':
            nbr_bu_value = nbr_bu_value[:, :, -pix_n:]
        elif side == 'bottom':
            nbr_bu_value = nbr_bu_value[:, :, :pix_n]
        elif side == 'left':
            nbr_bu_value = nbr_bu_value[:, :, :, -pix_n:]
        elif side == 'right':
            nbr_bu_value = nbr_bu_value[:, :, :, :pix_n]

        nbr_bu_value = nbr_bu_value * self.get_mask(shape[0], side, nbr_bu_value.device)
        return nbr_bu_value

    def merge_bu_value(self, bu_value, nbr_bu_value):
        merged_bu_value = None
        nbr_shape = nbr_bu_value[0].shape
        pix_n = 2
        final_shape = (*nbr_shape[:-2], nbr_shape[-2] + 2 * pix_n, nbr_shape[-1] + 2 * pix_n)

        nbr_bu_value = [
            self._process_nbr_bu_value(*x, pix_n=pix_n) for x in zip(['top', 'bottom', 'left', 'right'], nbr_bu_value)
        ]
        if self._enable_seep_merge:
            topb, bottomb, leftb, rightb = nbr_bu_value
            shape = bu_value.shape

            assert topb.shape[-2:] == bottomb.shape[-2:], 'top and bottom embedding should be symmetric'
            assert leftb.shape[-2:] == rightb.shape[-2:], 'left and right embedding should be symmetric'
            boundary = torch.zeros(final_shape).to(bu_value.device)
            dcols = final_shape[-1] - nbr_shape[-1]
            drows = final_shape[-2] - nbr_shape[-2]
            boundary[..., :pix_n, dcols // 2:-1 * dcols // 2] = topb
            boundary[..., -1 * pix_n:, dcols // 2:-dcols // 2] = bottomb
            boundary[..., drows // 2:-drows // 2, :1 * pix_n] = leftb
            boundary[..., drows // 2:-drows // 2, -1 * pix_n:] = rightb

            padn_col = (final_shape[-1] - shape[-1]) // 2
            padn_row = (final_shape[-2] - shape[-2]) // 2

            bu_value = F.pad(bu_value, (padn_col, padn_col, padn_row, padn_row), mode='constant', value=0)
            assert bu_value.shape == boundary.shape, f'{bu_value.shape} != {boundary.shape}'
            bu_value = self._merge_layer(bu_value, boundary)
            merged_bu_value = bu_value
        else:
            merged_bu_value = self._merge_layer(bu_value, *nbr_bu_value)
        return merged_bu_value


if __name__ == '__main__':
    latent_spatialsize = 8
    enable_seep_merge = True
    learnable_mask = True
    n_filters = 32
    merge_type = 'residual'
    nonlin = nn.ReLU
    batchnorm = True
    dropout = 0.0
    res_block_type = 'bacdbacd'
    res_block_kernel = 3

    manager = NeighborEmbedManager(latent_spatialsize,
                                   enable_seep_merge,
                                   learnable_mask,
                                   n_filters,
                                   merge_type=merge_type,
                                   nonlin=nonlin,
                                   batchnorm=batchnorm,
                                   dropout=dropout,
                                   res_block_type=res_block_type,
                                   res_block_kernel=res_block_kernel)
    bu_value = 4 * torch.rand((1, n_filters, latent_spatialsize + 2, latent_spatialsize + 2))
    nbr_bu_value = [10 * (i + 1) * torch.ones((1, n_filters, latent_spatialsize, latent_spatialsize)) for i in range(4)]
    merged_bu_value = manager.merge_bu_value(bu_value, nbr_bu_value)
    print(merged_bu_value.shape)

    import seaborn as sns
    sns.heatmap(merged_bu_value[0, 0].detach().cpu().numpy())
