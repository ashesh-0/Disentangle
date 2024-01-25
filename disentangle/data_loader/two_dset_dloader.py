import numpy as np
import torch

import ml_collections
from disentangle.core.data_split_type import DataSplitType
from disentangle.core.loss_type import LossType
from disentangle.data_loader.base_data_loader import BaseDataLoader
from disentangle.data_loader.lc_multich_dloader import LCMultiChDloader
from disentangle.data_loader.patch_index_manager import GridAlignement, GridIndexManager
from disentangle.data_loader.vanilla_dloader import MultiChDloader


class TwoDsetDloader(BaseDataLoader):
    """
    Here, we can have multiple datasets. We want to get the data from these datasets.
    One can get the data in a probablistic manner from these datasets.
    """

    def __init__(
        self,
        data_config,
        *dsets,
        use_one_mu_std=None,
    ) -> None:

        # self._enable_random_cropping = enable_random_cropping
        self._dsets = dsets
        self._use_one_mu_std = use_one_mu_std

        self._mean = None
        self._std = None
        # assert normalized_input is True, "We are doing the normalization in this dataloader.So you better pass it as True"
        # use_LC = 'multiscale_lowres_count' in data_config and data_config.multiscale_lowres_count is not None
        # data_class = LCMultiChDloader if use_LC else MultiChDloader

        # kwargs = {
        #     'normalized_input': normalized_input,
        #     'enable_rotation_aug': enable_rotation_aug,
        #     'use_one_mu_std': use_one_mu_std,
        #     'allow_generation': allow_generation,
        #     'datasplit_type': datasplit_type,
        #     'grid_alignment': grid_alignment,
        #     'overlapping_padding_kwargs': overlapping_padding_kwargs,
        # }
        # if use_LC:
        #     padding_kwargs = {'mode': data_config.padding_mode}
        #     if 'padding_value' in data_config and data_config.padding_value is not None:
        #         padding_kwargs['constant_values'] = data_config.padding_value
        #     kwargs['padding_kwargs'] = padding_kwargs
        #     kwargs['num_scales'] = data_config.multiscale_lowres_count
        # self._subdset_types = data_config.subdset_types
        # empty_patch_replacement_enabled = data_config.empty_patch_replacement_enabled_list

        self._subdset_types_prob = data_config.subdset_types_probab
        assert sum(self._subdset_types_prob) == 1
        print(f'[{self.__class__.__name__}] Probabs:{self._subdset_types_prob}')

    # def sum_channels(self, data, first_index_arr, second_index_arr):
    #     fst_channel = data[..., first_index_arr].sum(axis=-1, keepdims=True)
    #     scnd_channel = data[..., second_index_arr].sum(axis=-1, keepdims=True)
    #     return np.concatenate([fst_channel, scnd_channel], axis=-1)

    # def set_img_sz(self, image_size, grid_size, alignment=None):
    #     """
    #     Needed just for the notebooks
    #     If one wants to change the image size on the go, then this can be used.
    #     Args:
    #         image_size: size of one patch
    #         grid_size: frame is divided into square grids of this size. A patch centered on a grid having size `image_size` is returned.
    #     """
    #     self._img_sz = image_size
    #     self._grid_sz = grid_size

    #     if self._dset0 is not None:
    #         self._dset0.set_img_sz(image_size, grid_size, alignment=alignment)

    #     if self._dset1 is not None:
    #         self._dset1.set_img_sz(image_size, grid_size, alignment=alignment)

    #     self.idx_manager = GridIndexManager(self.get_data_shape(), self._grid_sz, self._img_sz, alignment)

    def get_mean_std(self):
        """
        Needed just for running the notebooks
        """
        return self._mean, self._std

    # def get_data_shape(self):
    #     N = 0
    #     default_shape = None

    #     if self._dset0 is not None:
    #         default_shape = self._dset0.get_data_shape()
    #         N += default_shape[0]

    #     if self._dset1 is not None:
    #         default_shape = self._dset1.get_data_shape()
    #         N += default_shape[0]

    #     default_shape = list(default_shape)
    #     default_shape[0] = N
    #     return tuple(default_shape)

    def __len__(self):
        sz = 0
        for i, dset in enumerate(self._dsets):
            if dset is not None:
                sz += int(self._subdset_types_prob[i] * len(dset))

        return sz

    def compute_mean_std_for_input(self, dloader):
        mean_for_input, std_for_input = dloader.compute_mean_std()
        mean_for_input = mean_for_input.squeeze()
        assert mean_for_input[0] == mean_for_input[1]
        mean_for_input = np.array(mean_for_input[0], dtype=np.float32)

        std_for_input = std_for_input.squeeze()
        assert std_for_input[0] == std_for_input[1]
        std_for_input = np.array([std_for_input[0]], dtype=np.float32)
        return mean_for_input, std_for_input

    def compute_individual_mean_std(self):
        mean_dict = {f'subdset_{i}': {} for i in range(len(self._dsets))}
        std_dict = {f'subdset_{i}': {} for i in range(len(self._dsets))}

        for i, dset in enumerate(self._dsets):
            if dset is not None:
                mean_, std_ = dset.compute_individual_mean_std()
                mean_for_input, std_for_input = self.compute_mean_std_for_input(dset)
                mean_dict[f'subdset_{i}'] = {'target': mean_, 'input': mean_for_input}
                std_dict[f'subdset_{i}'] = {'target': std_, 'input': std_for_input}

        # assert LossType.ElboMixedReconstruction in [self.get_loss_idx(0), self.get_loss_idx(1)]
        # if self.get_loss_idx(0) == LossType.ElboMixedReconstruction:
        #     # we are doing this for the model, not for the validation dadtaloader.
        #     mean_dict['subdset_0']['target'] = mean_dict['subdset_1']['target']
        #     mean_dict['subdset_0']['input'] = mean_dict['subdset_1']['input']
        # else:
        #     mean_dict['subdset_1']['target'] = mean_dict['subdset_0']['target']
        #     mean_dict['subdset_1']['input'] = mean_dict['subdset_0']['input']

        return mean_dict, std_dict

    # def _compute_mean_std(self, allow_for_validation_data=False):
    #     mean_dict = {'subdset_0': {}, 'subdset_1': {}}
    #     std_dict = {'subdset_0': {}, 'subdset_1': {}}

    #     if self._dset0 is not None:
    #         mean_, std_ = self._dset0.compute_mean_std(allow_for_validation_data=allow_for_validation_data)
    #         mean_dict['subdset_0'] = {'target': mean_}
    #         std_dict['subdset_0'] = {'target': std_}

    #     if self._dset1 is not None:
    #         mean_, std_ = self._dset1.compute_mean_std(allow_for_validation_data=allow_for_validation_data)
    #         mean_dict['subdset_1'] = {'target': mean_}
    #         std_dict['subdset_1'] = {'target': std_}
    #     return mean_dict, std_dict

    def compute_mean_std(self, allow_for_validation_data=False):
        assert self._use_one_mu_std is True, "We are not supporting separate mean and std for creating the input."
        return self.compute_individual_mean_std()

    def set_mean_std(self, mean_val, std_val):
        # NOTE:
        self._mean = mean_val
        self._std = std_val

    def per_side_overlap_pixelcount(self):
        for i, dset in enumerate(self._dsets):
            if dset is not None:
                return dset.per_side_overlap_pixelcount()

    def get_idx_manager(self):
        active_dsets = [dset is not None for dset in self._dsets]
        assert sum([int(x) for x in active_dsets]) == 1
        for dset in self._dsets:
            if dset is not None:
                return dset.get_idx_manager()

    def get_grid_size(self):
        active_dsets = [dset is not None for dset in self._dsets]
        assert sum([int(x) for x in active_dsets]) == 1
        for dset in self._dsets:
            if dset is not None:
                return dset.get_grid_size()

        # d0_active = self._dset0 is not None
        # d1_active = self._dset1 is not None
        # assert d0_active or d1_active
        # assert not (d0_active and d1_active)
        # if d0_active:
        #     return self._dset0.get_grid_size()
        # else:
        #     return self._dset1.get_grid_size()

    def get_loss_idx(self, dset_idx):
        if dset_idx != len(self._dsets) - 1:
            return LossType.Elbo
        else:
            return LossType.ElboMixedReconstruction

    def get_random_dset_index(self):
        coin_flip = np.random.rand()
        prob_list = np.cumsum(self._subdset_types_prob)
        for idx, cum_prob in enumerate(prob_list):
            if coin_flip <= cum_prob:
                return idx

        raise ValueError(f"This is invalid state.: p={coin_flip}, cdf={prob_list}")

    def __getitem__(self, index):
        """
        Returns:
            (inp,tar,dset_label,loss_idx)
        """
        assert sum(self._subdset_types_prob) == 1
        if 1.0 in self._subdset_types_prob or 1 in self._subdset_types_prob:
            dset_idx = np.argmax(self._subdset_types_prob)
            loss_idx = self.get_loss_idx(dset_idx)
            dset = self._dsets[dset_idx]
            return (*dset[index], dset_idx, loss_idx)

        # if self._subdset_types_prob[0] == 0 or self._subdset_types_prob[1] == 0:
        #     # This is typically only true when we are handling validation.``
        #     if self._subdset_types_prob[0] == 0:
        #         dset_idx = 1
        #         return (*self._dset1[index], dset_idx, self.get_loss_idx(dset_idx))
        #     elif self._subdset_types_prob[1] == 0:
        #         dset_idx = 0
        #         return (*self._dset0[index], dset_idx, self.get_loss_idx(dset_idx))
        #     else:
        #         raise ValueError("This is invalid state.")
        else:
            # prob_list = np.cumsum(self._subdset_types_prob)
            # coin_flip = np.random.rand()
            # if coin_flip <= prob_list[0]:
            #     dset_idx = 0
            # elif coin_flip > prob_list[0] and coin_flip <= prob_list[1]:
            #     dset_idx = 1
            dset_idx = self.get_random_dset_index()
            loss_idx = self.get_loss_idx(dset_idx)
            dset = self._dsets[dset_idx]
            idx = np.random.randint(len(dset))
            return (*dset[idx], dset_idx, loss_idx)

    def get_max_val(self):
        max_vals = []
        for dset in self._dsets:
            if dset is not None:
                max_vals.append(dset.get_max_val())
            else:
                max_vals.append(None)
        return max_vals
