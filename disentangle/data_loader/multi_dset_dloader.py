import numpy as np
import torch

import ml_collections
from disentangle.core.data_split_type import DataSplitType
from disentangle.core.loss_type import LossType
from disentangle.data_loader.ht_iba1_ki67_rawdata_loader import SubDsetType
from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from disentangle.data_loader.multiscale_mc_tiff_dloader import MultiScaleTiffDloader
from disentangle.data_loader.patch_index_manager import GridIndexManager
from disentangle.data_loader.pavia2_enums import Pavia2BleedthroughType


class MultiDsetDloader:

    def __init__(self,
                 data_config,
                 fpath: str,
                 datasplit_type: DataSplitType = None,
                 val_fraction=None,
                 test_fraction=None,
                 normalized_input=None,
                 enable_rotation_aug: bool = False,
                 enable_random_cropping: bool = False,
                 use_one_mu_std=None,
                 allow_generation=False,
                 max_val=None) -> None:

        self._datasplit_type = datasplit_type
        self._enable_random_cropping = enable_random_cropping
        self._dloader_0 = self._dloader_1 = self._dloader_mix = None
        self._use_one_mu_std = use_one_mu_std

        self._mean = None
        self._std = None
        assert normalized_input is True, "We are doing the normalization in this dataloader.So you better pass it as True"
        use_LC = 'multiscale_lowres_count' in data_config and data_config.multiscale_lowres_count is not None
        data_class = MultiScaleTiffDloader if use_LC else MultiChDeterministicTiffDloader

        kwargs = {
            'normalized_input': normalized_input,
            'enable_rotation_aug': enable_rotation_aug,
            'use_one_mu_std': use_one_mu_std,
            'allow_generation': allow_generation,
            'datasplit_type': datasplit_type
        }
        if use_LC:
            padding_kwargs = {'mode': data_config.padding_mode}
            if 'padding_value' in data_config and data_config.padding_value is not None:
                padding_kwargs['constant_values'] = data_config.padding_value
            kwargs['padding_kwargs'] = padding_kwargs
            kwargs['num_scales'] = data_config.multiscale_lowres_count

        self._subdset_types = data_config.subdset_types
        empty_patch_replacement_enabled = data_config.empty_patch_replacement_enabled_list

        if self._datasplit_type == DataSplitType.Train:
            dconf = ml_collections.ConfigDict(data_config)
            self._subdset_types_prob = dconf.subdset_types_probab
            assert sum(self._subdset_types_prob) == 1
            # take channels mean from this.
            dconf.subdset_type = self._subdset_types[0]
            dconf.empty_patch_replacement_enabled = empty_patch_replacement_enabled[0]
            self._dloader_0 = data_class(dconf,
                                         fpath,
                                         val_fraction=val_fraction[0],
                                         test_fraction=test_fraction[0],
                                         enable_random_cropping=True,
                                         max_val=None,
                                         **kwargs)

            dconf.subdset_type = self._subdset_types[1]
            dconf.empty_patch_replacement_enabled = empty_patch_replacement_enabled[1]
            self._dloader_1 = data_class(dconf,
                                         fpath,
                                         val_fraction=val_fraction[1],
                                         test_fraction=test_fraction[1],
                                         enable_random_cropping=True,
                                         max_val=None,
                                         **kwargs)
        else:
            self._dloader_0 = self._dloader_1 = None

            assert enable_random_cropping is False
            dconf = ml_collections.ConfigDict(data_config)
            dconf.subdset_type = self._subdset_types[dconf.validation_subdset_type_idx]
            self._subdset_types_prob = [0] * len(dconf.subdset_types_probab)
            self._subdset_types_prob[dconf.validation_subdset_type_idx] = 1
            max_val = max_val[dconf.validation_subdset_type_idx]
            # we want to evaluate on mixed samples.
            dloader = data_class(dconf,
                                 fpath,
                                 val_fraction=val_fraction[dconf.validation_subdset_type_idx],
                                 test_fraction=test_fraction[dconf.validation_subdset_type_idx],
                                 enable_random_cropping=enable_random_cropping,
                                 max_val=max_val,
                                 **kwargs)
            setattr(self, f'_dloader_{dconf.validation_subdset_type_idx}', dloader)

        self.process_data()

        # needed just during evaluation.
        if self._dloader_0 is not None:
            self._img_sz = self._dloader_0._img_sz
            self._grid_sz = self._dloader_0._grid_sz
        else:
            self._img_sz = self._dloader_1._img_sz
            self._grid_sz = self._dloader_1._grid_sz

        print(f'[{self.__class__.__name__}] Dtypes:{self._subdset_types} Probabs:{self._subdset_types_prob}')

    def sum_channels(self, data, first_index_arr, second_index_arr):
        fst_channel = data[..., first_index_arr].sum(axis=-1, keepdims=True)
        scnd_channel = data[..., second_index_arr].sum(axis=-1, keepdims=True)
        return np.concatenate([fst_channel, scnd_channel], axis=-1)

    def process_data(self):
        """
        """
        pass

    def set_img_sz(self, image_size, grid_size, alignment=None):
        """
        Needed just for the notebooks
        If one wants to change the image size on the go, then this can be used.
        Args:
            image_size: size of one patch
            grid_size: frame is divided into square grids of this size. A patch centered on a grid having size `image_size` is returned.
        """
        self._img_sz = image_size
        self._grid_sz = grid_size

        if self._dloader_0 is not None:
            self._dloader_0.set_img_sz(image_size, grid_size, alignment=alignment)

        if self._dloader_1 is not None:
            self._dloader_1.set_img_sz(image_size, grid_size, alignment=alignment)

        self.idx_manager = GridIndexManager(self.get_data_shape(), self._grid_sz, self._img_sz, alignment)

    def get_mean_std(self):
        """
        Needed just for running the notebooks
        """
        return self._mean, self._std

    def get_data_shape(self):
        N = 0
        default_shape = None

        if self._dloader_0 is not None:
            default_shape = self._dloader_0.get_data_shape()
            N += default_shape[0]

        if self._dloader_1 is not None:
            default_shape = self._dloader_1.get_data_shape()
            N += default_shape[0]

        default_shape = list(default_shape)
        default_shape[0] = N
        return tuple(default_shape)

    def __len__(self):
        sz = 0
        if self._dloader_0 is not None:
            sz += int(self._subdset_types_prob[0] * len(self._dloader_0))
        if self._dloader_1 is not None:
            sz += int(self._subdset_types_prob[1] * len(self._dloader_1))
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
        mean_dict = {'subdset_0': {}, 'subdset_1': {}}
        std_dict = {'subdset_0': {}, 'subdset_1': {}}

        # mean_dict = {'target': mean_, 'mix': mean_.sum(axis=1, keepdims=True)}
        # std_dict = {'target': std_, 'mix': np.sqrt((std_**2).sum(axis=1, keepdims=True))}

        if self._dloader_0 is not None:
            mean_, std_ = self._dloader_0.compute_individual_mean_std()
            mean_for_input, std_for_input = self.compute_mean_std_for_input(self._dloader_0)
            mean_dict['subdset_0'] = {'target': mean_, 'input': mean_for_input}
            std_dict['subdset_0'] = {'target': std_, 'input': std_for_input}

        if self._dloader_1 is not None:
            mean_, std_ = self._dloader_1.compute_individual_mean_std()
            mean_for_input, std_for_input = self.compute_mean_std_for_input(self._dloader_1)
            mean_dict['subdset_1'] = {'target': mean_, 'input': mean_for_input}
            std_dict['subdset_1'] = {'target': std_, 'input': std_for_input}
        return mean_dict, std_dict

    def _compute_mean_std(self):
        mean_dict = {'subdset_0': {}, 'subdset_1': {}}
        std_dict = {'subdset_0': {}, 'subdset_1': {}}

        if self._dloader_0 is not None:
            mean_, std_ = self._dloader_0.compute_mean_std()
            mean_dict['subdset_0'] = {'target': mean_}
            std_dict['subdset_0'] = {'target': std_}

        if self._dloader_1 is not None:
            mean_, std_ = self._dloader_1.compute_mean_std()
            mean_dict['subdset_1'] = {'target': mean_}
            std_dict['subdset_1'] = {'target': std_}
        return mean_dict, std_dict

    def compute_mean_std(self):
        if self._use_one_mu_std is False:
            return self.compute_individual_mean_std()
        else:
            return self._compute_mean_std()

    def set_mean_std(self, mean_val, std_val):
        if self._dloader_0 is not None:
            self._dloader_0.set_mean_std(mean_val['subdset_0']['target'], std_val['subdset_0']['target'])
        if self._dloader_1 is not None:
            self._dloader_1.set_mean_std(mean_val['subdset_0']['target'], std_val['subdset_0']['target'])

    def get_loss_idx(self, dset_idx):
        raise NotImplementedError("Not implemented")

    def __getitem__(self, index):
        """
        Returns:
            (inp,tar,dset_label)
        """
        coin_flip = np.random.rand()
        prob_list = np.cumsum(self._subdset_types_prob)
        if coin_flip <= prob_list[0]:
            dset_idx = 0
        elif coin_flip > prob_list[0] and coin_flip <= prob_list[1]:
            dset_idx = 1

        loss_idx = self.get_loss_idx(dset_idx)

        dset = getattr(self, f'_dloader_{dset_idx}')
        idx = np.random.randint(len(dset))
        inp, tar = dset[idx]

        assert dset._input_is_sum is True
        return (inp, tar, dset_idx, loss_idx)

    def get_max_val(self):
        max_val0 = self._dloader_0.get_max_val()
        max_val1 = self._dloader_1.get_max_val()
        return [max_val0, max_val1]


class IBA1Ki67DataLoader(MultiDsetDloader):

    def get_loss_idx(self, dset_idx):
        if self._subdset_types[dset_idx] == SubDsetType.OnlyIba1:
            loss_idx = LossType.Elbo
        elif self._subdset_types[dset_idx] == SubDsetType.Iba1Ki64:
            loss_idx = LossType.ElboMixedReconstruction
        else:
            raise Exception("Invalid subdset type")
        return loss_idx


if __name__ == '__main__':
    from disentangle.configs.ht_iba1_ki64_config import get_config
    config = get_config()
    fpath = '/group/jug/ashesh/data/Stefania/20230327_Ki67_and_Iba1_trainingdata'
    dloader = IBA1Ki67DataLoader(
        config.data,
        fpath,
        datasplit_type=DataSplitType.Train,
        val_fraction=0.1,
        test_fraction=0.1,
        normalized_input=True,
        use_one_mu_std=True,
        enable_random_cropping=False,
        max_val=[1000, 2000],
    )
    mean_val, std_val = dloader.compute_mean_std()
    dloader.set_mean_std(mean_val, std_val)
    inp, tar, dset_idx, loss_idx = dloader[0]
    len(dloader)
    print('This is working')
