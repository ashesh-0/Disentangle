import enum
from disentangle.core import data_split_type
from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.train_val_data import get_train_val_data
import numpy as np
from typing import Union, Tuple
from copy import deepcopy
import ml_collections
import torch


class SingleChannelMultiDatasetDloader:

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

        assert isinstance(data_config.mix_fpath_list, tuple) or isinstance(data_config.mix_fpath_list, list)
        self._dsets = []
        self._channelwise_quantile = data_config.get('channelwise_quantile', False)

        for i, fpath_tuple in enumerate(zip(data_config.mix_fpath_list, data_config.ch1_fpath_list)):
            new_data_config = ml_collections.ConfigDict(data_config)
            new_data_config.mix_fpath = fpath_tuple[0]
            new_data_config.ch1_fpath = fpath_tuple[1]
            dset = SingleChannelDloader(new_data_config,
                                        fpath,
                                        datasplit_type=datasplit_type,
                                        val_fraction=val_fraction,
                                        test_fraction=test_fraction,
                                        normalized_input=normalized_input,
                                        enable_rotation_aug=enable_rotation_aug,
                                        enable_random_cropping=enable_random_cropping,
                                        use_one_mu_std=use_one_mu_std,
                                        allow_generation=allow_generation,
                                        max_val=max_val[i] if max_val is not None else None)
            self._dsets.append(dset)

    def compute_mean_std(self, allow_for_validation_data=False):
        mean_arr = []
        std_arr = []
        for dset in self._dsets:
            mean, std = dset.compute_mean_std(allow_for_validation_data=allow_for_validation_data)
            mean_arr.append(mean[None])
            std_arr.append(std[None])

        mean_vec = np.concatenate(mean_arr, axis=0)
        std_vec = np.concatenate(std_arr, axis=0)
        return mean_vec, std_vec

    def compute_individual_mean_std(self):
        mean_arr = []
        std_arr = []
        for i, dset in enumerate(self._dsets):
            mean_, std_ = dset.compute_individual_mean_std()
            mean_arr.append(mean_[None])
            std_arr.append(std_[None])
        return np.concatenate(mean_arr, axis=0), np.concatenate(std_arr, axis=0)

    def set_mean_std(self, mean_val, std_val):
        for i, dset in enumerate(self._dsets):
            dset.set_mean_std(mean_val[i], std_val[i])

    def get_max_val(self):
        max_val_arr = []
        for dset in self._dsets:
            max_val = dset.get_max_val()
            if self._channelwise_quantile:
                max_val_arr.append(np.array(max_val)[None])
            else:
                max_val_arr.append(max_val)

        if self._channelwise_quantile:
            # 2D
            return np.concatenate(max_val_arr, axis=0)
        else:
            # 1D
            return np.array(max_val_arr)

    def set_max_val(self, max_val):
        for i, dset in enumerate(self._dsets):
            dset.set_max_val(max_val[i])

    def _get_dataset_index(self, index):
        cum_index = 0
        for i, dset in enumerate(self._dsets):
            if index < cum_index + len(dset):
                return i, index - cum_index
            cum_index += len(dset)
        raise ValueError('Too large index:', index)

    def __getitem__(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        dset_index, data_index = self._get_dataset_index(index)
        output = (*self._dsets[dset_index][data_index], dset_index)
        assert len(output) == 3
        return output

    def __len__(self):
        tot_len = 0
        for dset in self._dsets:
            tot_len += len(dset)
        return tot_len


class SingleChannelDloader(MultiChDeterministicTiffDloader):

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
                 max_val=None):
        super().__init__(data_config, fpath, datasplit_type, val_fraction, test_fraction, normalized_input,
                         enable_rotation_aug, enable_random_cropping, use_one_mu_std, allow_generation, max_val)

        assert self._use_one_mu_std is False, 'One of channels is target. Other is input. They must have different mean/std'
        assert self._normalized_input is True, 'Now that input is not related to target, this must be done on dataloader side'

    def load_data(self, data_config, datasplit_type, val_fraction=None, test_fraction=None, allow_generation=None):
        data_dict = get_train_val_data(data_config,
                                       self._fpath,
                                       datasplit_type,
                                       val_fraction=val_fraction,
                                       test_fraction=test_fraction,
                                       allow_generation=allow_generation)
        self._data = np.concatenate([data_dict['mix'][..., None], data_dict['C1'][..., None]], axis=-1)
        self.N = len(self._data)

    def normalize_input(self, inp):
        return (inp - self._mean.squeeze()[0]) / self._std.squeeze()[0]

    def __getitem__(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        inp, target = self._get_img(index)
        if self._enable_rotation:
            # passing just the 2D input. 3rd dimension messes up things.
            rot_dic = self._rotation_transform(image=img1[0], mask=img2[0])
            img1 = rot_dic['image'][None]
            img2 = rot_dic['mask'][None]

        inp = self.normalize_input(inp)
        if isinstance(index, int):
            return inp, target

        _, grid_size = index
        return inp, target, grid_size


if __name__ == '__main__':
    from disentangle.configs.semi_supervised_config import get_config
    config = get_config()
    datadir = '/group/jug/ashesh/data/EMBL_halfsupervised/Demixing_3P/'
    val_fraction = 0.1
    test_fraction = 0.1

    dset = SingleChannelMultiDatasetDloader(config.data,
                                            datadir,
                                            datasplit_type=DataSplitType.Train,
                                            val_fraction=val_fraction,
                                            test_fraction=test_fraction,
                                            normalized_input=config.data.normalized_input,
                                            enable_rotation_aug=False,
                                            enable_random_cropping=False,
                                            use_one_mu_std=config.data.use_one_mu_std,
                                            allow_generation=False,
                                            max_val=None)

    mean_val, std_val = dset.compute_mean_std()
    dset.set_mean_std(mean_val, std_val)
    inp, tar, dset_index = dset[0]
    print(inp.shape, tar.shape, dset_index)
