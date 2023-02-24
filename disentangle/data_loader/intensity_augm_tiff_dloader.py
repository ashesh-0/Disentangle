"""
Here, the motivation is to have Intensity based augmentation We'll change the amount of the overlap in order for it.
"""
from typing import Union

import numpy as np

from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader


class IntensityAugTiffDloader(MultiChDeterministicTiffDloader):

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
                 intensity_scaling_augmentation_list: Union[List[int], None] = None,
                 max_val=None):
        super().__init__(data_config,
                         fpath,
                         datasplit_type=datasplit_type,
                         val_fraction=val_fraction,
                         test_fraction=test_fraction,
                         normalized_input=normalized_input,
                         enable_rotation_aug=enable_rotation_aug,
                         enable_random_cropping=enable_random_cropping,
                         use_one_mu_std=use_one_mu_std,
                         allow_generation=allow_generation,
                         max_val=max_val)
        assert self._data.shape[-1] == 2
        self._ch1_min_scale_factor = data_config.ch1_min_scale_factor
        self._ch1_max_scale_factor = data_config.ch1_max_scale_factor
        assert self._use_one_mu_std is False, "We need individual channels mean and std to be able to get correct mean for scale factors."

    def _sample_scale_factor(self):
        if self._ch1_min_scale_factor is None or self._ch1_max_scale_factor is None:
            return None

        diff = self._ch1_max_scale_factor - self._ch1_min_scale_factor
        diff = np.random.rand() * diff
        factor = self._ch1_min_scale_factor + diff
        return factor

    def _compute_mean_std_with_scale_factor(self, factor):
        mean, std = self.get_mean_std()
        mean = mean.squeeze()
        std = std.squeeze()
        mean = mean[0] * factor + mean[1] * (1 - factor)
        std = std[0] * factor + std[1] * (1 - factor)
        return mean, std

    def _compute_input(self, img_tuples):
        factor = self._sample_scale_factor()
        assert factor is not None

        assert len(img_tuples) == 2
        assert self._normalized_input is True, "normalization should happen here"

        inp = img_tuples[0] * factor + img_tuples[1] * (1 - factor)
        mean, std = self._compute_mean_std_with_scale_factor(factor)
        inp = (inp - mean) / std
        return inp.astype(np.float32)
