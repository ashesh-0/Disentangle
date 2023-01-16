from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.train_val_data import get_train_val_data
import numpy as np
from types import Union, Tuple


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

    def load_data(self, data_config, datasplit_type, val_fraction=None, test_fraction=None, allow_generation=None):
        data_dict = get_train_val_data(data_config,
                                       self._fpath,
                                       datasplit_type,
                                       val_fraction=val_fraction,
                                       test_fraction=test_fraction,
                                       allow_generation=allow_generation)
        self._data = np.concatenate([data_dict['mix'], data_dict['C1']], axis=-1)
        self.N = len(self._data)

    def __getitem__(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        inp, target = self._get_img(index)
        if self._enable_rotation:
            # passing just the 2D input. 3rd dimension messes up things.
            rot_dic = self._rotation_transform(image=img1[0], mask=img2[0])
            img1 = rot_dic['image'][None]
            img2 = rot_dic['mask'][None]

        assert self._normalized_input is False

        if isinstance(index, int):
            return inp, target

        _, grid_size = index
        return inp, target, grid_size
