import numpy as np

from disentangle.core.data_split_type import DataSplitType
from disentangle.core.data_type import DataType
from disentangle.core.empty_patch_fetcher import EmptyPatchFetcher
from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from disentangle.data_loader.patch_index_manager import GridAlignement, GridIndexManager
from disentangle.data_loader.train_val_data import get_train_val_data


class SingleFileDset(MultiChDeterministicTiffDloader):

    def __init__(self,
                 preloaded_data,
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
                 max_val=None,
                 grid_alignment=GridAlignement.LeftTop,
                 overlapping_padding_kwargs=None,
                 print_vars=True):
        self._preloaded_data = preloaded_data
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
                         max_val=max_val,
                         grid_alignment=grid_alignment,
                         overlapping_padding_kwargs=overlapping_padding_kwargs,
                         print_vars=print_vars)

    def rm_bkground_set_max_val_and_upperclip_data(self, max_val, datasplit_type):
        pass

    def load_data(self, data_config, datasplit_type, val_fraction=None, test_fraction=None, allow_generation=None):
        self._data = self._preloaded_data
        self.N = len(self._data)


class MultiFileDset:
    """
    Here, we have multiple files, each file can have a different spatial dimension and number of frames (Z stack).
    """

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
                 max_val=None,
                 grid_alignment=GridAlignement.LeftTop,
                 overlapping_padding_kwargs=None):

        self._fpath = fpath
        self._background_quantile = data_config.get('background_quantile', 0.0)
        data = get_train_val_data(data_config,
                                  self._fpath,
                                  datasplit_type,
                                  val_fraction=val_fraction,
                                  test_fraction=test_fraction)
        self.dsets = []

        for i in range(len(data)):
            self.dsets.append(
                SingleFileDset(data[i][None],
                               data_config,
                               '',
                               datasplit_type=datasplit_type,
                               val_fraction=val_fraction,
                               test_fraction=test_fraction,
                               normalized_input=normalized_input,
                               enable_rotation_aug=enable_rotation_aug,
                               enable_random_cropping=enable_random_cropping,
                               use_one_mu_std=use_one_mu_std,
                               allow_generation=False,
                               max_val=max_val,
                               grid_alignment=grid_alignment,
                               overlapping_padding_kwargs=overlapping_padding_kwargs,
                               print_vars=i == len(data) - 1))

        self.rm_bkground_set_max_val_and_upperclip_data(max_val, datasplit_type)
        count = 0
        avg_height = 0
        avg_width = 0
        for dset in self.dsets:
            shape = dset.get_data_shape()
            avg_height += shape[1]
            avg_width += shape[2]
            count += shape[0]

        avg_height = int(avg_height / len(self.dsets))
        avg_width = int(avg_width / len(self.dsets))
        print(f'{self.__class__.__name__} avg height: {avg_height}, avg width: {avg_width}, count: {count}')

    def rm_bkground_set_max_val_and_upperclip_data(self, max_val, datasplit_type):
        assert self._background_quantile == 0.0
        self.set_max_val(max_val, datasplit_type)
        self.upperclip_data()

    def set_mean_std(self, mean_val, std_val):
        for dset in self.dsets:
            dset.set_mean_std(mean_val, std_val)

    def compute_max_val(self):
        max_val_arr = []
        for dset in self.dsets:
            max_val_arr.append(dset.compute_max_val())
        return np.max(max_val_arr)

    def set_max_val(self, max_val, datasplit_type):
        if datasplit_type == DataSplitType.Train:
            assert max_val is None
            max_val = self.compute_max_val()
        for dset in self.dsets:
            dset.set_max_val(max_val, datasplit_type)

    def upperclip_data(self):
        for dset in self.dsets:
            dset.upperclip_data()

    def get_max_val(self):
        return self.dsets[0].get_max_val()

    def compute_mean_std(self):
        cum_mean = 0
        cum_std = 0
        for dset in self.dsets:
            mean, std = dset.compute_mean_std()
            cum_mean += mean
            cum_std += std
        return cum_mean / len(self.dsets), cum_std / len(self.dsets)

    def __len__(self):
        out = 0
        for dset in self.dsets:
            out += len(dset)
        return out

    def __getitem__(self, idx):
        cum_len = 0
        for dset in self.dsets:
            cum_len += len(dset)
            if idx < cum_len:
                rel_idx = idx - (cum_len - len(dset))
                return dset[rel_idx]

        raise IndexError('Index out of range')
