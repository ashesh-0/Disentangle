from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.pavia2_rawdata_loader import Pavia2DataSetType, Pavia2DataSetChannels
import numpy as np
import ml_collections


class Pavia2Dloader:
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
        self._dloader1 = self._dloader2 = None

        if self._datasplit_type == DataSplitType.Train:
            assert enable_random_cropping is True
            dconf = ml_collections.ConfigDict(data_config)
            dconf.dset_type = Pavia2DataSetType.JustMAGENTA
            self._type1_prob = dconf.dset_justmagenta_prob
            self._dloader1 = MultiChDeterministicTiffDloader(dconf,
                                                             fpath,
                                                             datasplit_type=datasplit_type,
                                                             val_fraction=0,
                                                             test_fraction=0,
                                                             normalized_input=normalized_input,
                                                             enable_rotation_aug=enable_rotation_aug,
                                                             enable_random_cropping=True,
                                                             use_one_mu_std=use_one_mu_std,
                                                             allow_generation=allow_generation,
                                                             max_val=None)

            dconf.dset_type = Pavia2DataSetType.JustCYAN
            self._dloader2 = MultiChDeterministicTiffDloader(dconf,
                                                             fpath,
                                                             datasplit_type=datasplit_type,
                                                             val_fraction=0,
                                                             test_fraction=0,
                                                             normalized_input=normalized_input,
                                                             enable_rotation_aug=enable_rotation_aug,
                                                             enable_random_cropping=True,
                                                             use_one_mu_std=use_one_mu_std,
                                                             allow_generation=allow_generation,
                                                             max_val=None)
        else:
            assert enable_random_cropping is False
            dconf = ml_collections.ConfigDict(data_config)
            dconf.dset_type = Pavia2DataSetType.MIXED
            self._type1_prob = 1.0
            self._dloader1 = MultiChDeterministicTiffDloader(dconf,
                                                             fpath,
                                                             datasplit_type=datasplit_type,
                                                             val_fraction=val_fraction,
                                                             test_fraction=test_fraction,
                                                             normalized_input=normalized_input,
                                                             enable_rotation_aug=enable_rotation_aug,
                                                             enable_random_cropping=enable_random_cropping,
                                                             use_one_mu_std=use_one_mu_std,
                                                             allow_generation=allow_generation,
                                                             max_val=max(max_val))
        self.process_data()

    def sum_channels(self, data, first_index_arr, second_index_arr):
        fst_channel = data[..., first_index_arr].sum(axis=-1, keepdims=True)
        scnd_channel = data[..., second_index_arr].sum(axis=-1, keepdims=True)
        return np.concatenate([fst_channel, scnd_channel], axis=-1)

    def process_data(self):
        """
        We are ignoring the actin channel.
        We know that MTORQ(uise) has sigficant bleedthrough from TUBULIN channels. So, when MTORQ has no content, then 
        we sum it with TUBULIN so that tubulin has whole of its content. 
        When MTORQ has content, then we sum MTORQ and RFP670. This makes sure that input to our model is always
         the sum of 2 nuclear channels and tubulin. this can also work since RFP670 is not expected to contain any
          content. 
        """
        self._dloader1._data = self._dloader1._data[
            ..., [Pavia2DataSetChannels.NucMTORQ, Pavia2DataSetChannels.NucRFP670, Pavia2DataSetChannels.TUBULIN]]

        if self._datasplit_type == DataSplitType.Train:
            self._dloader1._data = self.sum_channels(self._dloader1._data, [1], [0, 2])
            self._dloader2._data = self.sum_channels(self._dloader2._data, [0, 1], [2])
        else:
            self._dloader1._data = self.sum_channels(self._dloader1._data, [0, 1], [2])

    def __len__(self):
        return len(self._dloader1) + (len(self._dloader2) if self._dloader2 is not None else 0)

    def compute_individual_mean_std(self):
        mean_std1 = self._dloader1.compute_individual_mean_std()
        mean_std2 = self._dloader2.compute_individual_mean_std() if self._dloader2 is not None else None
        if mean_std2 is None:
            return mean_std1

        mean_val = (mean_std1[0] + mean_std2[0]) / 2
        std_val = (mean_std1[1] + mean_std2[1]) / 2

        return (mean_val, std_val)

    def __getitem__(self, index):
        """
        Returns:
            (inp,tar,mixed_recons_flag): When mixed_recons_flag is set, then do only the mixed reconstruction. This is set when we've bleedthrough
        """
        if self._enable_random_cropping:
            if np.random.rand() <= self._type1_prob:
                inp, tar = self._dloader1[np.random.randint(len(self._dloader1))]
                inp = 2 * inp  # dataloader takes the average of the two channels. To, undo that, we are multipying it with 2.
                return (inp, tar, False)
            else:
                inp, tar = self._dloader2[np.random.randint(len(self._dloader2))]
                inp = 2 * inp
                return (inp, tar, True)
        else:
            return self._dloader1[index]

    def get_max_val(self):
        max_val1 = self._dloader1.get_max_val()
        max_val2 = self._dloader2.get_max_val() if self._dloader2 is not None else None
        return (max_val1, max_val2)

    def compute_mean_std(self):
        mean_std1 = self._dloader1.compute_mean_std()
        mean_std2 = self._dloader2.compute_mean_std() if self._dloader2 is not None else (None, None)
        if mean_std2 is None:
            return mean_std1

        mean_val = (mean_std1[0] + mean_std2[0]) / 2
        std_val = (mean_std1[1] + mean_std2[1]) / 2

        return (mean_val, std_val)

    def set_mean_std(self, mean_val, std_val):
        self._dloader1.set_mean_std(mean_val, std_val)
        if self._dloader2 is not None:
            self._dloader2.set_mean_std(mean_val, std_val)
