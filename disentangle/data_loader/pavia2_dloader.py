import torch
from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.pavia2_rawdata_loader import Pavia2DataSetType, Pavia2DataSetChannels
import numpy as np
import ml_collections


class Pavia2V1Dloader:
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
        self._dloader_clean = self._dloader_bleedthrough = self._dloader_mix = None
        self._use_one_mu_std = use_one_mu_std

        self._mean = None
        self._std = None
        assert normalized_input is True, "We are doing the normalization in this dataloader.So you better pass it as True"
        # We don't normalalize inside the self._dloader_clean or bleedthrough. We normalize in this class.
        normalized_input = False
        if self._datasplit_type == DataSplitType.Train:
            assert enable_random_cropping is True
            dconf = ml_collections.ConfigDict(data_config)
            # take channels mean from this.
            dconf.dset_type = Pavia2DataSetType.JustMAGENTA
            self._clean_prob = dconf.dset_clean_sample_probab
            self._bleedthrough_prob = dconf.dset_bleedthrough_sample_probab
            assert self._clean_prob + self._bleedthrough_prob <=1

            self._dloader_clean = MultiChDeterministicTiffDloader(dconf,
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
            self._dloader_bleedthrough = MultiChDeterministicTiffDloader(dconf,
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
            
            dconf.dset_type = Pavia2DataSetType.MIXED
            self._dloader_mix = MultiChDeterministicTiffDloader(dconf,
                                                             fpath,
                                                             datasplit_type=datasplit_type,
                                                             val_fraction=val_fraction,
                                                             test_fraction=test_fraction,
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
            # we want to evaluate on mixed samples.
            self._clean_prob = 0.0
            self._bleedthrough_prob = 0.0
            self._dloader_mix = MultiChDeterministicTiffDloader(dconf,
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
        print(f'[{self.__class__.__name__}] BleedTh prob:{self._bleedthrough_prob} Clean prob:{self._clean_prob}')

    def sum_channels(self, data, first_index_arr, second_index_arr):
        fst_channel = data[..., first_index_arr].sum(axis=-1, keepdims=True)
        scnd_channel = data[..., second_index_arr].sum(axis=-1, keepdims=True)
        return np.concatenate([fst_channel, scnd_channel], axis=-1)

    def process_data(self):
        """
        We are ignoring the actin channel.
        We know that MTORQ(uise) has sigficant bleedthrough from TUBULIN channels. So, when MTORQ has no content, then 
        we sum it with TUBULIN so that tubulin has whole of its content. 
        When MTORQ has content, then we sum RFP670 with tubulin. This makes sure that tubulin channel has the same data distribution. 
        During validation/testing, we always feed sum of these three channels as the input.
        """
        
        if self._datasplit_type == DataSplitType.Train:
            self._dloader_clean._data = self._dloader_clean._data[..., [Pavia2DataSetChannels.NucRFP670, Pavia2DataSetChannels.TUBULIN]]
            self._dloader_bleedthrough._data = self._dloader_bleedthrough._data[..., [Pavia2DataSetChannels.NucMTORQ, Pavia2DataSetChannels.TUBULIN]]
            self._dloader_mix._data = self._dloader_mix._data[..., [Pavia2DataSetChannels.NucRFP670, Pavia2DataSetChannels.NucMTORQ, Pavia2DataSetChannels.TUBULIN]]
            self._dloader_mix._data = self.sum_channels(self._dloader_mix._data, [0,1],[2])
            # self._dloader_clean._data = self.sum_channels(self._dloader_clean._data, [1], [0, 2])
            # In bleedthrough dataset, the nucleus channel is empty. 
            # self._dloader_bleedthrough._data = self.sum_channels(self._dloader_bleedthrough._data, [0], [1, 2])
        else:
            self._dloader_mix._data = self._dloader_mix._data[..., [Pavia2DataSetChannels.NucRFP670, Pavia2DataSetChannels.NucMTORQ, Pavia2DataSetChannels.TUBULIN]]
            self._dloader_mix._data = self.sum_channels(self._dloader_mix._data, [0,1], [2])

    def __len__(self):
        sz = 0 
        if self._dloader_clean is not None:
            sz += len(self._dloader_clean)
        if self._dloader_bleedthrough is not None:
            sz += len(self._dloader_bleedthrough)
        if self._dloader_mix is not None:
            sz += len(self._dloader_mix)
        return sz

    def compute_individual_mean_std(self):
        mean_, std_ = self._dloader_clean.compute_individual_mean_std()
        mean_dict = {'target':mean_, 'mix':mean_.sum(axis=1,keepdims=True)}
        std_dict = {'target':std_,'mix':np.sqrt((std_**2).sum(axis=1,keepdims=True))}
        # NOTE: dataloader2 does not has clean channel. So, no mean should be computed on it. 
        # mean_std2 = self._dloader_bleedthrough.compute_individual_mean_std() if self._dloader_bleedthrough is not None else (None,None)
        return mean_dict, std_dict
        
        # if mean_std2 is None:
        #     return mean_std1

        # mean_val = (mean_std1[0] + mean_std2[0]) / 2
        # std_val = (mean_std1[1] + mean_std2[1]) / 2

        # return (mean_val, std_val)

    def compute_mean_std(self):
        if self._use_one_mu_std is False:
            return self.compute_individual_mean_std()
        else:
            raise ValueError('This must not be called. We want to compute individual mean so that they can be \
                passed on to the model')
            mean_std1 = self._dloader_clean.compute_mean_std()
            mean_std2 = self._dloader2.compute_mean_std() if self._dloader_bleedthrough is not None else (None, None)
            if mean_std2 is None:
                return mean_std1

            mean_val = (mean_std1[0] + mean_std2[0]) / 2
            std_val = (mean_std1[1] + mean_std2[1]) / 2

            return (mean_val, std_val)

    def set_mean_std(self, mean_val, std_val):
        self._mean = mean_val
        self._std = std_val

        # self._dloader_clean.set_mean_std(mean_val, std_val)
        # if self._dloader_bleedthrough is not None:
        #     self._dloader_bleedthrough.set_mean_std(mean_val, std_val)

    def normalize_input(self, inp):
        return (inp - self._mean['mix'][0])/self._std['mix'][0]

    def __getitem__(self, index):
        """
        Returns:
            (inp,tar,mixed_recons_flag): When mixed_recons_flag is set, then do only the mixed reconstruction. This is set when we've bleedthrough
        """
        coin_flip = np.random.rand()
        if self._datasplit_type == DataSplitType.Train:

            if coin_flip <= self._clean_prob:
                inp, tar = self._dloader_clean[np.random.randint(len(self._dloader_clean))]
                mixed_recons_flag = False
            elif coin_flip > self._clean_prob and coin_flip <= self._clean_prob + self._bleedthrough_prob:
                inp, tar = self._dloader_bleedthrough[np.random.randint(len(self._dloader_bleedthrough))]
                mixed_recons_flag = True
            else:
                inp, tar = self._dloader_mix[np.random.randint(len(self._dloader_mix))]
                mixed_recons_flag = True

            inp = 2 * inp  # dataloader takes the average of the two channels. To, undo that, we are multipying it with 2.
            inp = self.normalize_input(inp)
            return (inp, tar, False)
        
        else:
            inp, tar = self._dloader_mix[index]
            inp = 2* inp
            inp = self.normalize_input(inp)
            return (inp,tar,False)

    def get_max_val(self):
        max_val1 = self._dloader_clean.get_max_val()
        max_val2 = self._dloader_bleedthrough.get_max_val() if self._dloader_bleedthrough is not None else None
        return (max_val1, max_val2)

   
if __name__ == '__main__':
    from disentangle.configs.pavia2_config import get_config
    config = get_config()
    fpath = '/group/jug/ashesh/data/pavia2/'
    dloader = Pavia2V1Dloader(config.data,fpath,datasplit_type=DataSplitType.Train,val_fraction=0.1,
    test_fraction=0.1,normalized_input=True,use_one_mu_std=False,enable_random_cropping=True)
    mean_val, std_val = dloader.compute_mean_std()
    dloader.set_mean_std(mean_val, std_val)
    inp, tar, source = dloader[0]
    print('This is working')