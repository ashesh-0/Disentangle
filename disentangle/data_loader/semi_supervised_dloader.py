from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from typing import Union
import numpy as np


class MixedInputType:
    Aligned = 'aligned'
    Randomized = 'randomized'


class SemiSupDloader(MultiChDeterministicTiffDloader):
    def __init__(self,
                 data_config,
                 fpath: str,
                 is_train: Union[None, bool] = None,
                 val_fraction=None,
                 normalized_input=None,
                 enable_rotation_aug: bool = False,
                 use_one_mu_std=None,
                 mixed_input_type=None,
                 supervised_data_fraction=0.0,
                 ):
        super().__init__(data_config,
                         fpath,
                         is_train=is_train,
                         val_fraction=val_fraction,
                         normalized_input=normalized_input,
                         enable_rotation_aug=enable_rotation_aug,
                         enable_random_cropping=False,
                         use_one_mu_std=use_one_mu_std, )
        """
        Args:
            mixed_input_type: If set to 'aligned', the mixed input always comes from the co-aligned channels mixing. If 
                set to 'randomized', when the data is not supervised, it is created by mixing random crops of the two
                channels. Note that when data is supervised, then all three channels are in sync: mix = channel1 + channel2
                and both channel crops are aligned.
            supervised_data_fraction: What fraction of the data is supervised ?
        """
        self._mixed_input_type = mixed_input_type
        assert self._mixed_input_type in [MixedInputType.Aligned, MixedInputType.Randomized]
        self._supervised_data_fraction = supervised_data_fraction
        self._supervised_indices = self._get_supervised_indices()
        print(f'[{self.__class__.__name__}] Supf:{self._supervised_data_fraction}')

    def _get_supervised_indices(self):
        N = len(self)
        arr = np.random.permutation(N)
        return arr[:int(N * self._supervised_data_fraction)]

    def __getitem__(self, index):
        if index in self._supervised_indices:
            mixed, singlechannnels = super().__getitem__(index)
            return mixed, singlechannnels, True  # np.array([1])

        elif self._mixed_input_type == MixedInputType.Aligned:
            mixed, _ = super().__getitem__(index)

            index = np.random.randint(len(self))
            img1, _ = self._get_img(index)
            index = np.random.randint(len(self))
            _, img2 = self._get_img(index)
            singlechannels = np.concatenate([img1, img2], axis=0)
            return mixed, singlechannels, False  # np.array([0])
