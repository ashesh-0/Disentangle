from typing import Tuple, Union

import numpy as np

from disentangle.data_loader.multi_channel_train_val_data import train_val_data
from disentangle.data_loader.tiff_dloader import TiffLoader


class MultiChTiffDloader(TiffLoader):
    def __init__(self,
                 img_sz: int,
                 fpath: str,
                 channel_1: int,
                 channel_2: int,
                 is_train: Union[None, bool] = None,
                 val_fraction=None,
                 enable_flips: bool = False,
                 repeat_factor: int = 1,
                 thresh: float = None,
                 normalized_input=None):
        super().__init__(img_sz,
                         enable_flips=enable_flips,
                         thresh=thresh,
                         repeat_factor=repeat_factor,
                         normalized_input=normalized_input)
        self._fpath = fpath

        self._data = train_val_data(self._fpath, is_train, channel_1, channel_2, val_fraction=val_fraction)

        max_val = np.quantile(self._data, 0.995)
        self._data[self._data > max_val] = max_val

        self.N = len(self._data)

        msg = f'[{self.__class__.__name__}] Sz:{img_sz} Ch:{channel_1},{channel_2}'
        msg += f' Train:{int(is_train)} N:{self.N} Flip:{int(enable_flips)} Repeat:{repeat_factor}'
        msg += f' Thresh:{thresh}'
        print(msg)

    def _load_img(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        imgs = self._data[index]
        return imgs[None, :, :, 0], imgs[None, :, :, 1]

    def get_mean_std(self):
        # mean = np.mean(self._data, axis=(0, 1, 2))
        # std = np.std(self._data, axis=(0, 1, 2))
        # return mean[None, :, None, None], std[None, :, None, None]
        mean = np.mean(self._data, keepdims=True).reshape(1, 1, 1, 1)
        std = np.std(self._data, keepdims=True).reshape(1, 1, 1, 1)
        mean = np.repeat(mean, 2, axis=1)
        std = np.repeat(std, 2, axis=1)
        return mean, std
