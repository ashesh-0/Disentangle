from typing import Tuple, Union

import numpy as np

from disentangle.data_loader.multi_channel_train_val_data import train_val_data


class MultiChDeterministicTiffDloader:
    def __init__(
        self,
        img_sz: int,
        fpath: str,
        channel_1: int,
        channel_2: int,
        is_train: Union[None, bool] = None,
        val_fraction=None,
    ):
        """
        Here, an image is split into grids of size img_sz. 
        Args:
            repeat_factor: Since we are doing a random crop, repeat_factor is
            given which can repeatedly sample from the same image. If self.N=12
            and repeat_factor is 5, then index upto 12*5 = 60 is allowed.

        """
        self._img_sz = img_sz
        self._fpath = fpath

        self._data = train_val_data(self._fpath, is_train, channel_1, channel_2, val_fraction=val_fraction)

        max_val = np.quantile(self._data, 0.995)
        self._data[self._data > max_val] = max_val

        self.N = len(self._data)
        self._repeat_factor = (self._data.shape[-2] // self._img_sz)**2

        msg = f'[{self.__class__.__name__}] Sz:{img_sz} Ch:{channel_1},{channel_2}'
        msg += f' Train:{int(is_train)} N:{self.N} Repeat:{self._repeat_factor}'
        print(msg)

    def _crop_determinstic(self, index, img1: np.ndarray, img2: np.ndarray):
        h, w = img1.shape[-2:]
        if self._img_sz is None:
            return img1, img2, {'h': [0, h], 'w': [0, w], 'hflip': False, 'wflip': False}

        h_start, w_start = self._get_deterministic_hw(index, h, w)
        img1 = self._crop_img(img1, h_start, w_start, False, False)
        img2 = self._crop_img(img2, h_start, w_start, False, False)

        return img1, img2, {
            'h': [h_start, h_start + self._img_sz],
            'w': [w_start, w_start + self._img_sz],
            'hflip': False,
            'wflip': False,
        }

    def _crop_img(self, img: np.ndarray, h_start: int, w_start: int, h_flip: bool, w_flip: bool):
        new_img = img[..., h_start:h_start + self._img_sz, w_start:w_start + self._img_sz]
        if h_flip:
            new_img = new_img[..., ::-1, :]
        if w_flip:
            new_img = new_img[..., :, ::-1]

        return new_img.astype(np.float32)

    def _get_deterministic_hw(self, index: int, h: int, w: int):
        """
        Random starting position for the crop for the img with index `index`.
        """
        assert h == w
        factor = index // self.N
        nrows = h // self._img_sz

        ith_row = factor // nrows
        jth_col = factor % nrows
        h_start = ith_row * self._img_sz
        w_start = jth_col * self._img_sz
        return h_start, w_start

    def __len__(self):
        return self.N * self._repeat_factor

    def _load_img(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        imgs = self._data[index % self.N]
        return imgs[None, :, :, 0], imgs[None, :, :, 1]

    def get_mean_std(self):
        return self._data.mean(), self._data.std()

    def _get_img(self, index: int):
        """
        Loads an image. 
        Crops the image such that cropped image has content.
        """
        img1, img2 = self._load_img(index)
        cropped_img1, cropped_img2 = self._crop_determinstic(index, img1, img2)[:2]
        return cropped_img1, cropped_img2

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        img1, img2 = self._get_img(index)
        inp = (0.5 * img1 + 0.5 * img2).astype(np.float32)
        target = np.concatenate([img1, img2], axis=0)
        return inp, target
