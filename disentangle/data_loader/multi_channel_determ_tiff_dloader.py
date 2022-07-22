from typing import Tuple, Union

import albumentations as A
import numpy as np

from disentangle.data_loader.multi_channel_train_val_data import train_val_data


class MultiChDeterministicTiffDloader:
    def __init__(self,
                 img_sz: int,
                 fpath: str,
                 channel_1: int,
                 channel_2: int,
                 is_train: Union[None, bool] = None,
                 val_fraction=None,
                 normalized_input=None,
                 enable_rotation_aug: bool = False,
                 enable_random_cropping: bool = False,
                 use_one_mu_std=None):
        """
        Here, an image is split into grids of size img_sz.
        Args:
            repeat_factor: Since we are doing a random crop, repeat_factor is
                given which can repeatedly sample from the same image. If self.N=12
                and repeat_factor is 5, then index upto 12*5 = 60 is allowed.
            use_one_mu_std: If this is set to true, then one mean and stdev is used
                for both channels. Otherwise, two different meean and stdev are used.

        """
        self._img_sz = img_sz
        self._fpath = fpath

        self._data = train_val_data(self._fpath, is_train, channel_1, channel_2, val_fraction=val_fraction)

        self._normalized_input = normalized_input
        max_val = np.quantile(self._data, 0.995)
        self._data[self._data > max_val] = max_val

        self.N = len(self._data)
        self._repeat_factor = (self._data.shape[-2] // self._img_sz) ** 2
        self._is_train = is_train
        self._mean = None
        self._std = None
        self._use_one_mu_std = use_one_mu_std
        self._enable_rotation = enable_rotation_aug
        self._enable_random_cropping = enable_random_cropping
        # Randomly rotate [-90,90]

        self._rotation_transform = None
        if self._enable_rotation:
            self._rotation_transform = A.Compose([A.Flip(), A.RandomRotate90()])

        msg = f'[{self.__class__.__name__}] Sz:{img_sz} Ch:{channel_1},{channel_2}'
        msg += f' Train:{int(is_train)} N:{self.N} NumPatchPerN:{self._repeat_factor}'
        msg += f' NormInp:{self._normalized_input}'
        msg += f' SingleNorm:{self._use_one_mu_std}'
        msg += f' Rot:{self._enable_rotation}'
        msg += f' RandCrop:{self._enable_random_cropping}'
        print(msg)

    def _crop_imgs(self, index, img1: np.ndarray, img2: np.ndarray):
        h, w = img1.shape[-2:]
        if self._img_sz is None:
            return img1, img2, {'h': [0, h], 'w': [0, w], 'hflip': False, 'wflip': False}

        if self._enable_random_cropping:
            h_start, w_start = self._get_random_hw(h, w)
        else:
            h_start, w_start = self._get_deterministic_hw(index, h, w)

        img1 = self._crop_flip_img(img1, h_start, w_start, False, False)
        img2 = self._crop_flip_img(img2, h_start, w_start, False, False)

        return img1, img2, {
            'h': [h_start, h_start + self._img_sz],
            'w': [w_start, w_start + self._img_sz],
            'hflip': False,
            'wflip': False,
        }

    def _crop_img(self, img: np.ndarray, h_start: int, w_start: int):
        new_img = img[..., h_start:h_start + self._img_sz, w_start:w_start + self._img_sz]
        return new_img

    def _crop_flip_img(self, img: np.ndarray, h_start: int, w_start: int, h_flip: bool, w_flip: bool):
        new_img = self._crop_img(img, h_start, w_start)
        if h_flip:
            new_img = new_img[..., ::-1, :]
        if w_flip:
            new_img = new_img[..., :, ::-1]

        return new_img.astype(np.float32)

    def _get_deterministic_hw(self, index: int, h: int, w: int):
        """
        Fixed starting position for the crop for the img with index `index`.
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
        return self._mean, self._std

    def set_mean_std(self, mean_val, std_val):
        self._mean = mean_val
        self._std = std_val

    def normalize_img(self, img1, img2):
        mean, std = self.get_mean_std()
        mean = mean.squeeze()
        std = std.squeeze()
        img1 = (img1 - mean[0]) / std[0]
        img2 = (img2 - mean[1]) / std[1]
        return img1, img2

    def compute_mean_std(self, allow_for_validation_data=False):
        """
        Note that we must compute this only for training data.
        """
        assert self._is_train is True or allow_for_validation_data, 'This is just allowed for training data'
        if self._use_one_mu_std:
            mean = np.mean(self._data, keepdims=True).reshape(1, 1, 1, 1)
            std = np.std(self._data, keepdims=True).reshape(1, 1, 1, 1)
            mean = np.repeat(mean, 2, axis=1)
            std = np.repeat(std, 2, axis=1)
            return mean, std
        else:
            mean = np.mean(self._data, axis=(0, 1, 2))
            std = np.std(self._data, axis=(0, 1, 2))
            return mean[None, :, None, None], std[None, :, None, None]

    def _get_random_hw(self, h: int, w: int):
        """
        Random starting position for the crop for the img with index `index`.
        """
        h_start = np.random.choice(h - self._img_sz)
        w_start = np.random.choice(w - self._img_sz)
        return h_start, w_start

    def _get_img(self, index: int):
        """
        Loads an image.
        Crops the image such that cropped image has content.
        """
        img1, img2 = self._load_img(index)
        cropped_img1, cropped_img2 = self._crop_imgs(index, img1, img2)[:2]
        return cropped_img1, cropped_img2

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        img1, img2 = self._get_img(index)
        if self._enable_rotation:
            # passing just the 2D input. 3rd dimension messes up things.
            rot_dic = self._rotation_transform(image=img1[0], mask=img2[0])
            img1 = rot_dic['image'][None]
            img2 = rot_dic['mask'][None]
        target = np.concatenate([img1, img2], axis=0)
        if self._normalized_input:
            img1, img2 = self.normalize_img(img1, img2)

        inp = (0.5 * img1 + 0.5 * img2).astype(np.float32)
        return inp, target
