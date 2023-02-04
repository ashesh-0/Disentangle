from typing import Tuple, Union

import albumentations as A
import numpy as np

from disentangle.core.data_type import DataType
from disentangle.data_loader.train_val_data import get_train_val_data
from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.patch_index_manager import GridIndexManager, GridAlignement


class MultiChDeterministicTiffDloader:

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
        """
        Here, an image is split into grids of size img_sz.
        Args:
            repeat_factor: Since we are doing a random crop, repeat_factor is
                given which can repeatedly sample from the same image. If self.N=12
                and repeat_factor is 5, then index upto 12*5 = 60 is allowed.
            use_one_mu_std: If this is set to true, then one mean and stdev is used
                for both channels. Otherwise, two different meean and stdev are used.

        """
        self._fpath = fpath
        self._data = self.N = None
        self.load_data(data_config,
                       datasplit_type,
                       val_fraction=val_fraction,
                       test_fraction=test_fraction,
                       allow_generation=allow_generation)
        self._normalized_input = normalized_input
        self._quantile = data_config.get('clip_percentile', 0.995)
        self._channelwise_quantile = data_config.get('channelwise_quantile', False)

        self.set_max_val_and_upperclip_data(max_val, datasplit_type)

        self._is_train = datasplit_type == DataSplitType.Train

        self._img_sz = self._grid_sz = self._repeat_factor = self.idx_manager = None
        if self._is_train:
            self.set_img_sz(data_config.image_size,
                            data_config.grid_size if 'grid_size' in data_config else data_config.image_size)
        else:
            self.set_img_sz(data_config.image_size, data_config.image_size)
        # For overlapping dloader, image_size and repeat_factors are not related. hence a different function.

        self._mean = None
        self._std = None
        self._use_one_mu_std = use_one_mu_std
        self._enable_rotation = enable_rotation_aug
        self._enable_random_cropping = enable_random_cropping
        # Randomly rotate [-90,90]

        self._rotation_transform = None
        if self._enable_rotation:
            self._rotation_transform = A.Compose([A.Flip(), A.RandomRotate90()])

        msg = self._init_msg()
        print(msg)

    def get_data_shape(self):
        return self._data.shape

    def load_data(self, data_config, datasplit_type, val_fraction=None, test_fraction=None, allow_generation=None):
        self._data = get_train_val_data(data_config,
                                        self._fpath,
                                        datasplit_type,
                                        val_fraction=val_fraction,
                                        test_fraction=test_fraction,
                                        allow_generation=allow_generation)
        self.N = len(self._data)

    def set_max_val_and_upperclip_data(self, max_val, datasplit_type):
        self.set_max_val(max_val, datasplit_type)
        self.upperclip_data()

    def upperclip_data(self):
        if isinstance(self.max_val, list):
            chN = self._data.shape[-1]
            assert chN == len(self.max_val)
            for ch in range(chN):
                ch_data = self._data[..., ch]
                ch_q = self.max_val[ch]
                ch_data[ch_data > ch_q] = ch_q
                self._data[..., ch] = ch_data
        else:
            self._data[self._data > self.max_val] = self.max_val

    def compute_max_val(self):
        if self._channelwise_quantile:
            return [np.quantile(self._data[..., i], self._quantile) for i in range(self._data.shape[-1])]
        else:
            return np.quantile(self._data, self._quantile)

    def set_max_val(self, max_val, datasplit_type):
        if datasplit_type == DataSplitType.Train:
            assert max_val is None
            self.max_val = self.compute_max_val()
        else:
            assert max_val is not None
            self.max_val = max_val

    def get_max_val(self):
        return self.max_val

    def get_img_sz(self):
        return self._img_sz

    def set_img_sz(self, image_size, grid_size, alignment=GridAlignement.LeftTop):
        """
        If one wants to change the image size on the go, then this can be used.
        Args:
            image_size: size of one patch
            grid_size: frame is divided into square grids of this size. A patch centered on a grid having size `image_size` is returned.
        """
        self._img_sz = image_size
        self._grid_sz = grid_size
        self.idx_manager = GridIndexManager(self._data.shape, self._grid_sz, self._img_sz, alignment)
        self.set_repeat_factor()

    def set_repeat_factor(self):
        self._repeat_factor = self.idx_manager.grid_rows(self._grid_sz) * self.idx_manager.grid_cols(self._grid_sz)

    def _init_msg(self, ):
        msg = f'[{self.__class__.__name__}] Sz:{self._img_sz}'
        msg += f' Train:{int(self._is_train)} N:{self.N} NumPatchPerN:{self._repeat_factor}'
        msg += f' NormInp:{self._normalized_input}'
        msg += f' SingleNorm:{self._use_one_mu_std}'
        msg += f' Rot:{self._enable_rotation}'
        msg += f' RandCrop:{self._enable_random_cropping}'
        msg += f' Q:{self._quantile}'
        return msg

    def _crop_imgs(self, index, img1: np.ndarray, img2: np.ndarray):
        h, w = img1.shape[-2:]
        if self._img_sz is None:
            return img1, img2, {'h': [0, h], 'w': [0, w], 'hflip': False, 'wflip': False}

        if self._enable_random_cropping:
            h_start, w_start = self._get_random_hw(h, w)
        else:
            h_start, w_start = self._get_deterministic_hw(index)

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

    def __len__(self):
        return self.N * self._repeat_factor

    def _load_img(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx = index[0]

        imgs = self._data[self.idx_manager.get_t(idx)]
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

    def _get_deterministic_hw(self, index: Union[int, Tuple[int, int]]):
        if isinstance(index, int):
            idx = index
            grid_size = self._grid_sz
        else:
            idx, grid_size = index

        return self.idx_manager.get_deterministic_hw(idx, grid_size=grid_size)

    def compute_individual_mean_std(self):
        # numpy 1.19.2 has issues in computing for large arrays. https://github.com/numpy/numpy/issues/8869
        # mean = np.mean(self._data, axis=(0, 1, 2))
        # std = np.std(self._data, axis=(0, 1, 2))
        mean = np.array([self._data[..., 0].mean(), self._data[..., 1].mean()])
        std = np.array([self._data[..., 0].std(), self._data[..., 1].std()])

        return mean[None, :, None, None], std[None, :, None, None]

    def compute_mean_std(self, allow_for_validation_data=False):
        """
        Note that we must compute this only for training data.
        """
        assert self._is_train is True or allow_for_validation_data, 'This is just allowed for training data'
        if self._use_one_mu_std is True:
            mean = np.mean(self._data, keepdims=True).reshape(1, 1, 1, 1)
            std = np.std(self._data, keepdims=True).reshape(1, 1, 1, 1)
            mean = np.repeat(mean, 2, axis=1)
            std = np.repeat(std, 2, axis=1)
            return mean, std
        elif self._use_one_mu_std is False:
            return self.compute_individual_mean_std()

        elif self._use_one_mu_std is None:
            return np.array([0.0, 0.0]).reshape(1, 2, 1, 1), np.array([1.0, 1.0]).reshape(1, 2, 1, 1)

    def _get_random_hw(self, h: int, w: int):
        """
        Random starting position for the crop for the img with index `index`.
        """
        if h != self._img_sz:
            h_start = np.random.choice(h - self._img_sz)
            w_start = np.random.choice(w - self._img_sz)
        else:
            h_start = 0
            w_start = 0
        return h_start, w_start

    def _get_img(self, index: Union[int, Tuple[int, int]]):
        """
        Loads an image.
        Crops the image such that cropped image has content.
        """
        img1, img2 = self._load_img(index)
        cropped_img1, cropped_img2 = self._crop_imgs(index, img1, img2)[:2]
        return cropped_img1, cropped_img2

    def __getitem__(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
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

        if isinstance(index, int):
            return inp, target

        _, grid_size = index
        return inp, target, grid_size
