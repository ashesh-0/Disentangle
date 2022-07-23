"""
Here, the input image is of multiple resolutions. Target image is the same.
"""
from typing import List, Tuple, Union

import numpy as np
from skimage.transform import resize

from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader


class MultiScaleTiffDloader(MultiChDeterministicTiffDloader):
    def __init__(self,
                 img_sz: int,
                 fpath: str,
                 channel_1: int,
                 channel_2: int,
                 is_train: Union[None, bool] = None,
                 val_fraction=None,
                 normalized_input=None,
                 enable_rotation_aug: bool = False,
                 use_one_mu_std=None,
                 num_scales: int = None,
                 enable_random_cropping=False,
                 ):
        """
        Args:
            num_scales: The number of resolutions at which we want the input. Note that the target is formed at the
                        highest resolution.
        """
        super().__init__(img_sz,
                         fpath,
                         channel_1,
                         channel_2,
                         is_train=is_train,
                         val_fraction=val_fraction,
                         normalized_input=normalized_input,
                         enable_rotation_aug=enable_rotation_aug,
                         enable_random_cropping=enable_random_cropping,
                         use_one_mu_std=use_one_mu_std)
        self.num_scales = num_scales
        assert self.num_scales is not None
        self._scaled_data = [self._data]
        assert isinstance(self.num_scales, int) and self.num_scales >= 1

        for _ in range(1, self.num_scales):
            shape = self._scaled_data[-1].shape
            assert len(shape) == 4
            new_shape = (shape[0], shape[1] // 2, shape[2] // 2, shape[3])
            ds_data = resize(self._scaled_data[-1], new_shape)
            self._scaled_data.append(ds_data)

    def _load_scaled_img(self, scaled_index, index: int) -> Tuple[np.ndarray, np.ndarray]:
        imgs = self._scaled_data[scaled_index][index % self.N]
        return imgs[None, :, :, 0], imgs[None, :, :, 1]

    def _crop_img(self, img: np.ndarray, h_start: int, w_start: int):
        """
        Here, h_start, w_start could be negative. That simply means we need to pick the content from 0. So,
        the cropped image will be smaller than self._img_sz * self._img_sz
        """
        h_end = h_start + self._img_sz
        w_end = w_start + self._img_sz
        h_start = max(0, h_start)
        w_start = max(0, w_start)
        new_img = img[..., h_start:h_end, w_start:w_end]
        return new_img

    def _get_img(self, index: int):
        """
        Loads an image.
        Crops the image such that cropped image has content.
        """
        img1, img2 = self._load_img(index)
        assert self._img_sz is not None
        h, w = img1.shape[-2:]
        if self._enable_random_cropping:
            h_start, w_start = self._get_random_hw(h, w)
        else:
            h_start, w_start = self._get_deterministic_hw(index, h, w)
        img1_cropped = self._crop_flip_img(img1, h_start, w_start, False, False)
        img2_cropped = self._crop_flip_img(img2, h_start, w_start, False, False)

        h_center = h_start + self._img_sz // 2
        w_center = w_start + self._img_sz // 2
        img1_versions = [img1_cropped]
        img2_versions = [img2_cropped]
        for scale_idx in range(1, self.num_scales):
            img1, img2 = self._load_scaled_img(scale_idx, index)
            h_center = h_center // 2
            w_center = w_center // 2
            img1_padded = np.zeros_like(img1_versions[-1])
            img2_padded = np.zeros_like(img2_versions[-1])
            h_start = h_center - self._img_sz // 2
            w_start = w_center - self._img_sz // 2
            img1_cropped = self._crop_flip_img(img1, h_start, w_start, False, False)
            img2_cropped = self._crop_flip_img(img2, h_start, w_start, False, False)

            h_start = max(0, -h_start)
            w_start = max(0, -w_start)
            h_end = h_start + img1_cropped.shape[1]
            w_end = w_start + img1_cropped.shape[2]
            img1_padded[:, h_start:h_end, w_start:w_end] = img1_cropped
            img2_padded[:, h_start:h_end, w_start:w_end] = img2_cropped

            img1_versions.append(img1_padded)
            img2_versions.append(img2_padded)

        img1 = np.concatenate(img1_versions, axis=0)
        img2 = np.concatenate(img2_versions, axis=0)
        return img1, img2

    def __getitem__(self, index: int):
        img1, img2 = self._get_img(index)
        assert self._enable_rotation is False
        target = np.concatenate([img1[:1], img2[:1]], axis=0)
        if self._normalized_input:
            img1, img2 = self.normalize_img(img1, img2)

        inp = (0.5 * img1 + 0.5 * img2).astype(np.float32)
        return inp, target
