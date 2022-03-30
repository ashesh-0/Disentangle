import os
import pickle
from typing import Union

import numpy as np
from skimage.io import imread
from tqdm import tqdm

from disentangle.core.tiff_reader import load_tiff


class CngbMitoActinLoader:
    def __init__(self, img_sz, mito_fpath, actin_fpath, enable_flips=False, thresh=None):
        self._img_sz = img_sz
        self._mito_fpath = mito_fpath
        self._actin_fpath = actin_fpath

        self._mito_data = load_tiff(self._mito_fpath).astype(np.float32)
        fac = 255 / self._mito_data.max()
        self._mito_data *= fac

        self._actin_data = load_tiff(self._actin_fpath).astype(np.float32)
        fac = 255 / self._actin_data.max()
        self._actin_data *= fac

        self._enable_flips = False
        assert len(self._mito_data) == len(self._actin_data)
        self.N = len(self._mito_data)
        self._avg_cropped_count = 1
        self._called_count = 0
        self._thresh = thresh
        assert self._thresh is not None

    def _crop_random(self, img1, img2):
        h, w = img1.shape[-2:]
        if self._img_sz is None:
            return img1, img2, {'h': [0, h], 'w': [0, w], 'hflip': False, 'wflip': False}

        h_start, w_start, h_flip, w_flip = self._get_random_hw(h, w)
        if self._enable_flips is False:
            h_flip = False
            w_flip = False

        img1 = self._crop_img(img1, h_start, w_start, h_flip, w_flip)
        img2 = self._crop_img(img2, h_start, w_start, h_flip, w_flip)

        return img1, img2, {
            'h': [h_start, h_start + self._img_sz],
            'w': [w_start, w_start + self._img_sz],
            'hflip': h_flip,
            'wflip': w_flip,
        }

    def _crop_img(self, img, h_start, w_start, h_flip, w_flip):
        new_img = img[..., h_start:h_start + self._img_sz, w_start:w_start + self._img_sz]
        if h_flip:
            new_img = new_img[..., ::-1, :]
        if w_flip:
            new_img = new_img[..., :, ::-1]

        return new_img.astype(np.float32)

    def _get_random_hw(self, h, w):
        """
        Random starting position for the crop for the img with index `index`.
        """
        h_start = np.random.choice(h - self._img_sz)
        w_start = np.random.choice(w - self._img_sz)
        h_flip, w_flip = np.random.choice(2, size=2) == 1
        return h_start, w_start, h_flip, w_flip

    def metric(self, img):
        return np.std(img)

    def in_allowed_range(self, metric_val):
        return metric_val >= self._thresh

    def __len__(self):
        return self.N

    def _is_content_present(self, img1, img2):
        met1 = self.metric(img1)
        met2 = self.metric(img2)
        print('Metric', met1, met2)
        if self.in_allowed_range(met1) or self.in_allowed_range(met2):
            return True
        return False

    def _load_img(self, index):
        img1 = self._mito_data[index]
        img2 = self._actin_data[index]
        return img1[None], img2[None]

    def _get_img(self, index):
        """
        Loads an image. 
        Crops the image such that cropped image has content.
        """
        img1, img2 = self._load_img(index)
        cropped_img1, cropped_img2 = self._crop_random(img1, img2)[:2]
        self._called_count += 1
        cropped_count = 1
        while (not self._is_content_present(cropped_img1, cropped_img2)):
            cropped_img1, cropped_img2 = self._crop_random(img1, img2)[:2]
            cropped_count += 1

        self._avg_cropped_count = (
            (self._called_count - 1) * self._avg_cropped_count + cropped_count) / self._called_count
        return cropped_img1, cropped_img2

    def __getitem__(self, index):
        img1, img2 = self._get_img(index)

        inp = (0.5 * img1 + 0.5 * img2).astype(np.float32)
        target = np.concatenate([img1, img2], axis=0)
        return inp, target

    def get_mean_std(self):
        return 0.0, 255.0
