import os
import pickle
from typing import Union

import numpy as np
from skimage.io import imread
from tqdm import tqdm

from git.objects import base


class NotMNISTNoisyLoader:
    """
    """
    def __init__(self, fpath_dict: dict, img_files_pkl, return_character_labels: bool = False) -> None:
        self._return_labels = return_character_labels

        # train/val split is defined in this file. It contains the list of images one needs to load from fpath_dict
        self._img_files_pkl = img_files_pkl
        fpath_noise_tuple = [(nlevel, dir) for nlevel, dir in fpath_dict.items()]
        fpath_noise_tuple = sorted(fpath_noise_tuple, key=lambda x: x[0])
        self.noise_levels = [x[0] for x in fpath_noise_tuple]
        self._noisy_datapath_list = [x[1] for x in fpath_noise_tuple]

        print(f'[{self.__class__.__name__}] Noise levels:', self.noise_levels)
        print(f'[{self.__class__.__name__}] Data fpaths:', self._noisy_datapath_list)
        self.N = None
        self._noise_level_count = len(self._noisy_datapath_list)
        self._all_data = self.load()

    def _load_one_noise_level(self, directory, img_files_dict):
        data_dict = {}
        for label in img_files_dict:
            data = np.zeros((len(img_files_dict[label]), 27, 27), dtype=np.float32)
            for i, img_fname in tqdm(enumerate(img_files_dict[label])):
                img_fpath = os.path.join(directory, label, img_fname)
                data[i] = imread(img_fpath)

            data = np.pad(data, pad_width=((0, 0), (1, 0), (1, 0)))
            data = data[:, None, ...].copy()

            data_dict[label] = data
        return data_dict

    def load(self):
        data = {}
        with open(self._img_files_pkl, 'rb') as f:
            img_files_dict = pickle.load(f)

        for noise_index, noise_directory in enumerate(self._noisy_datapath_list):
            data[noise_index] = self._load_one_noise_level(noise_directory, img_files_dict)

        sz = sum([data[noise_index][label].shape[0] for label in data[noise_index].keys()])
        self.labels = sorted(list(data[noise_index].keys()))
        label_sizes = [len(data[noise_index][label]) for label in self.labels]
        self.cumlative_label_sizes = [np.sum(label_sizes[:i]) for i in range(1, 1 + len(label_sizes))]

        for nlevel in data:
            assert sum([data[nlevel][label].shape[0] for label in data[nlevel].keys()]) == sz
        self.N = sz
        return data

    def _bs_label_index(self, base_index, start_label_index, end_label_index):
        """
        Binary search to find which label this index belongs to.
        """
        if end_label_index == start_label_index:
            return start_label_index

        mid = (start_label_index + end_label_index) // 2
        if self.cumlative_label_sizes[mid] <= base_index:
            return self._bs_label_index(base_index, mid + 1, end_label_index)
        else:
            return self._bs_label_index(base_index, start_label_index, mid)

    def get_img_index(self, base_index, label_index):
        if label_index == 0:
            return base_index
        else:
            return base_index - self.cumlative_label_sizes[label_index - 1]

    def get_label_index(self, index):
        base_index = self.get_base_index(index)
        return self._bs_label_index(base_index, 0, len(self.labels) - 1)

    def get_base_index(self, index):
        return index % self.N

    def get_noise_index(self, index):
        return index // self.N

    def __getitem__(self, index):
        noise_index = self.get_noise_index(index)
        label_index = self.get_label_index(index)
        img_index = self.get_img_index(self.get_base_index(index), label_index)
        img = self._all_data[noise_index][self.labels[label_index]][img_index]
        n_level = np.array([self.noise_levels[noise_index]])
        if self._return_labels:
            return img, n_level, self.labels[label_index]

        return img, n_level

    def get_mean_std(self):
        data = []
        for key in self._all_data:
            for label in self.labels:
                data.append(self._all_data[key][label])
        all_data = np.concatenate(data)
        return np.mean(all_data), np.std(all_data)

    def get_index(self, index, noise_level_index):
        return self.N * noise_level_index + index

    def __len__(self):
        return self.N * self._noise_level_count
