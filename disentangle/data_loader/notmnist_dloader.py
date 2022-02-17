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
    def __init__(self, data_fpath: str, img_files_pkl) -> None:

        # train/val split is defined in this file. It contains the list of images one needs to load from fpath_dict
        self._img_files_pkl = img_files_pkl
        self._datapath = data_fpath

        print(f'[{self.__class__.__name__}] Data fpath:', self._datapath)
        self.N = None
        self._all_data = self.load()

    def _load_one_directory(self, directory, img_files_dict):
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
        with open(self._img_files_pkl, 'rb') as f:
            img_files_dict = pickle.load(f)

        data = self._load_one_directory(self._datapath, img_files_dict)

        sz = sum([data[label].shape[0] for label in data.keys()])
        self.labels = sorted(list(data.keys()))
        label_sizes = [len(data[label]) for label in self.labels]
        self.cumlative_label_sizes = [np.sum(label_sizes[:i]) for i in range(1, 1 + len(label_sizes))]

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
        return self._bs_label_index(index, 0, len(self.labels) - 1)

    def __getitem__(self, index):
        label_index = self.get_label_index(index)
        img_index = self.get_img_index(index, label_index)
        img = self._all_data[self.labels[label_index]][img_index]
        return img, self.labels[label_index]

    def get_mean_std(self):
        data = []
        for label in self.labels:
            data.append(self._all_data[label])
        all_data = np.concatenate(data)
        return np.mean(all_data), np.std(all_data)

    def __len__(self):
        return self.N
