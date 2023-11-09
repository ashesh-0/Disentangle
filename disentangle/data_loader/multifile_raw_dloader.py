import os
from ast import literal_eval as make_tuple
from collections.abc import Sequence
from random import shuffle
from typing import List

import numpy as np

from disentangle.core.custom_enum import Enum
from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples
from disentangle.core.tiff_reader import load_tiff


class TwoChannelData(Sequence):
    """
    each element in data_arr should be a N*H*W array
    """

    def __init__(self, data_arr1, data_arr2):
        assert len(data_arr1) == len(data_arr2)
        self._data = []
        for i in range(len(data_arr1)):
            assert data_arr1[i].shape == data_arr2[i].shape
            assert len(data_arr1[i].shape) == 3, 'Each element in data arrays should be a N*H*W'
            self._data.append(np.concatenate([data_arr1[i][..., None], data_arr2[i][..., None]], axis=-1))

    def __len__(self):
        n = 0
        for x in self._data:
            n += x.shape[0]
        return n

    def __getitem__(self, idx):
        n = 0
        for x in self._data:
            if idx < n + x.shape[0]:
                return x[idx - n]
            n += x.shape[0]
        raise IndexError('Index out of range')


class SubDsetType(Enum):
    TwoChannel = 0
    OneChannel = 1


def subset_data(dataA, dataB, dataidx_list):
    dataidx_list = sorted(dataidx_list)
    subset_dataB = []
    subset_dataA = []
    cur_dataidx = 0
    cumulative_datacount = 0
    for arr_idx in range(len(dataA)):
        for data_idx in range(len(dataA[arr_idx])):
            cumulative_datacount += 1
            if dataidx_list[cur_dataidx] == cumulative_datacount - 1:
                subset_dataB.append(dataB[arr_idx][data_idx:data_idx + 1])
                subset_dataA.append(dataA[arr_idx][data_idx:data_idx + 1])
                cur_dataidx += 1
            if cur_dataidx >= len(dataidx_list):
                break
        if cur_dataidx >= len(dataidx_list):
            break
    return subset_dataA, subset_dataB


def get_train_val_data(datadir,
                       data_config,
                       datasplit_type: DataSplitType,
                       get_two_channel_files_fn,
                       val_fraction=None,
                       test_fraction=None):
    dset_subtype = data_config.subdset_type

    if dset_subtype == SubDsetType.TwoChannel:
        fnamessox, fnamesgolgi = get_two_channel_files_fn()
    else:
        raise NotImplementedError('Only TwoChannel is implemented')

    fpathssox = [os.path.join(datadir, x) for x in fnamessox]
    fpathsgolgi = [os.path.join(datadir, x) for x in fnamesgolgi]
    dataA = [load_tiff(fpath) for fpath in fpathssox]
    dataB = [load_tiff(fpath) for fpath in fpathsgolgi]
    assert len(dataA) == len(dataB)
    for i in range(len(dataA)):
        assert dataA[i].shape == dataB[
            i].shape, f'{dataA[i].shape} != {dataB[i].shape}, {fpathssox[i]} != {fpathsgolgi[i]} in shape'

    count = np.sum([x.shape[0] for x in dataA])
    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, count)

    if datasplit_type == DataSplitType.All:
        pass
    elif datasplit_type == DataSplitType.Train:
        print(train_idx)
        dataA, dataB = subset_data(dataA, dataB, train_idx)
    elif datasplit_type == DataSplitType.Val:
        print(val_idx)
        dataA, dataB = subset_data(dataA, dataB, val_idx)
    elif datasplit_type == DataSplitType.Test:
        print(test_idx)
        dataA, dataB = subset_data(dataA, dataB, test_idx)
    else:
        raise Exception("invalid datasplit")

    data = TwoChannelData(dataA, dataB)
    print('Loaded from', SubDsetType.name(dset_subtype), datadir, len(data))
    return data
