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


def get_two_channel_files():
    arr = [
        71,
        89,
        92,
        93,
        94,
        95,
    ]  #96,97,98,99,100, 1752, 1757, 1758, 1760, 1761]
    sox2 = [f'SOX2/C2-Experiment-{i}.tif' for i in arr]
    golgi = [f'GOLGI/C1-Experiment-{i}.tif' for i in arr]
    return sox2, golgi


# def get_shape_info(datadir, fpaths):
#     shapefpath = os.path.join(datadir, 'shape_info.txt')
#     if not os.path.exists(shapefpath):
#         print(f'Writing shape of each file to {shapefpath}')

#         with open(shapefpath, 'w') as f:
#             for fpath in fpaths:
#                 data = load_tiff(os.path.join(datadir,fpath))
#                 f.write(f'{fpath}:{data.shape}\n')

#     with open(shapefpath,'r') as f:
#         lines = f.readlines()
#     shape_info = {}
#     for line in lines:
#         line = line.strip()
#         if line == '':
#             continue
#         key, val = line.split(':')
#         shape_info[key] = make_tuple(val)

#     return [shape_info[fpath] for fpath in fpaths]


def subset_data(datasox, datagolgi, dataidx_list):
    dataidx_list = sorted(dataidx_list)
    subset_datagolgi = []
    subset_datasox = []
    cur_dataidx = 0
    cumulative_datacount = 0
    for arr_idx in range(len(datasox)):
        for data_idx in range(len(datasox[arr_idx])):
            cumulative_datacount += 1
            if dataidx_list[cur_dataidx] == cumulative_datacount - 1:
                subset_datagolgi.append(datagolgi[arr_idx][data_idx:data_idx + 1])
                subset_datasox.append(datasox[arr_idx][data_idx:data_idx + 1])
                cur_dataidx += 1
            if cur_dataidx >= len(dataidx_list):
                break
        if cur_dataidx >= len(dataidx_list):
            break
    return subset_datasox, subset_datagolgi


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    dset_subtype = data_config.subdset_type

    if dset_subtype == SubDsetType.TwoChannel:
        fnamessox, fnamesgolgi = get_two_channel_files()
    else:
        raise NotImplementedError('Only TwoChannel is implemented')

    fpathssox = [os.path.join(datadir, x) for x in fnamessox]
    fpathsgolgi = [os.path.join(datadir, x) for x in fnamesgolgi]
    datasox = [load_tiff(fpath) for fpath in fpathssox]
    datagolgi = [load_tiff(fpath) for fpath in fpathsgolgi]
    assert len(datasox) == len(datagolgi)
    for i in range(len(datasox)):
        assert datasox[i].shape == datagolgi[
            i].shape, f'{datasox[i].shape} != {datagolgi[i].shape}, {fpathssox[i]} != {fpathsgolgi[i]} in shape'

    count = np.sum([x.shape[0] for x in datasox])
    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, count)

    if datasplit_type == DataSplitType.All:
        pass
    elif datasplit_type == DataSplitType.Train:
        print(train_idx)
        datasox, datagolgi = subset_data(datasox, datagolgi, train_idx)
    elif datasplit_type == DataSplitType.Val:
        print(val_idx)
        datasox, datagolgi = subset_data(datasox, datagolgi, val_idx)
    elif datasplit_type == DataSplitType.Test:
        print(test_idx)
        datasox, datagolgi = subset_data(datasox, datagolgi, test_idx)
    else:
        raise Exception("invalid datasplit")

    data = TwoChannelData(datasox, datagolgi)
    print('Loaded from', SubDsetType.name(dset_subtype), datadir, len(data))
    return data


if __name__ == '__main__':
    from ml_collections.config_dict import ConfigDict
    data_config = ConfigDict()
    data_config.subdset_type = SubDsetType.TwoChannel
    datadir = '/group/jug/ashesh/data/Taverna/'
    data = get_train_val_data(datadir, data_config, DataSplitType.Test, val_fraction=0.1, test_fraction=0.1)
    print(len(data))
    # for i in range(len(data)):
    # print(i, data[i].shape)
