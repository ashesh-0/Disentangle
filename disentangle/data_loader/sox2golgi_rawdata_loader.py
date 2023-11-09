import os
from ast import literal_eval as make_tuple
from collections.abc import Sequence
from random import shuffle
from typing import List

import numpy as np

from disentangle.core.custom_enum import Enum
from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples
from disentangle.core.tiff_reader import load_tiff
from disentangle.data_loader.multifile_raw_dloader import get_train_val_data as get_train_val_data_twofiles


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


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    return get_train_val_data_twofiles(datadir,
                                       data_config,
                                       datasplit_type,
                                       get_two_channel_files,
                                       val_fraction=val_fraction,
                                       test_fraction=test_fraction)


if __name__ == '__main__':
    from disentangle.data_loader.multifile_raw_dloader import SubDsetType
    from ml_collections.config_dict import ConfigDict
    data_config = ConfigDict()
    data_config.subdset_type = SubDsetType.TwoChannel
    datadir = '/group/jug/ashesh/data/Taverna/'
    data = get_train_val_data(datadir, data_config, DataSplitType.Test, val_fraction=0.1, test_fraction=0.1)
    print(len(data))
    # for i in range(len(data)):
    # print(i, data[i].shape)
