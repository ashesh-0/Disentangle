import os
from ast import literal_eval as make_tuple
from collections.abc import Sequence
from functools import partial
from random import shuffle
from typing import List

import numpy as np

from disentangle.core.custom_enum import Enum
from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples
from disentangle.core.data_type import DataType
from disentangle.core.tiff_reader import load_tiff
from disentangle.data_loader.multifile_raw_dloader import SubDsetType
from disentangle.data_loader.multifile_raw_dloader import get_train_val_data as get_train_val_data_twochannels


def get_multi_channel_files():
    return ['SIM1-49_merged.tif', 'SIM201-263_merged.tif']


def get_multi_channel_files_with_input(noise_level):
    if noise_level == 'low':
        return ['SIM_3color_1channel_group1_small.tif']
    elif noise_level == 'high':
        return ['SIM_3color_1channel_group2.tif']  # This is a different noise level.


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    assert data_config.subdset_type == SubDsetType.MultiChannel
    if data_config.data_type == DataType.Dao3Channel:
        return get_train_val_data_twochannels(datadir,
                                              data_config,
                                              datasplit_type,
                                              get_multi_channel_files,
                                              val_fraction=val_fraction,
                                              test_fraction=test_fraction)
    elif data_config.data_type == DataType.Dao3ChannelWithInput:
        return get_train_val_data_twochannels(datadir,
                                              data_config,
                                              datasplit_type,
                                              partial(get_multi_channel_files_with_input, data_config.noise_level),
                                              val_fraction=val_fraction,
                                              test_fraction=test_fraction)
    else:
        raise NotImplementedError(f"Data type {data_config.data_type} not implemented.")


if __name__ == '__main__':
    from disentangle.data_loader.multifile_raw_dloader import SubDsetType
    from ml_collections.config_dict import ConfigDict
    data_config = ConfigDict()
    data_config.subdset_type = SubDsetType.MultiChannel
    datadir = '/group/jug/ashesh/data/Dao3ChannelReduced/'
    data = get_train_val_data(datadir, data_config, DataSplitType.Train, val_fraction=0.1, test_fraction=0.1)
    print(len(data))
    for i in range(len(data)):
        print(i, data[i][0].shape)
