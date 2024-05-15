import os
from functools import partial

import numpy as np

import imageio.v3 as iio
from disentangle.core.custom_enum import Enum
from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples
from disentangle.core.tiff_reader import load_tiff
from disentangle.data_loader.multifile_raw_dloader import SubDsetType
from disentangle.data_loader.multifile_raw_dloader import get_train_val_data as get_train_val_data_twochannels


def fname_schema():
    return 'data_{:d}.tif'


def get_multi_channel_files(max_idx):
    return [fname_schema().format(i) for i in range(1, max_idx + 1)]


def load_tiff_last_ch(fpath):
    data = load_tiff(fpath)
    return data.transpose(1, 2, 0)[None]


def find_max_idx(datadir):
    max_idx = 0
    while os.path.exists(os.path.join(datadir, fname_schema().format(max_idx + 1))):
        max_idx += 1
    return max_idx


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    assert data_config.subdset_type == SubDsetType.MultiChannel
    max_idx = find_max_idx(datadir)
    files_fn = partial(get_multi_channel_files, max_idx)
    # if datasplit_type in [DataSplitType.Train, DataSplitType.Val]:
    #     # datadir = os.path.join(datadir, 'train')
    # else:
    #     assert datasplit_type == DataSplitType.Test
    #     datadir = os.path.join(datadir, 'test')
    #     files_fn = get_multi_channel_files_test

    return get_train_val_data_twochannels(datadir,
                                          data_config,
                                          datasplit_type,
                                          files_fn,
                                          load_data_fn=load_tiff_last_ch,
                                          val_fraction=val_fraction,
                                          test_fraction=test_fraction)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from disentangle.configs.derain100H_config import get_config
    from disentangle.core.data_type import DataType
    from disentangle.core.loss_type import LossType
    from disentangle.core.model_type import ModelType
    from disentangle.core.sampler_type import SamplerType

    data_dir = '/group/jug/ashesh/data/Haze4KCombined/'
    config = get_config()
    data = get_train_val_data(data_dir, config.data, DataSplitType.All, val_fraction=0.1, test_fraction=0.1)
    _, ax = plt.subplots(figsize=(12, 3), ncols=4)
    idx = 0
    ax[0].imshow(data[idx, :3].transpose(1, 2, 0))
    ax[1].imshow(data[idx, 3:6].transpose(1, 2, 0))
    ax[2].imshow(data[idx, 6])
    ax[3].imshow(data[idx, 7])
