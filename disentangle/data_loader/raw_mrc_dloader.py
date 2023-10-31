import os

import numpy as np

from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples
from disentangle.core.tiff_reader import load_tiff
from disentangle.data_loader.read_mrc import read_mrc


def get_mrc_data(fpath):
    # HXWXN
    _, data = read_mrc(fpath)
    data = data[None]
    data = np.swapaxes(data, 0, 3)
    return data[..., 0]


def get_train_val_data(dirname, data_config, datasplit_type, val_fraction, test_fraction):
    # actin-60x-noise2-highsnr.tif  mito-60x-noise2-highsnr.tif
    fpath1 = os.path.join(dirname, data_config.ch1_fname)
    fpath2 = os.path.join(dirname, data_config.ch2_fname)

    print(f'Loading from {dirname} Channel1: '
          f'{fpath1},{fpath2}, Mode:{DataSplitType.name(datasplit_type)}')

    data1 = get_mrc_data(fpath1)[..., None]
    data2 = get_mrc_data(fpath2)[..., None]
    # assert abs(
    #     data1.shape[0] - data2.shape[0]
    # ) < 2, "Data shape mismatch by more than 1 N. this needs an alternate immplementation where both channels are loaded\
    # separately."

    N = min(data1.shape[0], data2.shape[0])
    data1 = data1[:N]
    data2 = data2[:N]

    data = np.concatenate([data1, data2], axis=3)

    if datasplit_type == DataSplitType.All:
        return data.astype(np.float32)

    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(data), starting_test=True)
    if datasplit_type == DataSplitType.Train:
        return data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        return data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        return data[test_idx].astype(np.float32)
