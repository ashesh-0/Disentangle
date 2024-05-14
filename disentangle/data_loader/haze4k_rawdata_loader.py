import os

import numpy as np

import imageio.v3 as iio
from disentangle.core.custom_enum import Enum
from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples
from disentangle.core.tiff_reader import load_tiff
from disentangle.data_loader.multifile_raw_dloader import SubDsetType
from disentangle.data_loader.multifile_raw_dloader import get_train_val_data as get_train_val_data_twochannels


def get_multi_channel_files():
    return [f'data_{i}.tif' for i in range(1, 1801)]


def load_tiff_last_ch(fpath):
    data = load_tiff(fpath)
    return data.transpose(1, 2, 0)[None]


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    assert data_config.subdset_type == SubDsetType.MultiChannel
    if data_config.get('eval_on_real', False):
        files_fn = get_multi_channel_practical_files
    else:
        files_fn = get_multi_channel_files

    return get_train_val_data_twochannels(datadir,
                                          data_config,
                                          datasplit_type,
                                          files_fn,
                                          load_data_fn=load_tiff_last_ch,
                                          val_fraction=val_fraction,
                                          test_fraction=test_fraction)


if __name__ == '__main__':
    direc = '/group/jug/ashesh/data/Haze4K/test'
    idx = 10
    cleanf = os.path.join(direc, f'gt/{idx}.png')
    hazef = glob.glob(os.path.join(direc, f'haze/{idx}*.png'))
    transf = os.path.join(direc, f'trans/{idx}.png')

    clean = load_png(cleanf)
    haze = load_png(hazef[0])
    trans = load_png(transf)
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(9, 3), ncols=3)
    ax[0].imshow(clean)
    ax[1].imshow(haze)
    ax[2].imshow(trans)
