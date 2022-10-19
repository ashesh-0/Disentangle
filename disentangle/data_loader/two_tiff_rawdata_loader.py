import os
from disentangle.core.tiff_reader import load_tiff
import numpy as np


def get_train_val_data(dirname, data_config, is_train, val_fraction):
    # actin-60x-noise2-highsnr.tif  mito-60x-noise2-highsnr.tif
    fpath1 = os.path.join(dirname, data_config.ch1_fname)
    fpath2 = os.path.join(dirname, data_config.ch2_fname)

    print(f'Loading from {dirname} Channel1: {fpath1},{fpath2}, is_train:{is_train}')

    data1 = load_tiff(fpath1)[..., None]
    data2 = load_tiff(fpath2)[..., None]

    data = np.concatenate([data1, data2], axis=3)

    if is_train is None:
        return data.astype(np.float32)

    val_start = int((1 - val_fraction) * len(data))
    if is_train:
        return data[:val_start].astype(np.float32)
    else:
        return data[val_start:].astype(np.float32)
