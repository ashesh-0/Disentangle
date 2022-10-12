from typing import Union

import numpy as np

from disentangle.core.tiff_reader import load_tiff
from disentangle.core.data_type import DataType


def train_val_data(fpath, data_config, is_train: Union[None, bool], val_fraction=None):
    print(f'Loading {fpath} with Channels {data_config.channel_1},{data_config.channel_2}, is_train:{is_train}')
    data = load_tiff(fpath)
    if data_config.data_type == DataType.Prevedel_EMBL:
        # Ensure that the last dimension is the channel dimension.
        data = data[..., None]
        data = np.swapaxes(data, 1, 4)
        data = data.squeeze()

    return _train_val_data(data, is_train, data_config.channel_1, data_config.channel_2, val_fraction=val_fraction)


def _train_val_data(data, is_train: Union[None, bool], channel_1, channel_2, val_fraction=None):
    assert data.shape[-1] > max(channel_1, channel_2), 'Invalid channels'
    data = data[..., [channel_1, channel_2]]
    if is_train is None:
        return data.astype(np.float32)

    val_start = int((1 - val_fraction) * len(data))
    if is_train:
        return data[:val_start].astype(np.float32)
    else:
        return data[val_start:].astype(np.float32)
