from typing import Union

import numpy as np
from disentangle.core import data_split_type

from disentangle.core.tiff_reader import load_tiff
from disentangle.core.data_type import DataType
from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples


def train_val_data(fpath, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    print(f'Loading {fpath} with Channels {data_config.channel_1},{data_config.channel_2},'
          f'datasplit mode:{DataSplitType.name(datasplit_type)}')
    data = load_tiff(fpath)
    
    starting_test = data_config.get('contiguous_splitting', False)
    if 'depth3D' in data_config and data_config.depth3D > 1:
        assert starting_test == True, "we need contiguous splitting of data for train/val/test when 3D is enabled"
    
    if data_config.data_type == DataType.Prevedel_EMBL:
        # Ensure that the last dimension is the channel dimension.
        data = data[..., None]
        data = np.swapaxes(data, 1, 4)
        data = data.squeeze()

    return _train_val_data(data,
                           datasplit_type,
                           data_config.channel_1,
                           data_config.channel_2,
                           val_fraction=val_fraction,
                           test_fraction=test_fraction,
                           starting_test=starting_test)


def _train_val_data(data, datasplit_type: DataSplitType, channel_1, channel_2, val_fraction=None, test_fraction=None, starting_test=False):
    assert data.shape[-1] > max(channel_1, channel_2), 'Invalid channels'
    data = data[..., [channel_1, channel_2]]
    if datasplit_type == DataSplitType.All:
        return data.astype(np.float32)

    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(data), starting_test=starting_test)
    if datasplit_type == DataSplitType.Train:
        return data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        return data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        return data[test_idx].astype(np.float32)
    else:
        raise Exception("invalid datasplit")