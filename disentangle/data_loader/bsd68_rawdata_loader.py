import os

import numpy as np

from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples
from disentangle.core.tiff_reader import load_tiff


def get_train_val_data(dirname, data_config, datasplit_type, val_fraction, test_fraction):
    print(f'Loading from {dirname} , Mode:{DataSplitType.name(datasplit_type)}')

    if datasplit_type == DataSplitType.All:
        raise NotImplementedError('All data not implemented for this dataset')

    if datasplit_type == DataSplitType.Train:
        fpath = os.path.join(dirname, 'train', 'DCNN400_train_gaussian25.npy')
        data = np.load(fpath).astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        fpath = os.path.join(dirname, 'val', 'DCNN400_validation_gaussian25.npy')
        data = np.load(fpath).astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        fpath = os.path.join(dirname, 'test', 'bsd68_gaussian25.npy')
        data = np.load(fpath, allow_pickle=True)
        data = np.array([x for x in data]).astype(np.float32)

    data = np.tile(data[..., None], (1, 1, 1, 2))
    return data
