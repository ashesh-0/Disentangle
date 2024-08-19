import os
from ast import literal_eval as make_tuple
from collections.abc import Sequence
from random import shuffle
from typing import List

import numpy as np

from czifile import imread as imread_czi
from disentangle.core.custom_enum import Enum
from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples
from disentangle.core.data_type import DataType
from disentangle.core.tiff_reader import load_tiff
from disentangle.data_loader.multifile_raw_dloader import MultiChannelData, SubDsetType


def get_multi_channel_files_v1():
    return {DataSplitType.Train:[
        'Experiment-155.czi', #722,18 => very good.
        'Experiment-156.czi', #722,18 => good.
        'Experiment-164.czi', #521, => good.
        'Experiment-162.czi', #362,9 => good
    ], DataSplitType.Val:[
        'Experiment-163.czi', #362,9 => good. it just has 3 bright spots which should be removed.
    ], DataSplitType.Test:[
        'Experiment-165.czi', #361,9 => good.
    ]
        # 'Experiment-140.czi', #400 => issue.
        # 'Experiment-160.czi', #521 => shift in between.
        # 'Experiment-157.czi', #561 => okay. the problem is a shift in between. This could be a good candidate for test val split.
        # 'Experiment-159.czi', #161,4 => does not make sense to use this.
        # 'Experiment-166.czi', #201
    }
def get_multi_channel_files_v2():
    return [
        'Experiment-447.czi',
        'Experiment-449.czi',
        'Experiment-448.czi',
        # 'Experiment-452.czi'
    ]


def get_multi_channel_files_v3():
    return [
        'Experiment-493.czi',
        'Experiment-494.czi',
        'Experiment-495.czi',
        'Experiment-496.czi',
        'Experiment-497.czi',
    ]


def load_data(fpath):
    # (4, 1, 4, 22, 512, 512, 1)
    data = imread_czi(fpath)
    clean_data = data[3, :, [0, 2], ..., 0]
    clean_data = np.swapaxes(clean_data[..., None], 0, 5)[0]
    return clean_data


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    if data_config.data_type == DataType.ExpMicroscopyV1:  
        fnames = get_multi_channel_files_v1()[datasplit_type]
    elif data_config.data_type == DataType.ExpMicroscopyV2:
        fnames = get_multi_channel_files_v2()
        assert len(fnames) == 3
        if datasplit_type == DataSplitType.Train:
            fnames = fnames[:-1]
        elif datasplit_type in [DataSplitType.Val, DataSplitType.Test]:
            fnames = fnames[-1:]
      
    fpaths = [os.path.join(datadir, fname) for fname in fnames]
    data = [load_data(fpath) for fpath in fpaths]
    if datasplit_type in [DataSplitType.Val, DataSplitType.Test] and data_config.data_type == DataType.ExpMicroscopyV2:
        assert len(data) == 1
        zN = data[0].shape[1]
        if datasplit_type == DataSplitType.Val:
            data[0] = data[0][:,:zN//2]
        elif datasplit_type == DataSplitType.Test:
            data[0] = data[0][:,zN//2:]
        
    data = MultiChannelData(data, paths=fpaths)
    return data
    

if __name__ == '__main__':
    from disentangle.data_loader.multifile_raw_dloader import SubDsetType
    from ml_collections.config_dict import ConfigDict
    data_config = ConfigDict()
    # data_config.data_type = DataType.ExpMicroscopyV2
    # datadir = '/group/jug/ashesh/data/expansion_microscopy_v2/datafiles'

    data_config.data_type = DataType.ExpMicroscopyV1
    datadir = '/group/jug/ashesh/data/expansion_microscopy_v1/MDCK_MitoDeepRed639_AlphaBetaTub488/'

    data = get_train_val_data(datadir, data_config, DataSplitType.Train, val_fraction=0.1, test_fraction=0.1)
    print(len(data))
    for i in range(len(data)):
        print(i, data[i][0].shape)
