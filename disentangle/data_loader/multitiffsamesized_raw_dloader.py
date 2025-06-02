"""
This handles a list of tiff files and loads them
"""
import os

import numpy as np

from disentangle.core.data_split_type import DataSplitType
from disentangle.core.tiff_reader import load_tiff


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType):
    if datasplit_type == DataSplitType.Train:
        fnames = data_config.train_fnames
    elif datasplit_type == DataSplitType.Val:
        fnames = data_config.val_fnames
    elif datasplit_type == DataSplitType.Test:
        fnames = data_config.test_fnames
    elif datasplit_type == DataSplitType.All:
        fnames = list(data_config.train_fnames) + list(data_config.val_fnames) + list(data_config.test_fnames)
    else:
        raise Exception("invalid datasplit")
    
    fpaths = [os.path.join(datadir, x) for x in fnames]
    # N x H x W x C
    data = [load_tiff(x) for x in fpaths]
    for x in data:
        assert len(x.shape) == 4, f"Expected 4D tiff data, got {x.shape} for {x}"
    # it could be that it just has one channel. in which case the easiest way is to just replicate it
    if data[0].shape[-1] == 1:
        data = [np.repeat(x, len(data_config.channel_idx_list), axis=-1) for x in data]
    else:    
        data = [x[...,data_config.channel_idx_list] for x in data]
    
    data = list(zip(data, fpaths))
    print('Loaded:', datadir, fnames, 'one data shape', data[0][0].shape)
    return data

if __name__ == '__main__':
    import ml_collections
    data_config = ml_collections.ConfigDict()
    data_config.train_fnames = ['Composite_region3_6am_CF_L3L2.tif', 
                                'Composite_region5_6am_L3L1.tif',
                                 'Composite_region2_6AM_CF_L3L3.tif',  
                                 'Composite_region4_6am_L3L2.tif']
    
    data_config.val_fnames = ['Composite_region1_CF_L3L3.tif']
    data_config.test_fnames = ['Composite_region7_CF_12noon_L3L4.tif']
    data_config.channel_idx_list = [0,1,2,3,4,5]
    data = get_train_val_data('/group/jug/ashesh/data/HHMI25/', data_config, DataSplitType.Train)
    print(data.shape)