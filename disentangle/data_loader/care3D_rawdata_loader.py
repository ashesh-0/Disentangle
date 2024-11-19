import os

import numpy as np

import nd2
from disentangle.core.data_split_type import DataSplitType
from disentangle.core.tiff_reader import load_tiff


def zebrafish_train_fnames():
    return ['farred_RFP_GFP_2109171.tif',
            'farred_RFP_GFP_2109172.tif',
            'farred_RFP_GFP_21091710.tif',  
            'farred_RFP_GFP_21091711.tif',  
            'farred_RFP_GFP_21091712.tif',]

def zebrafish_val_fnames():
    return ['farred_RFP_GFP_2109174.tif']

def zebrafish_test_fnames():
    return ['farred_RFP_GFP_2109175.tif']



def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    assert data_config.subdset_type == 'zebrafish'
    datadir = os.path.join(datadir, data_config.subdset_type)
    if datasplit_type == DataSplitType.All:
        raise Exception("All not supported")
    elif datasplit_type == DataSplitType.Train:
        fnames = zebrafish_train_fnames()
    elif datasplit_type == DataSplitType.Val:
        fnames = zebrafish_val_fnames()
    elif datasplit_type == DataSplitType.Test:
        fnames = zebrafish_test_fnames()
    else:
        raise Exception("invalid datasplit")
    
    fpaths = [os.path.join(datadir, x) for x in fnames]
    data = [(load_tiff(x),x) for x in fpaths]
    return data


if __name__ == '__main__':
    import ml_collections as ml
    datadir = '/group/jug/ashesh/data/CARE/care_florian/'
    data_config = ml.ConfigDict({
        'subdset_type': 'zebrafish',
    })
    datasplit_type = DataSplitType.Val
    data = get_train_val_data(datadir, data_config, datasplit_type, val_fraction=0.1, test_fraction=0.1)
    print(data[0].shape)