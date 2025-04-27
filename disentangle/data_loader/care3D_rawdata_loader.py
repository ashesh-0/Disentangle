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
    # return ['farred_RFP_GFP_2109175_small_for_debugging.tif']
    return ['farred_RFP_GFP_2109175.tif']

def liver_fnames():
    return ['channel_234_1.tif']

def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    datadir = os.path.join(datadir, data_config.subdset_type)
    if data_config.subdset_type == 'zebrafish':
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
        
        print('Loading zebrafish data:', datadir, fnames)
        fpaths = [os.path.join(datadir, x) for x in fnames]
        data = [(load_tiff(x),x) for x in fpaths]
        return data
    
    elif data_config.subdset_type == 'liver':
        fnames = liver_fnames()
        print('Loading liver data:', datadir, fnames)
        data = [load_tiff(os.path.join(datadir, x)) for x in fnames]
        assert len(data) == 1
        # NC=3 x Z x H x W
        data = data[0]
        test_start_idx = 0 
        test_end_idx = int(test_fraction*data.shape[1])
        val_start_idx = test_end_idx
        val_end_idx = val_start_idx + int(val_fraction*data.shape[1])
        train_start_idx = val_end_idx
        train_end_idx = data.shape[1]
        if datasplit_type == DataSplitType.Train:
            data = data[:,train_start_idx:train_end_idx]
        elif datasplit_type == DataSplitType.Val:
            data = data[:,val_start_idx:val_end_idx]
        elif datasplit_type == DataSplitType.Test:
            data = data[:,test_start_idx:test_end_idx]
        # transpose to Z * H * W * NC
        data = data.transpose(1,2,3,0)
        return [(data,fnames[0])]
    elif data_config.subdset_type =='zebrafish_similarity':
        ch1_factor = data_config.ch1_factor
        ch2_factor = data_config.ch2_factor
        if datasplit_type in [DataSplitType.Train, DataSplitType.Val]:
            fpath = os.path.join(datadir, f'F1-{ch1_factor}_F2-{ch2_factor}-farred_RFP_GFP_2109171.tif')
            assert os.path.exists(fpath), f"File not found: {fpath}"
            data = load_tiff(fpath)
            # split into train and val
            if datasplit_type == DataSplitType.Val:
                data = data[:int(data.shape[0]*val_fraction)]
            else:
                data = data[int(data.shape[0]*val_fraction):]
        elif datasplit_type == DataSplitType.Test:
            fpath = os.path.join(datadir, f'F1-{ch1_factor}_F2-{ch2_factor}-farred_RFP_GFP_2109175.tif')
            assert os.path.exists(fpath), f"File not found: {fpath}"
            data = load_tiff(fpath)
        else:
            raise Exception("invalid datasplit")
        
        return [(data, fpath)]

        
    
if __name__ == '__main__':
    import ml_collections as ml
    datadir = '/group/jug/ashesh/data/CARE/care_florian/'
    data_config = ml.ConfigDict({
        'subdset_type': 'liver',
    })
    datasplit_type = DataSplitType.Val
    data = get_train_val_data(datadir, data_config, datasplit_type, val_fraction=0.1, test_fraction=0.1)
    print(data[0][0].shape)