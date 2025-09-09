import os

import numpy as np

import nd2
from disentangle.core.custom_enum import Enum
from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples
from nis2pyr.reader import read_nd2file


class NikolaChannelList(Enum):
    Ch_A = 0
    Ch_B = 1
    Ch_C = 2
    Ch_D = 3
    ChBleedthrough_A = 4
    ChBleedthrough_B = 5
    ChBleedthrough_C = 6
    ChBleedthrough_D = 7
    Ch_AB = 8   
    Ch_AC = 9   
    Ch_AD = 10   
    Ch_BC = 11   
    Ch_BD = 12   
    Ch_CD = 13   
    Ch_ABC = 14  
    Ch_ABD = 15  
    Ch_ACD = 16  
    Ch_BCD = 17 
    Ch_ABCD = 18 

def get_ms_based_datafiles(ms:str):
    return [f'Set{i}/uSplit_{ms}.nd2' for i in range(1, 7)]

def get_raw_files_dict():
    files_dict = {
        'high':[
            'uSplit_14022025_highSNR.nd2',
            'uSplit_20022025_highSNR.nd2',
            'uSplit_20022025_001_highSNR.nd2',],
        'mid':[
            'uSplit_14022025_midSNR.nd2',
            'uSplit_20022025_midSNR.nd2',
            'uSplit_20022025_001_midSNR.nd2',],
        'low':[
            
            'uSplit_14022025_lowSNR.nd2',
            'uSplit_20022025_lowSNR.nd2',
            'uSplit_20022025_001_lowSNR.nd2',],
        'verylow':[
            
            'uSplit_14022025_verylowSNR.nd2',
            'uSplit_20022025_verylowSNR.nd2',
            'uSplit_20022025_001_verylowSNR.nd2',
            ],

        '2ms': get_ms_based_datafiles('2ms'),
        '3ms': get_ms_based_datafiles('3ms'),
        '5ms': get_ms_based_datafiles('5ms'),
        '20ms': get_ms_based_datafiles('20ms'),
        '500ms': get_ms_based_datafiles('500ms'),
        }
    # check that the order is correct
    keys = ['high', 'mid', 'low', 'verylow']
    for key1 in keys:
        filetokens1 = list(map (lambda x: x.replace(key1, ''), files_dict[key1]))
        for key2 in keys:
            filetokens2 = list(map (lambda x: x.replace(key2, ''), files_dict[key2]))
            assert np.array_equal(filetokens1, filetokens2), f'File tokens are not equal for {key1} and {key2}'
    return files_dict


def load_7D(fpath):    
    print(f'Loading from {fpath}')
    with nd2.ND2File(fpath) as nd2file:
        # Stdout: ND2 dimensions: {'P': 20, 'C': 19, 'Y': 1608, 'X': 1608}; RGB: False; datatype: uint16; legacy: False
        data = read_nd2file(nd2file)
    return data

def load_one_fpath(fpath, channel_list):
    data = load_7D(fpath)    
    # data.shape: (1, 20, 1, 19, 1608, 1608, 1) 
    data = data[0, :, 0, :, :, :, 0]
    # data.shape: (20, 19, 1608, 1608)
    # Here, 20 are different locations and 19 are different channels.
    data = data[:, channel_list,...]
    # swap the second and fourth axis
    data = np.swapaxes(data[...,None], 1, 4)[:,0]
    
    fname_prefix = '_'.join(os.path.basename(fpath).split('.')[0].split('_')[:-1])
    if fname_prefix == 'uSplit_20022025_001':
        data = np.delete(data, 2, axis=0)
    elif fname_prefix == 'uSplit_14022025':
        data = np.delete(data, [17, 19], axis=0)

    # data.shape: (20, 1608, 1608, C)
    return data

def load_data(datadir, dset_type, channel_list):
    files_dict = get_raw_files_dict()[dset_type]
    data_list = []
    for fname in files_dict:
        fpath = os.path.join(datadir, fname)
        data = load_one_fpath(fpath, channel_list)
        data_list.append(data)
    if len(data_list) > 1:
        data = np.concatenate(data_list, axis=0)
    else:
        data = data_list[0]
    return data

def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    if 'dset_type_per_channel' not in data_config or data_config.dset_type_per_channel is None:
        return _get_train_val_data_one_dsettype(datadir, data_config, datasplit_type, val_fraction=val_fraction, test_fraction=test_fraction)
    else:
        import ml_collections
        new_config = ml_collections.ConfigDict(data_config)
        with new_config.unlocked():
            del new_config.dset_type_per_channel
        data_list = {}
        for dset_type in data_config.dset_type_per_channel:
            if dset_type in data_list:
                continue
            new_config.dset_type = dset_type
            data_list[dset_type] = _get_train_val_data_one_dsettype(datadir, new_config, datasplit_type, val_fraction=val_fraction, test_fraction=test_fraction)
        data = []
        for ch_idx in range(len(data_config.channel_idx_list)):
            dset_type = data_config.dset_type_per_channel[ch_idx]
            data.append(data_list[dset_type][..., ch_idx:ch_idx+1])
        data = np.concatenate(data, axis=-1)
        return data
    
def _get_train_val_data_one_dsettype(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    dset_type = data_config.dset_type
    data = load_data(datadir, dset_type, data_config.channel_idx_list)
    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(data))
    if datasplit_type == DataSplitType.All:
        data = data.astype(np.float32)
    elif datasplit_type == DataSplitType.Train:
        data = data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        data = data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        data = data[test_idx].astype(np.float32)
    else:
        raise Exception("invalid datasplit")

    return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from disentangle.configs.nikola_denoisplit import get_config

    config = get_config()
    config.data.enable_gaussian_noise = False
    config.data.channel_idx_list = [NikolaChannelList.Ch_A, NikolaChannelList.Ch_B,  NikolaChannelList.Ch_AB]
    config.data.num_channels = len(config.data.channel_idx_list)
    # config.data.dset_type_per_channel = ['5ms', '5ms', '500ms']

    datadir = '/group/jug/ashesh/data/nikola_data/20240531/'
    data = get_train_val_data(datadir, config.data, DataSplitType.Train,
                              config.training.val_fraction, config.training.test_fraction)
    _,ax = plt.subplots(figsize=(18,6),ncols=3)
    ax[0].imshow(data[0,...,0])
    ax[1].imshow(data[0,...,1])
    ax[2].imshow(data[0,...,2])
    # 'high', [0, 1]