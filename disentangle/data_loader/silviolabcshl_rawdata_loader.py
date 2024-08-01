import os

import numpy as np

from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples
from disentangle.core.tiff_reader import load_tiff


def get_dataset_types():
    return [
        'ctrl_GM130_s1234_o1_c1111',
        'ctrl_GM130_s12_o1_c1010',
        'ctrl_beads_o1_c1111',
        'ctrl_calnexin_s1234_o1_c1111',
        'ctrl_s2_o1_c0010',
        'ctrl_s4_o1_c0001',
        'ctrl_GM130_s1234_o2_c1111',
        'ctrl_LAMP1_s1234_o1_c1111',
        'ctrl_beads_o2_c1111',
        'ctrl_calnexin_s1234_o2_c1111',
        'ctrl_s3_o1_c0100',
    ]


def get_filenames_gm130(ch_idx):
    """
    /group/jug/ashesh/data/svilen_cshl2024/ctrl_GM130_s12_o1_c1010/GM130_NHS_1_500/TIF_imp/20231013_GM130_NHS568_s12_o1_c4_001.msr.tif
    20231013_GM130_NHS568_s12_o1_c1_001.msr.tif 
    """
    fnames = []
    for i in range(1, 6):
        fname = f"20231013_GM130_NHS568_s12_o1_c{ch_idx}_{i:03d}.msr.tif"
        fnames.append(fname)
    return fnames


def get_filenames_lamp1(ch_idx):
    """
    20232010_LAMP1_s1234_o1_c3_001.msr.tif
    """
    fnames = []
    for i in range(1, 11):
        fname = f"20232010_LAMP1_s1234_o1_c{ch_idx}_{i:03d}.msr.tif"
        fnames.append(fname)
    return fnames


def load_data(dir, sub_data_type, ch_idx_list):
    data = []
    assert sub_data_type in get_dataset_types(), f"sub_data_type {sub_data_type} not in {get_dataset_types()}"
    for ch_idx in ch_idx_list:
        if sub_data_type == 'ctrl_GM130_s12_o1_c1010':
            fnames = get_filenames_gm130(ch_idx)
            subdir = os.path.join(sub_data_type, 'GM130_NHS_1_500')
        elif sub_data_type == 'ctrl_LAMP1_s1234_o1_c1111':
            subdir = sub_data_type
            fnames = get_filenames_lamp1(ch_idx)
        print(f"Loading idx:{ch_idx} from {dir}/{subdir}")
        fpaths = [os.path.join(dir, subdir, 'TIF_imp', fname) for fname in fnames]
        data.append(np.concatenate([load_tiff(fpath)[None, ..., None] for fpath in fpaths], axis=0))
    data = np.concatenate(data, axis=3)
    return data


def get_train_val_data(dirname, data_config, datasplit_type, val_fraction, test_fraction):
    # actin-60x-noise2-highsnr.tif  mito-60x-noise2-highsnr.tif
    data = load_data(dirname, data_config.sub_data_type, data_config.channel_idx_list)
    if datasplit_type == DataSplitType.All:
        return data.astype(np.float32)

    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(data), starting_test=True)
    if datasplit_type == DataSplitType.Train:
        return data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        return data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        return data[test_idx].astype(np.float32)


if __name__ == '__main__':
    direc = '/group/jug/ashesh/data/svilen_cshl2024/'
    data_type = 'ctrl_GM130_s12_o1_c1010'
    data = load_data(direc, data_type, [1, 3])
    print(data.shape)
