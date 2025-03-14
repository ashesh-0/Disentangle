import os

import numpy as np

from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples
from disentangle.core.tiff_reader import load_tiff
from disentangle.data_loader.read_mrc import read_mrc


def get_mrc_data(fpath):
    # HXWXN
    _, data = read_mrc(fpath)
    data = data[None]
    data = np.swapaxes(data, 0, 3)
    return data[..., 0]


def get_train_val_data(dirname, data_config, datasplit_type, val_fraction, test_fraction):
    # actin-60x-noise2-highsnr.tif  mito-60x-noise2-highsnr.tif
    num_channels = data_config.get('num_channels', 2)
    fpaths = []
    data_list = []
    for i in range(num_channels):
        fpath1 = os.path.join(dirname, data_config.get(f'ch{i + 1}_fname'))
        fpaths.append(fpath1)
        data = get_mrc_data(fpath1)[..., None]
        data_list.append(data)

    dirname = os.path.dirname(os.path.dirname(fpaths[0])) + '/'

    msg = ','.join([x[len(dirname):] for x in fpaths])
    print(f'Loaded from {dirname} Channels:{len(fpaths)} {msg} Mode:{DataSplitType.name(datasplit_type)}')
    N = data_list[0].shape[0]
    for data in data_list:
        N = min(N, data.shape[0])

    cropped_data = []
    for data in data_list:
        cropped_data.append(data[:N])

    data = np.concatenate(cropped_data, axis=3)

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
    from ml_collections.config_dict import ConfigDict
    data_config = ConfigDict()
    data_config.num_channels = 2
    data_config.ch1_fname = 'Microtubules/GT_all.mrc'#'CCPs/GT_all.mrc'
    data_config.ch2_fname = 'ER/GT_all.mrc'
    # data_config.ch3_fname = 'Microtubules/GT_all.mrc'
    datadir = '/group/jug/ashesh/data/BioSR/'
    datasplit_type = DataSplitType.Test
    data = get_train_val_data(datadir, data_config, datasplit_type, val_fraction=0.1, test_fraction=0.1)
    print(data.shape)

    import os

    import numpy as np

    from disentangle.core.tiff_reader import save_tiff
    ch1 = os.path.dirname(data_config.ch1_fname)
    ch2 = os.path.dirname(data_config.ch2_fname)
    if datasplit_type == DataSplitType.Train:
        split_str = 'train'
    elif datasplit_type ==DataSplitType.Val:
        split_str = 'val'
    elif datasplit_type == DataSplitType.Test:
        split_str = 'test'
    else:
        raise ValueError('Unknown split type')
    
    path = f'/group/jug/ashesh/data/diffsplit_BioSR/{ch1}-vs-{ch2}/{split_str}/{split_str}.tif'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(path)
    save_tiff(path,data)
    print(path, data.shape)