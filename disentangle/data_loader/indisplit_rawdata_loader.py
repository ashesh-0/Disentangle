import os

import numpy as np

from disentangle.core.data_split_type import DataSplitType
from disentangle.core.data_type import DataType
from disentangle.core.tiff_reader import load_tiff


def get_train_val_data(datadir,
            data_config,
            datasplit_type,
            val_fraction=None,
            test_fraction=None):
    
    if data_config.data_type in [DataType.indiSplit_BioSR, DataType.indiSplit_HTT24, DataType.indiSplit_HTLIF24, DataType.indiSplit_PaviaATN]:
        if datasplit_type == DataSplitType.Train:
            datadir = os.path.join(datadir, 'train')
        elif datasplit_type == DataSplitType.Val:
            datadir = os.path.join(datadir, 'val')
        elif datasplit_type == DataSplitType.Test:
            datadir = os.path.join(datadir, 'test')
        else:
            raise ValueError(f"Unknown data split type: {datasplit_type}")

        fpaths = [os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith('.tif')]
        assert len(fpaths) == 1, f"Expected one file in {datadir}, but found {len(fpaths)}"
        data = load_tiff(fpaths[0])

        if 'keep_real_input' in data_config and data_config.keep_real_input:
            pass
        else:
            # skip the input channel
            data = data[...,:2]

        print(f'Loaded {DataType.name(data_config.data_type)} data from {fpaths[0]}', data.shape)
        return data
    elif data_config.data_type == DataType.indiSplit_HagenEtAl:
        fpath1 = 'train/train_actin-60x-noise2-highsnr.tif'
        fpath2 = 'train/train_mito-60x-noise2-highsnr.tif'
        if datasplit_type == DataSplitType.Val:
            fpath1 = fpath1.replace('train', 'val')
            fpath2 = fpath2.replace('train', 'val')
        elif datasplit_type == DataSplitType.Test:
            fpath1 = fpath1.replace('train', 'test')
            fpath2 = fpath2.replace('train', 'test')
        data1 = load_tiff(os.path.join(datadir, fpath1))
        data2 = load_tiff(os.path.join(datadir, fpath2))
        data = np.stack([data1, data2], axis=-1)
        print(f'Loaded {DataType.name(data_config.data_type)} data from {fpath1} and {fpath2}', data.shape)
        return data