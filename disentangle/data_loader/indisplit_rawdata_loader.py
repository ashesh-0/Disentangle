import os

from disentangle.core.data_split_type import DataSplitType
from disentangle.core.data_type import DataType
from disentangle.core.tiff_reader import load_tiff


def get_train_val_data(datadir,
            data_config,
            datasplit_type,
            val_fraction=None,
            test_fraction=None):
    
    if data_config.data_type in [DataType.indiSplit_BioSR, DataType.indiSplit_HTT24, DataType.indiSplit_HTLIF24, DataType.indiSplit_PaviaATN, DataType.indiSplit_HagenEtAl]:
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
