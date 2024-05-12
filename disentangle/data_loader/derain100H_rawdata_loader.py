import os

import numpy as np

import imageio.v3 as iio
from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples


def load_png(fpath):
    im = iio.imread(fpath)
    return im


def fnames(subdir):
    assert subdir in ['norain', 'rain', 'rainregion', 'rainstreak']
    return [f"{subdir}/{subdir}-{i}.png" for i in range(1, 1801)]


def load_data(datadir, rainfiles, norainfiles, rainregionfiles, rainstreakfiles):
    print(f'Loading Data from', datadir)
    rainstack = [load_png(os.path.join(datadir, fname)) for fname in rainfiles]
    rain = np.stack()
    norain = np.stack([load_png(os.path.join(datadir, fname)) for fname in norainfiles])
    rainregion = np.stack([load_png(os.path.join(datadir, fname)) for fname in rainregionfiles])
    rainstreak = np.stack([load_png(os.path.join(datadir, fname)) for fname in rainstreakfiles])
    # N X H X W X 3 -> N x 3 x H x W
    rain = rain.transpose(0, 3, 1, 2)
    norain = norain.transpose(0, 3, 1, 2)
    rainregion = rainregion.transpose(0, 3, 1, 2)
    rainstreak = rainstreak.transpose(0, 3, 1, 2)
    # 1st 3 channels are for input.
    data = np.concatenate([rain, norain, rainregion, rainstreak], axis=1)
    return data


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    rainfiles = fnames('rain')
    norainfiles = fnames('norain')
    rainregionfiles = fnames('rainregion')
    rainstreakfiles = fnames('rainstreak')

    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(rainfiles))
    if datasplit_type == DataSplitType.All:
        data = load_data(datadir, rainfiles, norainfiles, rainregionfiles, rainstreakfiles)
    elif datasplit_type == DataSplitType.Train:
        data = load_data(datadir, rainfiles[train_idx], norainfiles[train_idx], rainregionfiles[train_idx],
                         rainstreakfiles[train_idx])
    elif datasplit_type == DataSplitType.Val:
        data = load_data(datadir, rainfiles[val_idx], norainfiles[val_idx], rainregionfiles[val_idx],
                         rainstreakfiles[val_idx])
    elif datasplit_type == DataSplitType.Test:
        data = load_data(datadir, rainfiles[test_idx], norainfiles[test_idx], rainregionfiles[test_idx],
                         rainstreakfiles[test_idx])
    else:
        raise Exception("invalid datasplit")

    data = data.astype(np.float32)
    return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from disentangle.configs.derain100H_config import get_config
    from disentangle.core.data_type import DataType
    from disentangle.core.loss_type import LossType
    from disentangle.core.model_type import ModelType
    from disentangle.core.sampler_type import SamplerType
    data_dir = '/group/jug/ashesh/data/RainTrainH/'
    config = get_config()
    data = get_train_val_data(data_dir, config.data, DataSplitType.All, val_fraction=0.1, test_fraction=0.1)
    _, ax = plt.subplots(12, 3, ncols=4)
    idx = 0
    ax[0].imshow(data[idx, :3].transpose(1, 2, 0))
    ax[1].imshow(data[idx, 3:6].transpose(1, 2, 0))
    ax[2].imshow(data[idx, 6:9].transpose(1, 2, 0))
    ax[3].imshow(data[idx, 9:].transpose(1, 2, 0))
