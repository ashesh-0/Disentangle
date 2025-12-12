"""
This handles a list of tiff files and loads them
"""

import os

import numpy as np

# import read_lif
from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples


def load_series(series, series_idx_list, channel_idx_list):
    print("Loading series:", len(series_idx_list), "from total:", len(series), "with channels:", channel_idx_list)
    data = []
    for sidx in series_idx_list:
        chosen = series[sidx]
        series_data = []
        for chidx in channel_idx_list:
            series_data.append(chosen.getFrame(T=0, channel=chidx, dtype=np.uint16)[..., None])
        series_data = np.concatenate(series_data, axis=-1)
        data.append(series_data)
    return data


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=0.1, test_fraction=0.1):
    fpath = os.path.join(datadir, data_config.data_fname)
    reader = read_lif.Reader(fpath)
    series = reader.getSeries()
    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(series))
    # breakpoint()
    if datasplit_type == DataSplitType.Train:
        data = load_series(series, train_idx, data_config.channel_idx_list)
        data = [(data[i], str(train_idx[i])) for i in range(len(data))]
    elif datasplit_type == DataSplitType.Val:
        data = load_series(series, val_idx, data_config.channel_idx_list)
        data = [(data[i], str(val_idx[i])) for i in range(len(data))]
    elif datasplit_type == DataSplitType.Test:
        data = load_series(series, test_idx, data_config.channel_idx_list)
        # data = [x[:6] for x in data]  # only take first 6 channels
        data = [(data[i], str(test_idx[i])) for i in range(len(data))]
    elif datasplit_type == DataSplitType.All:
        idx = np.arange(len(series))
        data = load_series(series, idx, data_config.channel_idx_list)
        data = [(data[i], str(idx[i])) for i in range(len(data))]
    else:
        raise Exception("invalid datasplit")

    print("Loaded:", fpath, "SeriesCount:", len(data), " one data shape", data[0][0].shape)
    return data


if __name__ == "__main__":
    import ml_collections

    data_config = ml_collections.ConfigDict()
    data_config.data_fname = "12_noon_female_con_liv4.lif"

    data_config.channel_idx_list = [1, 2, 4]
    data = get_train_val_data("/group/jug/ashesh/data/HHMI25_v2/", data_config, DataSplitType.Train)
    print(data.shape)
