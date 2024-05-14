"""
Here, we pre-process the Rain100H dataset to create a series of tiff files. 
"""
import argparse
import os

import numpy as np
from tqdm import tqdm

import imageio.v3 as iio
from disentangle.core.tiff_reader import load_tiff, save_tiff


def load_png(fpath):
    im = iio.imread(fpath)
    return im


def fnames(subdir):
    assert subdir in ['norain', 'rain', 'rainregion', 'rainstreak']
    # return [f"{subdir}/{subdir}-{i:03d}.png" for i in range(1, 101)]
    return [f"{subdir}/{subdir}-{i}.png" for i in range(1, 1801)]


def load_data(datadir, outputdatadir):
    print(f'Loading Data from', datadir)

    rainfiles = fnames('rain')
    norainfiles = fnames('norain')
    rainregionfiles = fnames('rainregion')
    rainstreakfiles = fnames('rainstreak')

    rain = [load_png(os.path.join(datadir, fname)) for fname in rainfiles]
    norain = [load_png(os.path.join(datadir, fname)) for fname in norainfiles]
    rainregion = [load_png(os.path.join(datadir, fname))[..., None] for fname in rainregionfiles]
    rainstreak = [load_png(os.path.join(datadir, fname))[..., None] for fname in rainstreakfiles]

    print(f'Saving Data to', outputdatadir)
    for i in tqdm(range(len(rain))):
        file_idx = int(rainfiles[i].replace('.png', '').split('-')[-1])
        fname = f'data_{file_idx}.tif'
        fpath = os.path.join(outputdatadir, fname)
        data = np.concatenate([rain[i], norain[i], rainregion[i], rainstreak[i]], axis=2).astype(np.int32)
        data = data.transpose(2, 0, 1)
        save_tiff(fpath, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process Rain100H dataset')
    parser.add_argument('--datadir', type=str, help='Directory containing the Rain100H dataset')
    parser.add_argument('--outputdatadir', type=str, help='Directory to save the pre-processed dataset')
    args = parser.parse_args()

    load_data(args.datadir, args.outputdatadir)
