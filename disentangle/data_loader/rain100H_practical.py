"""
Here, we convert the images to a single tiff file. This can then be used to do a test prediction. 
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


def load_data(datadir, outputdatadir):
    print(f'Loading Data from', datadir)
    fnames = sorted(os.listdir(datadir))
    rain = [load_png(os.path.join(datadir, fname)) for fname in fnames]
    print(f'Saving Data to', outputdatadir)
    for i in tqdm(range(len(rain))):
        file_token = fnames[i].split('_')[-1].replace('.jpg', '')
        fname = f'data_{file_token}.tif'
        fpath = os.path.join(outputdatadir, fname)
        # this makes the shape identical to the training data, ie have 8 channels
        data = np.concatenate([rain[i], rain[i], rain[i][..., :2]], axis=2).astype(np.int32)
        data = data.transpose(2, 0, 1)
        save_tiff(fpath, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process Test images dataset')
    parser.add_argument('--datadir', type=str, help='Directory containing the Rain100H dataset')
    parser.add_argument('--outputdatadir', type=str, help='Directory to save the pre-processed dataset')
    args = parser.parse_args()

    load_data(args.datadir, args.outputdatadir)
