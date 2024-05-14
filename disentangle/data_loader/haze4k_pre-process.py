"""
Here, we pre-process the Rain100H dataset to create a series of tiff files. 
"""
import argparse
import os
from glob import glob

import numpy as np
from tqdm import tqdm

import imageio.v3 as iio
from disentangle.core.tiff_reader import load_tiff, save_tiff


def load_png(fpath):
    im = iio.imread(fpath)
    return im


def get_train_idx():
    return 3000


def get_test_idx():
    return 1000


def fnames(subdir, end_idx):
    assert subdir in ['gt', 'haze', 'trans']
    if subdir in ['gt', 'trans']:
        return [f'{subdir}/{i}.png' for i in range(1, end_idx)]
    else:
        return [f"{subdir}/{i}_*.png" for i in range(1, end_idx)]


def transform_data(datadir, outputdatadir, end_idx):
    os.makedirs(outputdatadir, exist_ok=True)
    print(f'Loading Data from', datadir)
    gtf = fnames('gt', end_idx)
    hazef = fnames('haze', end_idx)
    transf = fnames('trans', end_idx)

    gtfpaths = [os.path.join(datadir, fname) for fname in gtf]
    hazefpaths = [glob(os.path.join(datadir, fname))[0] for fname in hazef]
    transfpaths = [os.path.join(datadir, fname) for fname in transf]

    print(f'Saving Data to', outputdatadir)
    for i in tqdm(range(len(gtfpaths))):
        gt = load_png(gtfpaths[i])
        haze = load_png(hazefpaths[i])
        trans = load_png(transfpaths[i])
        data = np.concatenate([haze, gt, trans], axis=2).astype(np.int32)

        file_idx = os.path.basename(gtf[i]).split('.')[0]
        fname = f'data_{file_idx}.tif'
        output_fpath = os.path.join(outputdatadir, fname)
        data = data.transpose(2, 0, 1)
        save_tiff(output_fpath, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process Rain100H dataset')
    parser.add_argument('--datadir',
                        type=str,
                        help='Directory containing the Rain100H dataset',
                        default='/group/jug/ashesh/data/Haze4K/')
    parser.add_argument('--outputdatadir',
                        type=str,
                        help='Directory to save the pre-processed dataset',
                        default='/group/jug/ashesh/data/Haze4KCombined/')
    args = parser.parse_args()
    for dtypedir in ['train', 'test']:
        transform_data(os.path.join(args.datadir, dtypedir), os.path.join(args.outputdatadir, dtypedir), 10)
