import os

import numpy as np

from czifile import imread as imread_czi


def get_fpaths(datadir):
    fnames = ['Experiment-447.czi', 'Experiment-449.czi', 'Experiment-452.czi', 'Experiment-448.czi']
    return [os.path.join(datadir, x) for x in fnames]


def load_data(fpaths):
    imgs = []
    for fpath in fpaths:
        img = imread_czi(fpath)
        assert img.shape[3] == 1
        img = np.swapaxes(img, 0, 3)
        # the first dimension of img stored in imgs will have dim of 1, where the contenation will happen
        imgs.append(img)
    return np.concatenate(imgs, axis=0)
