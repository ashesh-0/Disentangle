# pylibtiff
import numpy as np

from libtiff import TIFF


def load_tiff(path):
    tif = TIFF.open(path)
    imgs = []
    for image in tif.iter_images():
        imgs.append(image[None])

    return np.concatenate(imgs, axis=0)
