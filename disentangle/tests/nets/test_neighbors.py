import matplotlib.pyplot as plt
import torch

from disentangle.analysis.plot_utils import clean_ax
from disentangle.nets.lvae_autoregressive_ra import Neighbors


def plot_nbrs(top, bottom, center, left, right):
    vmin = min(top.min(), center.min(), bottom.min(), left.min(), right.min())
    vmax = max(top.max(), center.max(), bottom.max(), left.max(), right.max())
    _, ax = plt.subplots(figsize=(9, 9), ncols=3, nrows=3)
    ax[1, 1].imshow(center[0, 0], vmin=vmin, vmax=vmax)
    ax[0, 1].imshow(top[0, 0], vmin=vmin, vmax=vmax)
    ax[2, 1].imshow(bottom[0, 0], vmin=vmin, vmax=vmax)
    ax[1, 0].imshow(left[0, 0], vmin=vmin, vmax=vmax)
    ax[1, 2].imshow(right[0, 0], vmin=vmin, vmax=vmax)
    clean_ax(ax)
    for oneax in ax.reshape(-1, ):
        oneax.spines["top"].set_visible(False)
        oneax.spines["right"].set_visible(False)
        oneax.spines["bottom"].set_visible(False)
        oneax.spines["left"].set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)


def _test_neighbors():
    top = img[..., :64, 64:128]
    center = img[..., 64:128, 64:128]
    bottom = img[..., 128:192, 64:128]
    left = img[..., 64:128, :64]
    right = img[..., 64:128, 128:192]

    nbr_cls = Neighbors(top, bottom, left, right)
    hflip = True
    vflip = True
    nbr_cls.flip(hflip=hflip, vflip=vflip)
    new_center = center
    if hflip:
        new_center = torch.flip(new_center, dims=(3, ))
    if vflip:
        new_center = torch.flip(new_center, dims=(2, ))

    new_top, new_bottom, new_left, new_right = nbr_cls.get()
    plot_nbrs(new_top, new_bottom, new_center, new_left, new_right)

    # Here, we now rotate the image.
    nbr_cls = Neighbors(top, bottom, left, right)
    hflip = True
    vflip = True
    nbr_cls.flip(hflip=hflip, vflip=vflip)
    new_center = center
    if hflip:
        new_center = torch.flip(new_center, dims=(3, ))
    if vflip:
        new_center = torch.flip(new_center, dims=(2, ))

    new_top, new_bottom, new_left, new_right = nbr_cls.get()
    plot_nbrs(new_top, new_bottom, new_center, new_left, new_right)
