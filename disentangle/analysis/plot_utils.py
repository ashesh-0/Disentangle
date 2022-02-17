import numpy as np

import matplotlib.pyplot as plt


def clean_ax(ax):
    # 2D or 1D axes are of type np.ndarray
    if isinstance(ax, np.ndarray):
        for one_ax in ax:
            clean_ax(one_ax)
        return

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(left=False, right=False, top=False, bottom=False)


def plot_one_batch_twinnoise(imgs, plot_width=20):
    batch_size = len(imgs)
    ncols = batch_size // 2
    img_sz = plot_width // ncols
    _, ax = plt.subplots(figsize=(ncols * img_sz, 2 * img_sz), ncols=ncols, nrows=2)
    for i in range(ncols):
        ax[0, i].imshow(imgs[i, 0])
        ax[1, i].imshow(imgs[i + batch_size // 2, 0])

        ax[1, i].set_title(f'{i+1+batch_size//2}.')
        ax[0, i].set_title(f'{i+1}.')

        ax[0, i].tick_params(left=False, right=False, top=False, bottom=False)
        ax[0, i].axis('off')
        ax[1, i].tick_params(left=False, right=False, top=False, bottom=False)
        ax[1, i].axis('off')
