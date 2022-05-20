import numpy as np
import matplotlib.pyplot as plt
import torch

from disentangle.utils import PSNR
from disentangle.analysis.lvae_utils import get_img_from_forward_output
from disentangle.analysis.critic_notebook_utils import get_mmse_dict, get_label_separated_loss


def clean_ax(ax):
    # 2D or 1D axes are of type np.ndarray
    if isinstance(ax, np.ndarray):
        for one_ax in ax:
            clean_ax(one_ax)
        return

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(left=False, right=False, top=False, bottom=False)


def add_text(ax, text, img_shape, place='TOP_LEFT'):
    """
    Adding text on image
    """
    assert place in ['TOP_LEFT', 'BOTTOM_RIGHT']
    if place == 'TOP_LEFT':
        ax.text(img_shape[1] * 20 / 500, img_shape[0] * 35 / 500, text, bbox=dict(facecolor='white', alpha=0.9))
    elif place == 'BOTTOM_RIGHT':
        s0 = img_shape[1]
        s1 = img_shape[0]
        ax.text(s0 - s0 * 150 / 500, s1 - s1 * 35 / 500, text, bbox=dict(facecolor='white', alpha=0.9))


def plot_one_batch_twinnoise(imgs, plot_width=20):
    batch_size = len(imgs)
    ncols = batch_size // 2
    img_sz = plot_width // ncols
    _, ax = plt.subplots(figsize=(ncols * img_sz, 2 * img_sz), ncols=ncols, nrows=2)
    for i in range(ncols):
        ax[0, i].imshow(imgs[i, 0])
        ax[1, i].imshow(imgs[i + batch_size // 2, 0])

        ax[1, i].set_title(f'{i + 1 + batch_size // 2}.')
        ax[0, i].set_title(f'{i + 1}.')

        ax[0, i].tick_params(left=False, right=False, top=False, bottom=False)
        ax[0, i].axis('off')
        ax[1, i].tick_params(left=False, right=False, top=False, bottom=False)
        ax[1, i].axis('off')


def get_worst_k(arr, N):
    """
    Returns the k lowest elements, in the sorted order.
    """
    ind = np.argpartition(arr, -1 * N)[-1 * N:]
    return ind[np.argsort(arr[ind])]


def plot_imgs_from_idx(idx_list, val_dset, model, model_type):
    """
    Plots  images and their disentangled predictions. Input is a list of idx for which this is done.
    """
    ncols = 5
    nrows = len(idx_list)
    img_sz = 20 / ncols
    _, ax = plt.subplots(figsize=(ncols * img_sz, nrows * img_sz), ncols=ncols, nrows=nrows)

    with torch.no_grad():
        for ax_idx, img_idx in enumerate(idx_list):
            inp, tar = val_dset[img_idx]
            inp = torch.Tensor(inp[None]).cuda()
            tar = torch.Tensor(tar[None]).cuda()

            x_normalized = model.normalize_input(inp)
            target_normalized = model.normalize_target(tar)

            recon_normalized, td_data = model(x_normalized)
            imgs = get_img_from_forward_output(recon_normalized, model)
            loss_dic = get_mmse_dict(model, x_normalized, target_normalized, 1, model_type)
            ll1, ll2 = get_label_separated_loss(loss_dic['mmse_rec_loss'])

            inp = inp.cpu().numpy()
            tar = tar.cpu().numpy()
            imgs = imgs.cpu().numpy()

            psnr1 = PSNR(tar[0, 0], imgs[0, 0])
            psnr2 = PSNR(tar[0, 1], imgs[0, 1])

            ax[ax_idx, 0].imshow(inp[0, 0])

            # max and min values for label 1
            l1_max = max(tar[0, 0].max(), imgs[0, 0].max())
            l1_min = min(tar[0, 0].min(), imgs[0, 0].min())

            ax[ax_idx, 1].imshow(tar[0, 0], vmin=l1_min, vmax=l1_max)
            ax[ax_idx, 2].imshow(imgs[0, 0], vmin=l1_min, vmax=l1_max)
            add_text(ax[ax_idx, 2], f'PSNR:{psnr1:.1f}', inp.shape[-2:])
            txt = f'{int(l1_min)}-{int(l1_max)}'
            add_text(ax[ax_idx, 2], txt, inp.shape[-2:], place='BOTTOM_RIGHT')
            add_text(ax[ax_idx, 1], txt, inp.shape[-2:], place='BOTTOM_RIGHT')

            # max and min values for label 2
            l2_max = max(tar[0, 1].max(), imgs[0, 1].max())
            l2_min = min(tar[0, 1].min(), imgs[0, 1].min())
            ax[ax_idx, 3].imshow(tar[0, 1], vmin=l2_min, vmax=l2_max)
            ax[ax_idx, 4].imshow(imgs[0, 1], vmin=l2_min, vmax=l2_max)
            txt = f'{int(l2_min)}-{int(l2_max)}'
            add_text(ax[ax_idx, 4], f'PSNR:{psnr2:.1f}', inp.shape[-2:])
            add_text(ax[ax_idx, 4], txt, inp.shape[-2:], place='BOTTOM_RIGHT')
            add_text(ax[ax_idx, 3], txt, inp.shape[-2:], place='BOTTOM_RIGHT')

            ax[ax_idx, 2].set_title(f'Error: {ll1[0]:.3f}')
            ax[ax_idx, 4].set_title(f'Error: {ll2[0]:.3f}')
            ax[ax_idx, 0].set_title(f'Id:{img_idx}')
            ax[ax_idx, 1].set_title('Image 1')
            ax[ax_idx, 3].set_title('Image 2')
