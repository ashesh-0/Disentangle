import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from disentangle.analysis.plot_utils import clean_ax
from disentangle.core.psnr import RangeInvariantPsnr


def plot_one_sample(tar_unnorm, pred_unnorm, inp_unnorm, config, best_t_estimate, hs=None, ws=None, sz=800, img_idx=None):
    ncols = tar_unnorm[0].shape[-1] + 1
    imgsz = 4
    _,ax = plt.subplots(figsize=((1+ncols)*imgsz,2*imgsz),nrows=2,ncols=ncols + 1)

    if img_idx is None:
        img_idx = np.random.randint(len(tar_unnorm))
    if hs is None:
        hs = np.random.randint(tar_unnorm[0].shape[1]-sz)
    if ws is None:
        ws = np.random.randint(tar_unnorm[0].shape[2]-sz)
    print(img_idx, hs, ws)

    for i in range(ncols-1):
        vmin = tar_unnorm[img_idx][0,hs:hs+sz, ws:ws+sz ,i].min()
        vmax = tar_unnorm[img_idx][0,hs:hs+sz, ws:ws+sz ,i].max()
        ax[0,i+1].imshow(tar_unnorm[img_idx][0,hs:hs+sz, ws:ws+sz ,i], vmin=vmin, vmax=vmax)
        ax[1,i+1].imshow(pred_unnorm[img_idx][0,hs:hs+sz, ws:ws+sz,i], vmin=vmin, vmax=vmax)

    if 'input_idx' in config.data and config.data.input_idx is not None:
        inp = inp_unnorm[img_idx][0]
    else:
        inp = np.mean(tar_unnorm[img_idx][0], axis=-1)

    ax[0,0].imshow(inp)
    rect = patches.Rectangle((ws, hs), sz,sz, linewidth=2, edgecolor='r', facecolor='none')
    ax[0,0].add_patch(rect)

    ax[1,0].imshow(inp[hs:hs+sz, ws:ws+sz])
    # reconstructed input
    inp_recons = pred_unnorm[img_idx][0,...,0] *best_t_estimate + pred_unnorm[img_idx][0,...,1] * (1-best_t_estimate)
    ax[1,-1].imshow(inp_recons[hs:hs+sz, ws:ws+sz])
    psnr_inp = f'{RangeInvariantPsnr(inp[None]*1.0, inp_recons[None]).item():.1f}'
    ax[1, -1].set_title(f'Recons Input (PSNR {psnr_inp})')

    ax[0, -1].axis("off")


    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    ax[0,0].set_title('Input')
    twinx = ax[0,-2].twinx()
    twinx.set_ylabel('Target')
    clean_ax(twinx)
    twinx = ax[1,-2].twinx()
    clean_ax(twinx)
    twinx.set_ylabel('Prediction')
    clean_ax(ax)
    # plt.tight_layout()
    return (img_idx, hs, ws)


def plot_finetuning_loss(finetuning_output_dict):
    _,ax = plt.subplots(figsize=(18,6), ncols=6,nrows=2)
    pd.Series(finetuning_output_dict['loss']).rolling(50).mean().plot(ax=ax[0,0], logy=True, label = 'total loss')
    pd.Series(finetuning_output_dict['factor1']).rolling(5).mean().plot(ax=ax[0,1], label='$\sigma_1$')
    pd.Series(finetuning_output_dict['offset1']).rolling(5).mean().plot(ax=ax[0,2], label='$\mu_1$')
    pd.Series(finetuning_output_dict['factor1']).rolling(5).mean().plot(ax=ax[0,3], label='$\sigma_2$')
    pd.Series(finetuning_output_dict['offset2']).rolling(5).mean().plot(ax=ax[0,4], label='$\mu_2$')
    pd.Series(finetuning_output_dict['mixing_ratio']).rolling(5).mean().plot(ax=ax[0,5], label='$t$')
    pd.Series(finetuning_output_dict['loss_inp']).rolling(50).mean().plot(ax=ax[1,0], logy=True, label = '$loss_{inp}$')
    pd.Series(finetuning_output_dict['loss_pred']).rolling(50).mean().plot(ax=ax[1,1], logy=True, label = '$loss_{pred}$')
    ax[0,0].legend()  
    ax[0,1].legend()
    ax[0,2].legend()
    ax[0,3].legend()
    ax[0,4].legend()
    ax[0,5].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    plt.tight_layout()

