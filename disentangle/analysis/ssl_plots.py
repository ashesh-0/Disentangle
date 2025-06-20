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
        hs = np.random.randint(tar_unnorm[0].shape[-3]-sz)
    if ws is None:
        ws = np.random.randint(tar_unnorm[0].shape[-2]-sz)
    print(img_idx, hs, ws)

    for i in range(ncols-1):
        vmin = tar_unnorm[img_idx][hs:hs+sz, ws:ws+sz ,i].min()
        vmax = tar_unnorm[img_idx][hs:hs+sz, ws:ws+sz ,i].max()
        ax[0,i+1].imshow(tar_unnorm[img_idx][...,hs:hs+sz, ws:ws+sz ,i].squeeze(), vmin=vmin, vmax=vmax)
        ax[1,i+1].imshow(pred_unnorm[img_idx][...,hs:hs+sz, ws:ws+sz,i].squeeze(), vmin=vmin, vmax=vmax)

    if 'input_idx' in config.data and config.data.input_idx is not None:
        inp = inp_unnorm[img_idx].squeeze()
    else:
        inp = np.mean(tar_unnorm[img_idx].squeeze(), axis=-1)

    ax[0,0].imshow(inp)
    rect = patches.Rectangle((ws, hs), sz,sz, linewidth=2, edgecolor='r', facecolor='none')
    ax[0,0].add_patch(rect)

    ax[1,0].imshow(inp[hs:hs+sz, ws:ws+sz])
    # reconstructed input
    inp_recons = pred_unnorm[img_idx][...,0] *best_t_estimate + pred_unnorm[img_idx][...,1] * (1-best_t_estimate)
    inp_recons = inp_recons.squeeze()
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


def plot_finetuning_loss(finetuning_output_dict, loss_rolling=50):
    print('Plotting finetuning loss')
    _,ax = plt.subplots(figsize=(18,9), ncols=6,nrows=3)
    pd.Series(finetuning_output_dict['loss']).rolling(loss_rolling).mean().plot(ax=ax[0,0], logy=True, label = 'total loss')
    pd.Series(finetuning_output_dict['factor1']).rolling(5).mean().plot(ax=ax[0,1], label='$\sigma_1$')
    pd.Series(finetuning_output_dict['offset1']).rolling(5).mean().plot(ax=ax[0,2], label='$\mu_1$')
    pd.Series(finetuning_output_dict['val_loss']).rolling(5).mean().plot(ax=ax[0,3], label='val loss')
    # pd.Series(finetuning_output_dict['factor2']).rolling(5).mean().plot(ax=ax[0,3], label='$\sigma_2$')
    # pd.Series(finetuning_output_dict['offset2']).rolling(5).mean().plot(ax=ax[0,4], label='$\mu_2$')
    pd.Series(finetuning_output_dict['mixing_ratio']).rolling(5).mean().plot(ax=ax[0,5], label='$t$')
    pd.Series(finetuning_output_dict['loss_inp']).rolling(loss_rolling).mean().plot(ax=ax[1,0], logy=True, label = '$loss_{inp}$')
    pd.Series(finetuning_output_dict['loss_pred']).rolling(loss_rolling).mean().plot(ax=ax[1,1], logy=True, label = '$loss_{pred}$')
    pd.Series(finetuning_output_dict['stats_loss']).plot(ax=ax[1,2], label = '$loss_{stats}$')
    # place a vertical line at finetuning_output_dict['best_step'] for each ax
    for i in range(6):
        if i ==3:
            continue
        ax[0,i].axvline(finetuning_output_dict['best_step'], color='r', linestyle='--', ymax = 0.2, label='Best Step')
        ax[1,i].axvline(finetuning_output_dict['best_step'], color='r', linestyle='--', ymax =0.2, label='Best Step')
    # 
    if 'psnr' in finetuning_output_dict and finetuning_output_dict['psnr'][0] is not None:
        psnr = np.array(finetuning_output_dict['psnr'])
        for i in range(psnr.shape[1]):
            pd.Series(psnr[:,i]).plot(ax=ax[2,i], label = f'PSNR {i}')
            ax[2,i].legend()

    if 'discrim_real' in finetuning_output_dict:
        assert 'discrim_fake' in finetuning_output_dict
        assert 'grad_penalty' in finetuning_output_dict
        ax[1,3].plot(pd.Series(finetuning_output_dict['discrim_real']).rolling(loss_rolling).mean(), label='D_Real')
        ax[1,3].plot(pd.Series(finetuning_output_dict['discrim_fake']).rolling(loss_rolling).mean(), label='D_Fake')
        ax[1,4].plot(pd.Series(finetuning_output_dict['grad_penalty']).rolling(loss_rolling).mean(), label='Gradient Penalty')
        ax[1,3].legend()
        ax[1,4].legend()

    ax[0,0].legend()  
    ax[0,1].legend()
    ax[0,2].legend()
    ax[0,3].legend()
    ax[0,4].legend()
    ax[0,5].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    ax[1,2].legend()

    plt.tight_layout()

