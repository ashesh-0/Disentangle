from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from disentangle.analysis.plot_utils import clean_ax
from disentangle.core.psnr import RangeInvariantPsnr


class PredictionData:

    def __init__(self, data_tuples) -> None:
        """
        3 nested list
        first list is list of models. 
        second list is list of channels.
        third list contains [prediction, target]
        
        """
        self._data = data_tuples

    def target(self, channel_idx, model_idx=None):
        model_idx = 0 if model_idx is None else model_idx
        return self._data[model_idx][channel_idx][1]

    def prediction(self, model_idx, channel_idx):
        return self._data[model_idx][channel_idx][0]

    def input(self, model_idx=None):
        return (self.target(0, model_idx=model_idx) + self.target(1, model_idx=model_idx)) / 2

    def model_count(self):
        return len(self._data)


def get_random_positions_around(h_list, w_list, randomness=100):
    pos = []
    for h_center, w_center in zip(h_list, w_list):
        h_ = h_center + (np.random.randint(randomness) - 2 * randomness)
        w_ = w_center + (np.random.randint(randomness) - 2 * randomness)
        pos.append((h_, w_))
    return pos


def plot_crop_predictions(
    pred_data_obj,
    h_list,
    w_list,
    crop_sz=150,
    mplib_img_sz=2,
    mplib_example_spacing=1,
    mplib_grid_factor=5,
    inset_rect=[0.1, 0.1, 0.4, 0.2],
    inset_min_labelsize=10,
    color_ch_list=['goldenrod', 'cyan'],
    color_pred='red',
    output_filepath=None,
    add_inset=True,
    add_psnr=True,
    imshow_with_vmax=True,
):
    """
    Here, we have single frame predictions. From it we crop patches from h_list,w_list being top left corner locations.
    We show the input, target and predictions.
    """

    axes_list = []
    # h_list = [2000, 1200,  1550, 850] #1500,
    # w_list = [1600, 2300, 1050, 650] #1250
    sz = crop_sz

    ncol_imgs = pred_data_obj.model_count() + 2  # Total number of images in one row
    img_sz = mplib_img_sz  # Size of 1 image.
    example_spacing = mplib_example_spacing  # how much offset should be there in terms of grid spec grids.

    grid_factor = mplib_grid_factor  # if grid factor is 5, then we will create 5 times more rows and columns than the total image size. This allows us to correctly set example_spacing
    # For example, if with example_spacing=1, we still see large pacing between examples, we can simply increase the grid_factor and the spacing should get down.
    # We will also have to set the hspace and wspace appropriately if we change grid_factor.

    pos_list = list(zip(h_list, w_list))
    nrow_imgs = 2 * len(pos_list)

    fig_w = ncol_imgs * img_sz
    fig_h = img_sz * nrow_imgs + example_spacing * (len(pos_list) - 1) / grid_factor
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(nrows=int(grid_factor * fig_h), ncols=grid_factor * fig_w, hspace=0.2, wspace=0.2)
    grid_img_sz = img_sz * grid_factor
    # ax_inp = fig.add_subplot(gs[:grid_img_sz, :grid_img_sz])
    inp = pred_data_obj.input()
    # ax_inp.imshow(inp)

    for pos_idx in range(len(pos_list)):
        h_, w_ = pos_list[pos_idx]
        row_s = 2 * pos_idx * grid_img_sz + pos_idx * example_spacing
        # Get the targets.
        ch0_tar = pred_data_obj.target(0)[h_:h_ + sz, w_:w_ + sz]
        ch1_tar = pred_data_obj.target(1)[h_:h_ + sz, w_:w_ + sz]
        vmax0 = pred_data_obj.target(0).max()
        vmax1 = pred_data_obj.target(1).max()
        if imshow_with_vmax != True:
            vmax0 = None
            vmax1 = None
        # Full input frame.
        ax_temp = fig.add_subplot(gs[row_s:row_s + grid_img_sz, :grid_img_sz])
        ax_temp.imshow(inp)
        axes_list.append(ax_temp)
        ax_temp.add_patch(Rectangle((w_, h_), sz, sz, edgecolor='red', facecolor='none', lw=1))
        clean_ax(ax_temp)

        # input patch.
        ax_temp = fig.add_subplot(gs[row_s + grid_img_sz:row_s + grid_img_sz + grid_img_sz, :grid_img_sz])
        ax_temp.imshow(inp[h_:h_ + sz, w_:w_ + sz])
        axes_list.append(ax_temp)
        ax_temp.add_patch(Rectangle((1, 1), int(sz * 0.99), int(sz * 0.99), edgecolor='red', facecolor='none', lw=2))

        if add_inset:
            _ = add_pixel_kde(ax_temp,
                              inset_rect,
                              pred_data_obj.target(0)[h_:h_ + sz, w_:w_ + sz],
                              pred_data_obj.target(1)[h_:h_ + sz, w_:w_ + sz],
                              inset_min_labelsize,
                              label1='Ch1',
                              label2='Ch2',
                              color1=color_ch_list[0],
                              color2=color_ch_list[1])

        # inset_ax.set_xlim([0,vmax]) #xmin=0,xmax= xmax_data
        # inset_ax.set_xlim(0,max(pred_data_obj.target(0)[h_:h_+sz,w_:w_+sz]))

        clean_ax(ax_temp)

        for i in range(1, 1 + pred_data_obj.model_count()):
            # channel 0
            ch0_pred = pred_data_obj.prediction(i - 1, 0)[h_:h_ + sz, w_:w_ + sz]

            ax_temp = fig.add_subplot(gs[row_s:row_s + grid_img_sz, grid_img_sz * i:grid_img_sz * i + grid_img_sz])
            ax_temp.imshow(ch0_pred, vmax=vmax0)
            axes_list.append(ax_temp)

            if add_psnr:
                psnr = RangeInvariantPsnr(ch0_tar[None], ch0_pred[None]).item()
                ax_temp.text(sz * 1 / 15,
                             sz * 2 / 15,
                             f"PSNR {psnr:.1f}",
                             bbox=dict(fill=False, linewidth=0),
                             color='white')  #edgecolor='red', linewidth=1

            clean_ax(ax_temp)
            if add_inset:
                _ = add_pixel_kde(ax_temp,
                                  inset_rect,
                                  ch0_pred,
                                  ch0_tar,
                                  inset_min_labelsize,
                                  label1='Ch1',
                                  label2='Ch2',
                                  color1=color_pred,
                                  color2=color_ch_list[0])

            # inset_ax.set_xlim([0,vmax])

            # channel 1
            ch1_pred = pred_data_obj.prediction(i - 1, 1)[h_:h_ + sz, w_:w_ + sz]
            ax_temp = fig.add_subplot(gs[row_s + grid_img_sz:row_s + 2 * grid_img_sz,
                                         grid_img_sz * i:grid_img_sz * i + grid_img_sz])
            ax_temp.imshow(ch1_pred, vmax=vmax1)
            axes_list.append(ax_temp)

            if add_psnr:
                psnr = RangeInvariantPsnr(ch1_tar[None], ch1_pred[None]).item()
                ax_temp.text(sz * 1 / 15,
                             sz * 2 / 15,
                             f"PSNR {psnr:.1f}",
                             bbox=dict(fill=False, linewidth=0),
                             color='white')  #edgecolor='red', linewidth=1

            clean_ax(ax_temp)
            if add_inset:
                _ = add_pixel_kde(ax_temp,
                                  inset_rect,
                                  ch1_pred,
                                  ch1_tar,
                                  inset_min_labelsize,
                                  label1='Ch1',
                                  label2='Ch2',
                                  color1=color_pred,
                                  color2=color_ch_list[1])
            # inset_ax.set_xlim([0,vmax])

        # Show target
        i += 1
        ax_temp = fig.add_subplot(gs[row_s:row_s + grid_img_sz, grid_img_sz * i:grid_img_sz * i + grid_img_sz])
        ax_temp.imshow(ch0_tar, vmax=vmax0)
        axes_list.append(ax_temp)
        if add_inset:
            _ = add_pixel_kde(ax_temp,
                              inset_rect,
                              ch0_tar,
                              None,
                              inset_min_labelsize,
                              label1='Ch1',
                              label2='Ch2',
                              color1=color_ch_list[0])
        clean_ax(ax_temp)
        # inset_ax.set_xlim([0,vmax])

        ax_temp = fig.add_subplot(gs[row_s + grid_img_sz:row_s + 2 * grid_img_sz,
                                     grid_img_sz * i:grid_img_sz * i + grid_img_sz])
        ax_temp.imshow(ch1_tar, vmax=vmax1)
        axes_list.append(ax_temp)
        if add_inset:
            _ = add_pixel_kde(ax_temp,
                              inset_rect,
                              ch1_tar,
                              None,
                              inset_min_labelsize,
                              label1='Ch1',
                              label2='Ch2',
                              color1=color_ch_list[1])
        clean_ax(ax_temp)
        # inset_ax.set_xlim([0,vmax])

    if output_filepath:
        plt.savefig(output_filepath, dpi=200, bbox_inches='tight')

        print('Saved to ', output_filepath)

    return axes_list
