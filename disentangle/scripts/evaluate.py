import argparse
import glob
import json
import os
import pickle
import random
import re
import sys
from copy import deepcopy
from posixpath import basename
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

import ml_collections
from disentangle.analysis.critic_notebook_utils import get_label_separated_loss, get_mmse_dict
from disentangle.analysis.lvae_utils import get_img_from_forward_output
from disentangle.analysis.mmse_prediction import get_dset_predictions
from disentangle.analysis.paper_plots import get_predictions as get_patch_predictions
from disentangle.analysis.plot_utils import clean_ax, get_k_largest_indices, plot_imgs_from_idx
from disentangle.analysis.results_handler import PaperResultsHandler
from disentangle.analysis.stitch_prediction import stitch_predictions
from disentangle.config_utils import load_config
from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples
from disentangle.core.data_type import DataType
from disentangle.core.loss_type import LossType
from disentangle.core.model_type import ModelType
from disentangle.core.psnr import PSNR, RangeInvariantPsnr
from disentangle.core.ssim import compute_SE
from disentangle.core.tiff_reader import load_tiff
from disentangle.data_loader.lc_multich_dloader import LCMultiChDloader
from disentangle.data_loader.patch_index_manager import TilingMode
# from disentangle.data_loader.two_tiff_rawdata_loader import get_train_val_data
from disentangle.data_loader.vanilla_dloader import MultiChDloader, get_train_val_data
from disentangle.sampler.random_sampler import RandomSampler
from disentangle.scripts.run import overwride_with_cmd_params
from disentangle.training import create_dataset, create_model
from microssim import MicroMS3IM, MicroSSIM

torch.multiprocessing.set_sharing_strategy("file_system")
DATA_ROOT = "/group/jug/ashesh/data/"
CODE_ROOT = "PUT THE ROOT DIRECTORY FOR THE CODE HERE"


def _avg_psnr(target, prediction, psnr_fn):
    """
    Returns the mean PSNR and the standard error of the mean.
    """
    # multiplication with 1.0 is to ensure that the data is float.
    psnr_arr = [psnr_fn(target[i][None]*1.0, prediction[i][None]*1.0).item() for i in range(len(prediction))]
    mean_psnr = np.mean(psnr_arr)
    std_err_psnr = compute_SE(psnr_arr)
    return round(mean_psnr, 2), round(std_err_psnr, 3)


def avg_range_inv_psnr(target, prediction):
    return _avg_psnr(target, prediction, RangeInvariantPsnr)


def avg_psnr(target, prediction):
    return _avg_psnr(target, prediction, PSNR)



def fix_seeds():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True


def upperclip_data(data, max_val):
    """
    data: (N, H, W, C)
    """
    if isinstance(max_val, list):
        chN = data.shape[-1]
        assert chN == len(max_val)
        for ch in range(chN):
            ch_data = data[..., ch]
            ch_q = max_val[ch]
            ch_data[ch_data > ch_q] = ch_q
            data[..., ch] = ch_data
    else:
        data[data > max_val] = max_val
    return True


def compute_max_val(data, data_config):
    if data_config.get("channelwise_quantile", False):
        max_val_arr = [np.quantile(data[..., i], data_config.clip_percentile) for i in range(data.shape[-1])]
        return max_val_arr
    else:
        return np.quantile(data, data_config.clip_percentile)

def _get_list_of_images_from_gt_pred(gt, pred, ch_idx):
    """
    Whether you have 2D data or 3D data, this function will return a list of images HixWi.
    """
    gt_list = []
    pred_list = []
    if isinstance(gt, list):
        # assert len(gt[0].shape) == 4, f"expected N x H x W x C, but got {gt[0].shape}"
        for i in range(len(gt)):
            gt_list_tmp, pred_list_tmp = _get_list_of_images_from_gt_pred(gt[i], pred[i], ch_idx)
            gt_list += gt_list_tmp
            pred_list += pred_list_tmp
    elif isinstance(gt, np.ndarray):
        if len(gt.shape) == 3:
            return [gt[..., ch_idx]], [pred[..., ch_idx]]
        else:
            assert gt.shape == pred.shape, f"gt shape: {gt.shape}, pred shape: {pred.shape}"
            for n_idx in range(gt.shape[0]):
                gt_list_tmp, pred_list_tmp = _get_list_of_images_from_gt_pred(gt[n_idx], pred[n_idx], ch_idx)
                gt_list += gt_list_tmp
                pred_list += pred_list_tmp
    return gt_list, pred_list


def compute_high_snr_stats(highres_data, pred_unnorm, verbose=True):
    """
    last dimension is the channel dimension
    """
    # is_5D = isinstance(highres_data, np.ndarray) and len(highres_data.shape) == 5
    
    #  channel index -> ssim object
    # ssim_obj_dict = {}
    # m3ssim_obj_dict = {}
    psnr_list = []
    microssim_list = []
    ms3im_list = []
    for ch_idx in range(highres_data[0].shape[-1]):
        # list of gt and prediction images. This handles both 2D and 3D data. This also handles when individual images are lists.
        gt_ch, pred_ch = _get_list_of_images_from_gt_pred(highres_data, pred_unnorm, ch_idx)
        # PSNR
        psnr_list.append(avg_range_inv_psnr(gt_ch, pred_ch))
        # MicroSSIM
        microssim_obj = MicroSSIM()
        microssim_obj.fit(gt_ch, pred_ch)
        mssim_scores = [microssim_obj.score(gt_ch[i], pred_ch[i]) for i in range(len(gt_ch))]
        microssim_list.append((np.mean(mssim_scores), compute_SE(mssim_scores)))
        # MicroS3IM
        m3sim_obj = MicroMS3IM()
        m3sim_obj.fit(gt_ch, pred_ch)
        ms3im_scores = [m3sim_obj.score(gt_ch[i], pred_ch[i]) for i in range(len(gt_ch))]
        ms3im_list.append((np.mean(ms3im_scores), compute_SE(ms3im_scores)))


    if verbose:
        def ssim_str(ssim_tmp):
            return f'{np.round(ssim_tmp[0], 3):.3f}+-{np.round(ssim_tmp[1], 3):.3f}'
        def psnr_str(psnr_tmp):
            return f'{np.round(psnr_tmp[0], 2)}+-{np.round(psnr_tmp[1], 3)}'
        print("PSNR on Highres", '\t'.join([psnr_str(psnr_tmp) for psnr_tmp in psnr_list]))
        print("MicroSSIM on Highres", '\t'.join([ssim_str(ssim) for ssim in microssim_list]))
        print("MicroS3IM on Highres", '\t'.join([ssim_str(ssim) for ssim in ms3im_list]))
    
    return {
        "rangeinvpsnr": psnr_list,
        "microssim": microssim_list,
        "ms3im": ms3im_list,
    }


def get_data_without_synthetic_noise(data_dir, config, eval_datasplit_type):
    """
    Here, we don't add any synthetic noise.
    """
    assert ("synthetic_gaussian_scale" in config.data or "poisson_noise_factor" in config.data)
    assert config.data.synthetic_gaussian_scale > 0
    data_config = deepcopy(config.data)
    if "poisson_noise_factor" in data_config:
        data_config.poisson_noise_factor = -1
    if "synthetic_gaussian_scale" in data_config:
        data_config.synthetic_gaussian_scale = None
    return _get_highres_data_internal(data_dir, data_config, config.training, eval_datasplit_type)


def _get_highres_data_internal(data_dir, data_config, training_config, eval_datasplit_type):
    highres_data = get_train_val_data(
        data_config,
        data_dir,
        DataSplitType.Train,
        training_config.val_fraction,
        training_config.test_fraction,
    )
    if not isinstance(highres_data, np.ndarray):
        highres_data = np.concatenate([highres_data[i][0] for i in range(len(highres_data))], axis=0)

    hres_max_val = compute_max_val(highres_data, data_config)
    del highres_data

    highres_data = get_train_val_data(
        data_config,
        data_dir,
        eval_datasplit_type,
        training_config.val_fraction,
        training_config.test_fraction,
    )
    if not isinstance(highres_data, np.ndarray):
        highres_data = [highres_data[i][0] for i in range(len(highres_data))]
        for x in highres_data:
            upperclip_data(x, hres_max_val)
        
        if 'mode_3D' in data_config and data_config.mode_3D and len(highres_data[0].shape) == 4:
            highres_data = [x[None] for x in highres_data]
    else:
        upperclip_data(highres_data, hres_max_val)
        if len(highres_data.shape) == 5 and ('mode_3D' in data_config and not data_config.mode_3D):
            highres_data = highres_data.reshape(-1, *highres_data.shape[2:])
    
    return highres_data


def get_highres_data_ventura(data_dir, config, eval_datasplit_type):
    data_config = ml_collections.ConfigDict()
    data_config.ch1_fname = "actin-60x-noise2-highsnr.tif"
    data_config.ch2_fname = "mito-60x-noise2-highsnr.tif"
    data_config.data_type = DataType.SeparateTiffData
    highres_data = get_train_val_data(
        data_config,
        data_dir,
        DataSplitType.Train,
        config.training.val_fraction,
        config.training.test_fraction,
    )

    hres_max_val = compute_max_val(highres_data, config.data)
    del highres_data

    highres_data = get_train_val_data(
        data_config,
        data_dir,
        eval_datasplit_type,
        config.training.val_fraction,
        config.training.test_fraction,
    )

    # highres_data = highres_data[::5].copy()
    upperclip_data(highres_data, hres_max_val)
    return highres_data


def get_data_dir(dtype):
    if dtype == DataType.CustomSinosoid:
        data_dir = f"{DATA_ROOT}/sinosoid/"
    elif dtype == DataType.CustomSinosoidThreeCurve:
        data_dir = f"{DATA_ROOT}/sinosoid/"
    elif dtype == DataType.OptiMEM100_014:
        data_dir = f"{DATA_ROOT}/microscopy/"
    elif dtype == DataType.Prevedel_EMBL:
        data_dir = f"{DATA_ROOT}/Prevedel_EMBL/PKG_3P_dualcolor_stacks/NoAverage_NoRegistration/"
    elif dtype == DataType.AllenCellMito:
        data_dir = f"{DATA_ROOT}/allencell/2017_03_08_Struct_First_Pass_Seg/AICS-11/"
    elif dtype == DataType.SeparateTiffData:
        data_dir = f"{DATA_ROOT}/ventura_gigascience"
    elif dtype == DataType.BioSR_MRC:
        data_dir = f"{DATA_ROOT}/BioSR/"
    elif dtype == DataType.NicolaData:
        # data_dir = f"{DATA_ROOT}/nikola_data/raw"
        data_dir = f"{DATA_ROOT}/nikola_data/20240531/"
    elif dtype == DataType.Dao3ChannelWithInput:
        data_dir = f"{DATA_ROOT}/Dao4Channel/"
    elif dtype == DataType.Dao3Channel:
        data_dir = f"{DATA_ROOT}/Dao3Channel/"
    elif dtype == DataType.ExpMicroscopyV2:
        data_dir = f"{DATA_ROOT}/expansion_microscopy_v2/datafiles"
    elif dtype == DataType.ExpMicroscopyV1:
        data_dir = f"{DATA_ROOT}/expansion_microscopy_v1/MDCK_MitoDeepRed639_AlphaBetaTub488/"
    elif dtype == DataType.Pavia3SeqData:
        data_dir = f"{DATA_ROOT}/pavia3_sequential_cropped/"
    elif dtype == DataType.TavernaSox2GolgiV2:
        data_dir = f"{DATA_ROOT}/TavernaSox2Golgi/acquisition2/"
    elif dtype == DataType.TavernaSox2Golgi:
        data_dir = f"{DATA_ROOT}/TavernaSox2Golgi/"
    elif dtype == DataType.HTIba1Ki67:
        data_dir = f'{DATA_ROOT}/Stefania/20230327_Ki67_and_Iba1_trainingdata/'
    elif dtype == DataType.Elisa3DData:
        data_dir = f"{DATA_ROOT}/Elisa3D/"
    return data_dir

def get_calibration_stats(calibration_factors, pred, pred_std, tar_normalized, eps= 1e-8):
    from disentangle.metrics.calibration import Calibration
    calib = Calibration(num_bins=30, mode='pixelwise')
    stats = calib.compute_stats(pred, np.log(eps + pred_std * calibration_factors), tar_normalized)
    return stats

def get_calibration_factor(pred, pred_std, tar_normalized, epochs = 10000, lr = 160.0, eps= 1e-8):
    from disentangle.metrics.calibration import get_calibrated_factor_for_stdev
    calib_dicts = []
    for col_idx in range(pred.shape[-1]):

        calib_dict = get_calibrated_factor_for_stdev(pred[...,col_idx], np.log(eps + (pred_std[...,col_idx]**2)), tar_normalized[...,col_idx], 
                                                          lr=lr, epochs=epochs)
        calib_dicts.append(calib_dict)
    
    return calib_dicts

def main(
    ckpt_dir,
    image_size_for_grid_centers=64,
    mmse_count=1,
    custom_image_size=64,
    batch_size=16,
    num_workers=4,
    COMPUTE_LOSS=False,
    use_deterministic_grid=None,
    threshold=None,  # 0.02,
    compute_kl_loss=False,
    evaluate_train=False,
    eval_datasplit_type=DataSplitType.Val,
    val_repeat_factor=None,
    psnr_type="range_invariant",
    ignored_last_pixels=0,
    ignore_first_pixels=0,
    print_token="",
    normalized_ssim=True,
    save_to_file=False,
    # trim_boundary=True,
    predict_kth_frame=None,
    predict_samples_N=None,
    compare_with_highsnr=True,
    train_calibration=False,
    eval_calibration_factors=None,
    override_kwargs=None,
):
    global DATA_ROOT, CODE_ROOT

    homedir = os.path.expanduser("~")
    nodename = os.uname().nodename

    if nodename == "capablerutherford-02aa4":
        DATA_ROOT = "/mnt/ashesh/"
        CODE_ROOT = "/home/ubuntu/ashesh/"
    elif nodename in [
            "capableturing-34a32",
            "colorfuljug-fa782",
            "agileschroedinger-a9b1c",
            "rapidkepler-ca36f",
    ]:
        DATA_ROOT = "/home/ubuntu/ashesh/data/"
        CODE_ROOT = "/home/ubuntu/ashesh/"
    elif (re.match("lin-jug-\d{2}", nodename) or re.match("gnode\d{2}", nodename)
          or re.match("lin-jug-m-\d{2}", nodename) or re.match("lin-jug-l-\d{2}", nodename)):
        DATA_ROOT = "/group/jug/ashesh/data/"
        CODE_ROOT = "/home/ashesh.ashesh/"

    dtype = int(ckpt_dir.split("/")[-2].split("-")[0][1:])
    data_dir = get_data_dir(dtype)

    homedir = os.path.expanduser("~")
    nodename = os.uname().nodename

    def get_best_checkpoint(ckpt_dir):
        output = []
        for filename in glob.glob(ckpt_dir + "/*_best.ckpt"):
            output.append(filename)
        assert len(output) == 1, "\n".join(output)
        return output[0]

    config = load_config(ckpt_dir)
    config = ml_collections.ConfigDict(config)
    if override_kwargs is not None:
        with config.unlocked():
            overwride_with_cmd_params(config, json.loads(override_kwargs))
            config.data.disable_checks = True

    old_image_size = None
    with config.unlocked():
        try:
            if "batchnorm" not in config.model.encoder:
                config.model.encoder.batchnorm = config.model.batchnorm
                assert "batchnorm" not in config.model.decoder
                config.model.decoder.batchnorm = config.model.batchnorm

            if "conv2d_bias" not in config.model.decoder:
                config.model.decoder.conv2d_bias = True

            if config.model.model_type == ModelType.LadderVaeSepEncoder:
                if "use_random_for_missing_inp" not in config.model:
                    config.model.use_random_for_missing_inp = False
                if "learnable_merge_tensors" not in config.model:
                    config.model.learnable_merge_tensors = False

            if "input_is_sum" not in config.data:
                config.data.input_is_sum = False
            # config.model.noise_model_ch1_fpath = config.model.noise_model_ch1_fpath.replace('/home/ashesh.ashesh/training/', '/group/jug/ashesh/training_pre_eccv/')
            # config.model.noise_model_ch2_fpath = config.model.noise_model_ch2_fpath.replace('/home/ashesh.ashesh/training/', '/group/jug/ashesh/training_pre_eccv/')
        except:
            pass

        if 'depth3D' in config.data and config.data.depth3D > 1:
            config.data.mode_3D = True
            config.model.mode_3D = True

        if config.model.model_type == ModelType.UNet and "n_levels" not in config.model:
            config.model.n_levels = 4
        if "test_fraction" not in config.training:
            config.training.test_fraction = 0.0

        if "datadir" not in config:
            config.datadir = ""
        if "encoder" not in config.model:
            config.model.encoder = ml_collections.ConfigDict()
            assert "decoder" not in config.model
            config.model.decoder = ml_collections.ConfigDict()

            config.model.encoder.dropout = config.model.dropout
            config.model.decoder.dropout = config.model.dropout
            config.model.encoder.blocks_per_layer = config.model.blocks_per_layer
            config.model.decoder.blocks_per_layer = config.model.blocks_per_layer
            config.model.encoder.n_filters = config.model.n_filters
            config.model.decoder.n_filters = config.model.n_filters

        if "multiscale_retain_spatial_dims" not in config.model:
            config.multiscale_retain_spatial_dims = False

        if "res_block_kernel" not in config.model.encoder:
            config.model.encoder.res_block_kernel = 3
            assert "res_block_kernel" not in config.model.decoder
            config.model.decoder.res_block_kernel = 3

        if "res_block_skip_padding" not in config.model.encoder:
            config.model.encoder.res_block_skip_padding = False
            assert "res_block_skip_padding" not in config.model.decoder
            config.model.decoder.res_block_skip_padding = False

        if config.data.data_type == DataType.CustomSinosoid:
            if "max_vshift_factor" not in config.data:
                config.data.max_vshift_factor = config.data.max_shift_factor
                config.data.max_hshift_factor = 0
            if "encourage_non_overlap_single_channel" not in config.data:
                config.data.encourage_non_overlap_single_channel = False

        if "skip_bottom_layers_count" in config.model:
            config.model.skip_bottom_layers_count = 0

        if "logvar_lowerbound" not in config.model:
            config.model.logvar_lowerbound = None
        if "train_aug_rotate" not in config.data:
            config.data.train_aug_rotate = False
        if "multiscale_lowres_separate_branch" not in config.model:
            config.model.multiscale_lowres_separate_branch = False
        if "multiscale_retain_spatial_dims" not in config.model:
            config.model.multiscale_retain_spatial_dims = False
        config.data.train_aug_rotate = False

        if "randomized_channels" not in config.data:
            config.data.randomized_channels = False

        if "predict_logvar" not in config.model:
            config.model.predict_logvar = None
        if config.data.data_type in [
                DataType.OptiMEM100_014,
                DataType.CustomSinosoid,
                DataType.CustomSinosoidThreeCurve,
                DataType.SeparateTiffData,
        ]:
            if custom_image_size is not None:
                old_image_size = config.data.image_size
                config.data.image_size = custom_image_size
            if use_deterministic_grid is not None:
                config.data.deterministic_grid = use_deterministic_grid
            if threshold is not None:
                config.data.threshold = threshold
            if val_repeat_factor is not None:
                config.training.val_repeat_factor = val_repeat_factor


    if dtype == DataType.HTIba1Ki67:
        from disentangle.data_loader.ht_iba1_ki67_rawdata_loader import SubDsetType
        config.data.subdset_type = SubDsetType.OnlyIba1P30
        config.data.empty_patch_replacement_enabled = False

    config.model.mode_pred = not compute_kl_loss
    print(config)
    with config.unlocked():
        config.model.skip_nboundary_pixels_from_loss = None

    padding_kwargs = {
        "mode": config.data.get("padding_mode", "constant"),
    }

    if padding_kwargs["mode"] == "constant":
        padding_kwargs["constant_values"] = config.data.get("padding_value", 0)

    dloader_kwargs = {
        "overlapping_padding_kwargs": padding_kwargs,
        "tiling_mode": TilingMode.ShiftBoundary,
    }

    train_dset, val_dset = create_dataset(
        config,
        data_dir,
        eval_datasplit_type=eval_datasplit_type,
        kwargs_dict=dloader_kwargs,
    )

    # For normalizing, we should be using the training data's mean and std.
    mean_dict, std_dict = train_dset.compute_mean_std()
    train_dset.set_mean_std(mean_dict, std_dict)
    val_dset.set_mean_std(mean_dict, std_dict)

    if image_size_for_grid_centers is not None:
        val_dset.set_img_sz(config.data.image_size, image_size_for_grid_centers)
    
    if evaluate_train:
        val_dset = train_dset

    with config.unlocked():
        if (config.data.data_type in [
                DataType.OptiMEM100_014,
                DataType.CustomSinosoid,
                DataType.CustomSinosoidThreeCurve,
                DataType.SeparateTiffData,
        ] and old_image_size is not None):
            config.data.image_size = old_image_size

    model = create_model(config, deepcopy(mean_dict), deepcopy(std_dict))
    ckpt_fpath = get_best_checkpoint(ckpt_dir)
    checkpoint = torch.load(ckpt_fpath)

    _ = model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    _ = model.cuda()

    # model.data_mean = model.data_mean.cuda()
    # model.data_std = model.data_std.cuda()
    model.set_params_to_same_device_as(torch.Tensor([1]).cuda())
    print("Loading from epoch", checkpoint["epoch"])

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model has {count_parameters(model)/1000_000:.3f}M parameters")
    # reducing the data here.
    if predict_kth_frame is not None:
        assert predict_kth_frame >= 0 and isinstance(predict_kth_frame, int), f"Invalid kth frame. {predict_kth_frame}"
        if predict_kth_frame >= val_dset.get_num_frames():
            return None, None
        else:
            val_dset.reduce_data(t_list=[predict_kth_frame])

    if (config.data.multiscale_lowres_count is not None and custom_image_size is not None):
        model.reset_for_different_output_size(custom_image_size)

    # Predict samples and return that.
    if predict_samples_N is not None:
        idx_list = []
        inp_list = []
        tar_list = []
        recons_img_list = {}
        np.random.seed(955)
        for _ in range(predict_samples_N):
            idx = np.random.randint(len(val_dset))
            idx_list.append(idx)
            inp_patch, tar_patch, recons_list = get_patch_predictions(
                idx,
                val_dset,
                model,
                mmse_count=mmse_count,
                patch_size=custom_image_size,
                grid_size=image_size_for_grid_centers,
            )
            inp_list.append(inp_patch)
            tar_list.append(tar_patch)
            samples = np.concatenate([x[None] for x in recons_list], axis=0)
            for channel_idx in range(tar_patch.shape[1]):
                if channel_idx not in recons_img_list:
                    recons_img_list[channel_idx] = []
                recons_img_list[channel_idx].append(samples[:, channel_idx])
        return {
            "inp": inp_list,
            "tar": tar_list,
            "pred": recons_img_list,
            "idx": idx_list,
        }, None
    

    
    pred_tiled, rec_loss, logvar_tiled, patch_psnr_tuple, pred_std_tiled = get_dset_predictions(
        model,
        val_dset,
        batch_size,
        num_workers=num_workers,
        mmse_count=mmse_count,
        model_type=config.model.model_type,
    )
    if pred_tiled.shape[-1] != val_dset.get_img_sz():
        pad = (val_dset.get_img_sz() - pred_tiled.shape[-1]) // 2
        pred_tiled = np.pad(pred_tiled, ((0, 0), (0, 0), (pad, pad), (pad, pad)))

    pred = stitch_predictions(pred_tiled, val_dset)
    pred_std = stitch_predictions(pred_std_tiled,val_dset)

    is_list_prediction = isinstance(pred, list)

    if pred[0].shape[-1] == 2 and pred[0][..., 1].std() == 0:
        print("Denoiser model. Ignoring the second channel")
        pred = pred[..., :1].copy()
        pred_std = pred_std[..., :1].copy()

    print("Stitched predictions shape before ignoring boundary pixels", len(pred), pred[0].shape)

    def print_ignored_pixels():
        ignored_pixels = 1
        if len(pred) == 1:
            return 0

        if isinstance(pred, list):
            return 0
        
        while (pred[
                :10,
                -ignored_pixels:,
                -ignored_pixels:,
        ].std() == 0):
            ignored_pixels += 1
        ignored_pixels -= 1
        return ignored_pixels

    actual_ignored_pixels = print_ignored_pixels()
    assert (ignored_last_pixels >= actual_ignored_pixels
            ), f"ignored_last_pixels: {ignored_last_pixels} < actual_ignored_pixels: {actual_ignored_pixels}"
    assert actual_ignored_pixels == 0, "With the default tiling mode, we should not be ignoring any pixels."

    tar = (val_dset._data if not is_list_prediction else [val_dset.dsets[i]._data for i in range(len(val_dset.dsets))])

    if "target_idx_list" in config.data and config.data.target_idx_list is not None:
        tar = tar[..., config.data.target_idx_list]
        pred = pred[..., :(tar.shape[-1])]
        pred_std = pred_std[...,:(tar.shape[-1])]

    print("Stitched predictions shape after", len(pred), pred[0].shape)


    sep_mean, sep_std = model.data_mean["target"], model.data_std["target"]
    sep_mean = sep_mean.squeeze().reshape(1, 1, 1, -1)
    sep_std = sep_std.squeeze().reshape(1, 1, 1, -1)

    if train_calibration:
        tar_normalized = (tar - sep_mean.cpu().numpy()) / sep_std.cpu().numpy()
        assert eval_datasplit_type == DataSplitType.Val, "Calibration model should be trained on the validation set."
        calib_factors= get_calibration_factor(pred, pred_std, tar_normalized)
        return calib_factors, None
    elif eval_calibration_factors is not None:
        calib_factors = [eval_calibration_factors[i]['scalar'] for i in range(len(eval_calibration_factors))]
        calib_factors = np.array(calib_factors).reshape(1,1,1,-1)
        tar_normalized = (tar - sep_mean.cpu().numpy()) / sep_std.cpu().numpy()
        # assert eval_datasplit_type == DataSplitType.Test, "Calibration model should be evaluated on the test set."
        calib_stats = get_calibration_stats(calib_factors, pred, pred_std, tar_normalized)
        return {'calib_stats':calib_stats}, None

    if is_list_prediction:
        pred_unnorm = [pred[i] * sep_std.cpu().numpy() + sep_mean.cpu().numpy() for i in range(len(pred))]
    else:
        pred_unnorm = pred * sep_std.cpu().numpy() + sep_mean.cpu().numpy()

    highres_data = (get_highsnr_data(config, data_dir, eval_datasplit_type) if compare_with_highsnr else None)
    if predict_kth_frame is not None and highres_data is not None:
        highres_data = highres_data[[predict_kth_frame]].copy()

    if (highres_data is not None and "target_idx_list" in config.data and config.data.target_idx_list is not None):
        highres_data = highres_data[..., config.data.target_idx_list]

    if highres_data is None:
        # Computing the output statistics.
        print(print_token)
        stats_dict = compute_high_snr_stats(tar, pred_unnorm)
        output_stats = {}
        output_stats["rangeinvpsnr"] = stats_dict["rangeinvpsnr"]
        output_stats["microssim"] = stats_dict["microssim"]
        output_stats["ms3im"] = stats_dict["ms3im"]
        print("")

    # highres data
    else:
        # highres_data = ignore_pixels(highres_data)
        # for denoiser, we don't need both channels.
        if config.model.model_type == ModelType.Denoiser:
            if model.denoise_channel == "Ch1":
                highres_data = highres_data[..., :1]
            elif model.denoise_channel == "Ch2":
                highres_data = highres_data[..., 1:]
            elif model.denoise_channel == "input":
                highres_data = np.mean(highres_data, axis=-1, keepdims=True)

        print(print_token)
        stats_dict = compute_high_snr_stats(highres_data, pred_unnorm)
        output_stats = {}
        output_stats["rangeinvpsnr"] = stats_dict["rangeinvpsnr"]
        output_stats["microssim"] = stats_dict["microssim"]
        output_stats["ms3im"] = stats_dict["ms3im"]
        print("")
    return output_stats, pred_unnorm


def synthetic_noise_present(config):
    """
    Returns True if synthetic noise is present.
    """
    gaussian_noise = ("synthetic_gaussian_scale" in config.data and config.data.synthetic_gaussian_scale is not None
                      and config.data.synthetic_gaussian_scale > 0)
    poisson_noise = ("poisson_noise_factor" in config.data and config.data.poisson_noise_factor is not None
                     and config.data.poisson_noise_factor > 0)
    return gaussian_noise or poisson_noise


def get_highsnr_data(config, data_dir, eval_datasplit_type):
    """
    Get the high SNR data.
    """
    highres_data = None
    if (config.model.model_type == ModelType.DenoiserSplitter or config.data.data_type == DataType.SeparateTiffData):
        highres_data = get_highres_data_ventura(data_dir, config, eval_datasplit_type)
    elif config.data.data_type == DataType.NicolaData and config.data.dset_type in ['2ms','3ms','5ms','20ms','500ms']:
        new_config = deepcopy(config)
        new_config.data.dset_type = "500ms"
        highres_data = _get_highres_data_internal(data_dir, new_config.data, config.training, eval_datasplit_type)
    elif config.data.data_type == DataType.NicolaData:
        new_config = deepcopy(config)
        new_config.data.dset_type = "high"
        highres_data = _get_highres_data_internal(data_dir, new_config.data, config.training, eval_datasplit_type)
    elif ("synthetic_gaussian_scale" in config.data or "enable_poisson_noise" in config.data):
        if config.data.data_type == DataType.OptiMEM100_014:
            data_dir = os.path.join(data_dir, "OptiMEM100x014.tif")
        if synthetic_noise_present(config):
            highres_data = get_data_without_synthetic_noise(data_dir, config, eval_datasplit_type)
    
    return highres_data

def stack_list_of_images(images):
    """
    images: list of images
    """
    assert isinstance(images, list)
    shape = list(images[0].shape)

    for img in images:
        for i in range(1,len(shape)):
            shape[i] = max(shape[i], img.shape[i])
    
    shape = [len(images)] + shape
    output = np.zeros(shape, dtype=images[0].dtype)

    for i, img in enumerate(images):
        if len(shape) ==5:
            output[i,:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3]] = img
        elif len(shape) == 6:
            output[i,:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3], :img.shape[4]] = img
        else:
            raise Exception(f"Invalid shape:{shape}")
    
    return output

def save_hardcoded_ckpt_evaluations_to_file(
    normalized_ssim=True,
    save_prediction=False,
    mmse_count=1,
    predict_kth_frame=None,
    ckpt_dir=None,
    patch_size=None,
    grid_size=32,
    overwrite_saved_predictions=True,
    predict_samples_N=None,
    save_prediction_factor=1.0,
    skip_highsnr=False,
    train_calibration=False,
    eval_calibration=False,
    override_kwargs=None,
    # trim_boundary=True,
):
    if ckpt_dir is None:
        ckpt_dirs = [
            '/group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/4',
            '/group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/5',
            '/group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/6',
            '/group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/14',
            '/group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/17',
        ]
    else:
        ckpt_dirs = [ckpt_dir]
    if ckpt_dirs[0].startswith("/group/jug/ashesh"):
        OUTPUT_DIR = os.path.expanduser("/group/jug/ashesh/data/paper_stats/")
    elif ckpt_dirs[0].startswith("/home/ubuntu/ashesh"):
        OUTPUT_DIR = os.path.expanduser("~/data/paper_stats/")
    else:
        raise Exception("Invalid server")

    ckpt_dirs = [x[:-1] if "/" == x[-1] else x for x in ckpt_dirs]

    patchsz_gridsz_tuples = [(patch_size, grid_size)]
    print("Using patch,grid size", patchsz_gridsz_tuples)
    for custom_image_size, image_size_for_grid_centers in patchsz_gridsz_tuples:
        for eval_datasplit_type in [DataSplitType.Test]:
            for ckpt_dir in ckpt_dirs:
                # data_type = int(os.path.basename(os.path.dirname(ckpt_dir)).split("-")[0][1:])
                # if data_type in [
                #         DataType.OptiMEM100_014,
                #         DataType.SemiSupBloodVesselsEMBL,
                #         DataType.Pavia2VanillaSplitting,
                #         DataType.ExpMicroscopyV1,
                #         DataType.ShroffMitoEr,
                #         DataType.HTIba1Ki67,
                # ]:
                #     ignored_last_pixels = 32
                # elif data_type == DataType.BioSR_MRC:
                #     ignored_last_pixels = 44
                #     # assert val_dset.get_img_sz() == 64
                #     # ignored_last_pixels = 108
                # elif data_type == DataType.NicolaData:
                #     ignored_last_pixels = 8
                # elif data_type == DataType.ExpMicroscopyV2:
                #     ignored_last_pixels = 16
                # elif data_type == DataType.TavernaSox2GolgiV2:
                #     ignored_last_pixels = 8
                # else:
                #     ignored_last_pixels = 0
                
                ignored_last_pixels = 0
                if custom_image_size is None:
                    custom_image_size = load_config(ckpt_dir).data.image_size

                handler = PaperResultsHandler(
                    OUTPUT_DIR,
                    eval_datasplit_type,
                    custom_image_size,
                    image_size_for_grid_centers,
                    mmse_count,
                    ignored_last_pixels,
                    predict_kth_frame=predict_kth_frame,
                    multiplicative_factor=save_prediction_factor,
                    train_calibration=train_calibration,
                    eval_calibration=eval_calibration,
                    override_kwargs=override_kwargs,
                )
                eval_calibration_factors = None
                if eval_calibration:
                    calib_factors_fpath = handler.get_calib_factors_path(ckpt_dir)
                    assert os.path.exists(calib_factors_fpath), f"Calibration factors not found at {calib_factors_fpath}"
                    with open(calib_factors_fpath, "rb") as f:
                        eval_calibration_factors = pickle.load(f)
                    
                data, prediction = main(
                    ckpt_dir,
                    image_size_for_grid_centers=image_size_for_grid_centers,
                    mmse_count=mmse_count,
                    custom_image_size=custom_image_size,
                    batch_size=64,
                    num_workers=4,
                    COMPUTE_LOSS=False,
                    use_deterministic_grid=None,
                    threshold=None,  # 0.02,
                    compute_kl_loss=False,
                    evaluate_train=False,
                    eval_datasplit_type=eval_datasplit_type,
                    val_repeat_factor=None,
                    psnr_type="range_invariant",
                    ignored_last_pixels=ignored_last_pixels,
                    ignore_first_pixels=0,
                    print_token=handler.dirpath(),
                    normalized_ssim=normalized_ssim,
                    predict_kth_frame=predict_kth_frame,
                    predict_samples_N=predict_samples_N,
                    compare_with_highsnr=not skip_highsnr,
                    train_calibration=train_calibration,
                    # trim_boundary=trim_boundary,
                    eval_calibration_factors=eval_calibration_factors,
                    override_kwargs=override_kwargs,
                )
                if data is None:
                    return None, None

                fpath = handler.save(ckpt_dir, data)
                # except:
                #     print('FAILED for ', handler.get_output_fpath(ckpt_dir))
                #     continue
                # print(handler.load(fpath))
                print("")
                print("")
                print("")
                if save_prediction and prediction is not None:
                    if isinstance(prediction, list):
                        prediction = stack_list_of_images(prediction)

                    handler.dump_predictions(
                        ckpt_dir,
                        prediction,
                        {},
                        overwrite=overwrite_saved_predictions,
                    )

    return data, prediction

def parse_grid_size(grid_size):
        grid_size = list(map(int,grid_size.split(',')))
        if len(grid_size) == 1:
            grid_size = grid_size[0]
        else:
            grid_size = tuple(grid_size)
        return grid_size
    
if __name__ == "__main__":
    import os

    import numpy as np

    # shape1 = (1, 1800, 1500, 2)
    # shape2 = (1, 1000, 1300, 2)
    # gt = [np.random.rand(*shape1), np.random.rand(*shape2)]
    # combined = stack_list_of_images(gt)
    # pred = [np.random.rand(*shape)]
    # gt_list, pred_list = _get_list_of_images_from_gt_pred(gt, pred, 0)
    # os.environ['TQDM_DISABLE']='1'

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--grid_size", type=parse_grid_size, default=32)
    parser.add_argument("--normalized_ssim", action="store_true")
    parser.add_argument("--save_prediction", action="store_true")
    parser.add_argument("--save_prediction_factor", type=float, default=1.0)
    parser.add_argument("--mmse_count", type=int, default=1)
    parser.add_argument("--predict_kth_frame", type=int, default=None)
    parser.add_argument("--preserve_older_prediction", action="store_true")
    parser.add_argument("--predict_samples_N", type=int, default=None)
    parser.add_argument("--skip_highsnr", action="store_true")
    parser.add_argument("--train_calibration", action="store_true")
    parser.add_argument("--eval_calibration", action="store_true")
    parser.add_argument("--override_kwargs", type=str, default=None)
    # parser.add_argument("--donot_trim_boundary", action="store_true")

    args = parser.parse_args()
    
    save_hardcoded_ckpt_evaluations_to_file(
        normalized_ssim=args.normalized_ssim,
        save_prediction=args.save_prediction,
        mmse_count=args.mmse_count,
        predict_kth_frame=args.predict_kth_frame,
        ckpt_dir=args.ckpt_dir,
        patch_size=args.patch_size,
        grid_size=args.grid_size,
        overwrite_saved_predictions=not args.preserve_older_prediction,
        predict_samples_N=args.predict_samples_N,
        save_prediction_factor=args.save_prediction_factor,
        skip_highsnr=args.skip_highsnr,
        train_calibration=args.train_calibration,
        eval_calibration=args.eval_calibration,
        override_kwargs=args.override_kwargs,
        # trim_boundary=not args.donot_trim_boundary,
    )
