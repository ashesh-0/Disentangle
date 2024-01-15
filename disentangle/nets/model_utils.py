import glob
import os
import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn

from disentangle.config_utils import get_updated_config
from disentangle.core.data_type import DataType
from disentangle.core.loss_type import LossType
from disentangle.core.model_type import ModelType
from disentangle.nets.brave_net import BraveNetPL
from disentangle.nets.denoiser_splitter import DenoiserSplitter
from disentangle.nets.lvae import LadderVAE
from disentangle.nets.lvae_bleedthrough import LadderVAEWithMixedRecons
from disentangle.nets.lvae_deepencoder import LVAEWithDeepEncoder
from disentangle.nets.lvae_denoiser import LadderVAEDenoiser
from disentangle.nets.lvae_multidset_multi_input_branches import LadderVaeMultiDatasetMultiBranch
from disentangle.nets.lvae_multidset_multi_optim import LadderVaeMultiDatasetMultiOptim
from disentangle.nets.lvae_multiple_encoder_single_opt import LadderVAEMulEncoder1Optim
from disentangle.nets.lvae_multiple_encoders import LadderVAEMultipleEncoders
from disentangle.nets.lvae_multires_target import LadderVAEMultiTarget
from disentangle.nets.lvae_restricted_reconstruction import LadderVAERestrictedReconstruction
from disentangle.nets.lvae_semi_supervised import LadderVAESemiSupervised
from disentangle.nets.lvae_twindecoder import LadderVAETwinDecoder
from disentangle.nets.lvae_twodset import LadderVaeTwoDset
from disentangle.nets.lvae_twodset_restrictedrecons import LadderVaeTwoDsetRestrictedRecons
from disentangle.nets.lvae_with_critic import LadderVAECritic
from disentangle.nets.lvae_with_stitch import LadderVAEwithStitching
from disentangle.nets.lvae_with_stitch_2stage import LadderVAEwithStitching2Stage
from disentangle.nets.splitter_denoiser import SplitterDenoiser
from disentangle.nets.unet import UNet


def create_model(config, data_mean, data_std, val_idx_manager=None):
    if config.model.model_type == ModelType.LadderVae:
        target_ch = config.data.get('num_channels', 2)
        model = LadderVAE(data_mean, data_std, config, target_ch=target_ch, val_idx_manager=val_idx_manager)
    elif config.model.model_type == ModelType.LadderVaeTwinDecoder:
        model = LadderVAETwinDecoder(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVAECritic:
        model = LadderVAECritic(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeSepEncoder:
        model = LadderVAEMultipleEncoders(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVAEMultiTarget:
        model = LadderVAEMultiTarget(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeSepEncoderSingleOptim:
        model = LadderVAEMulEncoder1Optim(data_mean, data_std, config)
    elif config.model.model_type == ModelType.UNet:
        model = UNet(data_mean, data_std, config)
    elif config.model.model_type == ModelType.BraveNet:
        model = BraveNetPL(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeStitch:
        model = LadderVAEwithStitching(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeMixedRecons:
        model = LadderVAEWithMixedRecons(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeSemiSupervised:
        model = LadderVAESemiSupervised(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeStitch2Stage:
        model = LadderVAEwithStitching2Stage(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeTwoDataSet:
        model = LadderVaeTwoDset(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeTwoDatasetMultiBranch:
        model = LadderVaeMultiDatasetMultiBranch(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeTwoDatasetMultiOptim:
        model = LadderVaeMultiDatasetMultiOptim(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LVaeDeepEncoderIntensityAug:
        model = LVAEWithDeepEncoder(data_mean, data_std, config)
    elif config.model.model_type == ModelType.Denoiser:
        model = LadderVAEDenoiser(data_mean, data_std, config)
    elif config.model.model_type == ModelType.DenoiserSplitter:
        model = DenoiserSplitter(data_mean, data_std, config)
    elif config.model.model_type == ModelType.SplitterDenoiser:
        model = SplitterDenoiser(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVAERestrictedReconstruction:
        model = LadderVAERestrictedReconstruction(data_mean, data_std, config, val_idx_manager=val_idx_manager)
    elif config.model.model_type == ModelType.LadderVAETwoDataSetRestRecon:
        model = LadderVaeTwoDsetRestrictedRecons(data_mean, data_std, config)
    else:
        raise Exception('Invalid model type:', config.model.model_type)
    return model


def get_best_checkpoint(ckpt_dir):
    output = []
    for filename in glob.glob(ckpt_dir + "/*_best.ckpt"):
        output.append(filename)
    assert len(output) == 1, '\n'.join(output)
    return output[0]


def load_model_checkpoint(ckpt_dir: str,
                          data_mean: float,
                          data_std: float,
                          config=None,
                          model=None) -> pl.LightningModule:
    """
    It loads the model from the checkpoint directory
    """
    import ml_collections  # Needed due to loading in pickle
    if model is None:
        # load config, if the config is not provided
        if config is None:
            with open(os.path.join(ckpt_dir, 'config.pkl'), 'rb') as f:
                config = pickle.load(f)

        config = get_updated_config(config)
        model = create_model(config, data_mean, data_std)
    ckpt_fpath = get_best_checkpoint(ckpt_dir)
    checkpoint = torch.load(ckpt_fpath)
    _ = model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model from ckpt dir', ckpt_dir, f' at epoch:{checkpoint["epoch"]}')
    return model
