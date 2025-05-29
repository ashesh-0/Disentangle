import numpy as np
import torch
import torch.nn as nn

import ml_collections
from disentangle.core.data_type import DataType
from disentangle.core.loss_type import LossType
from disentangle.core.model_type import ModelType
from disentangle.core.sampler_type import SamplerType
from disentangle.nets.lvae import LadderVAE
from finetunesplit.asymmetric_transforms import TransformEnum


def get_vae_config(input_channels=2, z_dim=None):
    config = ml_collections.ConfigDict()
    config.training = ml_collections.ConfigDict()
    config.training.lr = 1e-3
    config.training.lr_scheduler_patience = 10
    config.training.val_fraction = 0.1
    config.training.test_fraction = 0.1

    config.loss = ml_collections.ConfigDict()
    config.loss.loss_type = LossType.Elbo
    # config.loss.usplit_w = 0.1
    # config.loss.denoisplit_w = 1 - config.loss.usplit_w
    config.loss.kl_loss_formulation = 'denoisplit'
    # config.loss.mixed_rec_weight = 1
    config.loss.restricted_kl = False
    config.loss.kl_weight = 1.0
    config.loss.reconstruction_weight = 1.0
    config.loss.kl_annealing = False
    config.loss.kl_annealtime = 10
    config.loss.kl_start = -1
    config.loss.kl_min = 1e-7
    config.loss.free_bits = 1.0


    config.data = ml_collections.ConfigDict()
    config.data.input_is_sum = False
    config.data.image_size = 28
    config.data.normalized_input = True
    
    # input has two channels.
    config.data.color_ch = input_channels
    config.data.multiscale_lowres_count = None

    # for loading MNIST dataset
    config.data.data_type = DataType.MNIST
    config.data.num_channels = 2
    config.data.sampler_type = SamplerType.DefaultSampler
    config.data.ch0_labels_list = [0, 1]
    config.data.ch1_labels_list = [3,4]
    config.data.ch0_transforms_params = [{'name':TransformEnum.PatchShuffle,'patch_size':28, 'grid_size':14}]
    config.data.ch1_transforms_params = [{'name':TransformEnum.Translate,'max_fraction':1.0}]







    config.model = ml_collections.ConfigDict()
    config.model.encoder = ml_collections.ConfigDict()
    config.model.decoder = ml_collections.ConfigDict()
    config.model.model_type = ModelType.LadderVae
    if z_dim is not None:
        config.model.z_dims = [z_dim]*4
    else:
        config.model.z_dims = [8,8,8,8]

    config.model.encoder.batchnorm = True
    config.model.encoder.blocks_per_layer = 1
    config.model.encoder.n_filters = 64
    config.model.encoder.dropout = 0.1
    config.model.encoder.res_block_kernel = 3
    config.model.encoder.res_block_skip_padding = False
    config.model.decoder.batchnorm = True
    config.model.decoder.blocks_per_layer = 1
    config.model.decoder.n_filters = 64
    config.model.decoder.dropout = 0.1
    config.model.decoder.res_block_kernel = 3
    config.model.decoder.res_block_skip_padding = False

    config.model.decoder.conv2d_bias = True

    config.model.skip_nboundary_pixels_from_loss = None
    config.model.nonlin = 'elu'
    config.model.merge_type = 'residual'
    config.model.stochastic_skip = True
    config.model.learn_top_prior = True
    config.model.img_shape = None
    config.model.res_block_type = 'bacdbacd'

    config.model.gated = True
    config.model.no_initial_downscaling = True
    config.model.analytical_kl = False
    config.model.mode_pred = False
    config.model.var_clip_max = 20
    # predict_logvar takes one of the four values: [None,'global','channelwise','pixelwise']
    config.model.predict_logvar = None  #'pixelwise' #'channelwise'
    config.model.logvar_lowerbound = -5  # -2.49 is log(1/12), from paper "Re-parametrizing VAE for stablity."
    config.model.multiscale_lowres_separate_branch = False
    config.model.multiscale_retain_spatial_dims = True
    config.model.monitor = 'val_psnr'  # {'val_loss','val_psnr'}
    config.model.enable_noise_model = False


    config.model.non_stochastic_version = True
    config.model.skip_bottomk_buvalues = len(config.model.z_dims) - 1
    return config

def get_vae(input_channels=2, pretrained_weights_fpath=None, z_dim=None):
    config = get_vae_config(input_channels, z_dim=z_dim)
    data_mean = {'target': np.array([0.0]), 'input':np.array([0.0])} 
    data_std = {'target': np.array([1.0]), 'input':np.array([1.0])}

    class LVAEForward(LadderVAE):
        def get_embedding(self, x):
            _, td_data = self.forward(x)
            embedding = td_data['z'][-1]
            return embedding
        
    model = LVAEForward(data_mean, data_std, config, target_ch=input_channels)
    print(model)
    if pretrained_weights_fpath is not None:
        model.load_state_dict(torch.load(pretrained_weights_fpath))
        print(f"Loaded pretrained weights from {pretrained_weights_fpath}")
    
    return model, config.model.z_dims[-1]
