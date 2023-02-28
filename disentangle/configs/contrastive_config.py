from tkinter.tix import Tree

import numpy as np

from disentangle.configs.default_config import get_default_config
from disentangle.core.data_type import DataType
from disentangle.core.loss_type import LossType
from disentangle.core.model_type import ModelType
from disentangle.core.sampler_type import SamplerType
from disentangle.data_loader.pavia2_enums import Pavia2DataSetChannels


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 128
    data.data_type = DataType.OptiMEM100_014
    data.channel_1 = 0
    data.channel_2 = 2

    data.ch1_min_alpha = 0.02
    data.ch1_max_alpha = 0.98
    data.ch1_alpha_interval_count = 20
    # data.channel_2_downscale_factor = 1

    data.sampler_type = SamplerType.ContrastiveSampler
    data.threshold = 0.02
    data.deterministic_grid = True
    data.normalized_input = True
    data.clip_percentile = 0.995
    # If this is set to true, then one mean and stdev is used for both channels. Otherwise, two different
    # meean and stdev are used.
    data.use_one_mu_std = False
    data.train_aug_rotate = False
    data.randomized_channels = False
    data.multiscale_lowres_count = None
    data.padding_mode = 'reflect'
    data.padding_value = None
    # If this is set to True, then target channels will be normalized from their separate mean.
    # otherwise, target will be normalized just the same way as the input, which is determined by use_one_mu_std
    data.target_separate_normalization = True

    loss = config.loss
    loss.loss_type = LossType.Elbo
    loss.cl_tau_pos = 0.005
    loss.cl_tau_neg = 0.5
    loss.cl_weight = 1
    # loss.mixed_rec_weight = 1

    loss.kl_weight = 1
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 0.0
    loss.lres_recloss_w = [0.4, 0.2, 0.2, 0.2]
    loss.skip_cl_on_alpha = False

    model = config.model
    model.model_type = ModelType.LadderVaeCL
    model.z_dims = [128, 128, 128, 128]

    model.encoder.blocks_per_layer = 1
    model.encoder.n_filters = 64
    model.encoder.dropout = 0.1
    model.encoder.res_block_kernel = 3
    model.encoder.res_block_skip_padding = False

    model.decoder.blocks_per_layer = 1
    model.decoder.n_filters = 64
    model.decoder.dropout = 0.1
    model.decoder.res_block_kernel = 3
    model.decoder.res_block_skip_padding = False
    model.decoder.multiscale_retain_spatial_dims = True

    model.skip_nboundary_pixels_from_loss = None
    model.nonlin = 'elu'
    model.merge_type = 'residual'
    model.batchnorm = True
    model.stochastic_skip = True
    model.learn_top_prior = True
    model.img_shape = None
    model.res_block_type = 'bacdbacd'

    model.gated = True
    model.no_initial_downscaling = True
    model.analytical_kl = False
    model.mode_pred = False
    model.var_clip_max = 20
    # predict_logvar takes one of the four values: [None,'global','channelwise','pixelwise']
    model.predict_logvar = 'global'
    model.logvar_lowerbound = -5  # -2.49 is log(1/12), from paper "Re-parametrizing VAE for stablity."
    model.multiscale_lowres_separate_branch = False
    model.multiscale_retain_spatial_dims = True
    model.monitor = 'val_psnr'  # {'val_loss','val_psnr'}
    model.non_stochastic_version = False
    model.cl_latent_start_end_alpha = (0, 2)
    diff = model.z_dims[0] - model.cl_latent_start_end_alpha[1]
    model.cl_latent_start_end_ch1 = (model.cl_latent_start_end_alpha[1], model.cl_latent_start_end_alpha[1] + diff // 2)
    model.cl_latent_start_end_ch2 = (model.cl_latent_start_end_ch1[1], model.z_dims[0])

    training = config.training
    training.lr = 0.001
    training.lr_scheduler_patience = 30
    training.max_epochs = 400
    training.batch_size = 32
    training.num_workers = 4
    training.val_repeat_factor = None
    training.train_repeat_factor = None
    training.val_fraction = 0.1
    training.test_fraction = 0.1
    training.earlystop_patience = 200
    training.precision = 16

    return config
