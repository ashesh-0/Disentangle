from tkinter.tix import Tree

import numpy as np

from disentangle.configs.default_config import get_default_config
from disentangle.core.data_type import DataType
from disentangle.core.loss_type import LossType
from disentangle.core.model_type import ModelType
from disentangle.core.sampler_type import SamplerType
from disentangle.data_loader.ht_iba1_ki67_rawdata_loader import SubDsetType


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 64
    data.data_type = DataType.HTIba1Ki67
    data.subdset_type = None
    data.validation_subdset_type_idx = 0

    data.sampler_type = SamplerType.DefaultSampler
    data.deterministic_grid = False
    data.normalized_input = True
    data.clip_percentile = 0.995
    data.background_quantile = 0.01

    # If this is set to true, then one mean and stdev is used for both channels while computing input.
    # Otherwise, two different meean and stdev are used.
    data.use_one_mu_std = True
    data.train_aug_rotate = False
    data.randomized_channels = False
    data.multiscale_lowres_count = None
    data.padding_mode = 'reflect'
    data.padding_value = None
    # If this is set to True, then target channels will be normalized from their separate mean.
    # otherwise, target will be normalized just the same way as the input, which is determined by use_one_mu_std
    data.target_separate_normalization = True

    # Replacing one channel's content with empty patch.
    data.empty_patch_replacement_enabled = False
    data.empty_patch_replacement_channel_idx = 0
    data.empty_patch_replacement_probab = 0.5
    data.empty_patch_max_val_threshold = 180

    loss = config.loss
    loss.loss_type = LossType.ElboMixedReconstruction
    loss.mixed_rec_weight = 1

    loss.kl_weight = 1
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 0.0

    model = config.model
    model.model_type = ModelType.LadderVaeMultiDataSet
    model.z_dims = [128, 128, 128, 128]

    model.encoder.batchnorm = True
    model.encoder.blocks_per_layer = 1
    model.encoder.n_filters = 64
    model.encoder.dropout = 0.1
    model.encoder.res_block_kernel = 3
    model.encoder.res_block_skip_padding = False

    model.decoder.batchnorm = True
    model.decoder.blocks_per_layer = 1
    model.decoder.n_filters = 64
    model.decoder.dropout = 0.1
    model.decoder.res_block_kernel = 3
    model.decoder.res_block_skip_padding = False

    model.decoder.multiscale_retain_spatial_dims = True

    model.skip_nboundary_pixels_from_loss = None
    model.nonlin = 'elu'
    model.merge_type = 'residual'
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
    model.predict_logvar = 'pixelwise'
    model.logvar_lowerbound = -5  # -2.49 is log(1/12), from paper "Re-parametrizing VAE for stablity."
    model.multiscale_lowres_separate_branch = False
    model.multiscale_retain_spatial_dims = True
    model.monitor = 'val_psnr'  # {'val_loss','val_psnr'}
    model.non_stochastic_version = False

    model.enable_learnable_interchannel_weights = True

    training = config.training
    training.lr = 0.001
    training.lr_scheduler_patience = 15
    training.max_epochs = 200
    training.batch_size = 32
    training.num_workers = 4
    training.val_repeat_factor = None
    training.train_repeat_factor = None
    # training.val_fraction = 0.0
    # training.test_fraction = 0.0
    training.earlystop_patience = 100
    training.precision = 16

    # when working with multi datasets, it might make sense to predict the mixing constants. This will be applied to
    # dataset which will have mixed reconstruction as loss
    data.subdset_types = [SubDsetType.OnlyIba1, SubDsetType.Iba1Ki64]
    data.subdset_types_probab = [0.7, 0.3]
    data.empty_patch_replacement_enabled_list = [True, False]
    training.test_fraction = [0, 0.2]
    training.val_fraction = [0.2, 0]
    data.input_is_sum_list = [True, False]
    data.input_is_sum = False
    return config
