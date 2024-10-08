import math

from disentangle.configs.default_config import get_default_config
from disentangle.core.data_type import DataType
from disentangle.core.loss_type import LossType
from disentangle.core.model_type import ModelType
from disentangle.core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 128
    data.frame_size = 128
    data.data_type = DataType.CustomSinosoidThreeCurve
    data.total_size = 1000
    data.curve_amplitude = 8.0
    data.num_curves = 5
    data.max_rotation = 0.0
    data.curve_thickness = 21
    data.max_vshift_factor = 0.9
    data.max_hshift_factor = 0.3
    data.frequency_range_list = [(0.05, 0.07), (0.12, 0.14), (0.3, 0.32), (0.6, 0.62)]

    data.sampler_type = SamplerType.DefaultSampler
    data.deterministic_grid = False
    data.normalized_input = True
    # If this is set to true, then one mean and stdev is used for both channels. If False, two different
    # meean and stdev are used. If None, 0 mean and 1 std is used.
    data.use_one_mu_std = True
    data.train_aug_rotate = False
    data.randomized_channels = False
    data.multiscale_lowres_count = None
    data.padding_mode = 'constant'
    data.padding_value = 0
    data.encourage_non_overlap_single_channel = True
    data.vertical_min_spacing = data.curve_amplitude * 2
    # 0.5 would mean that 50% of the points would be covered with the connecting w.
    data.connecting_w_len = 0.2
    data.curve_initial_phase = 0.0
    data.target_separate_normalization = True

    loss = config.loss
    loss.loss_type = LossType.Elbo
    # loss.mixed_rec_weight = 1

    loss.kl_weight = 1
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 0.0

    model = config.model
    model.model_type = ModelType.LadderVae
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
    #True

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
    # predict_logvar takes one of the three values: [None,'global','channelwise','pixelwise']
    model.predict_logvar = 'pixelwise'
    model.logvar_lowerbound = -5  # -2.49 is log(1/12), from paper "Re-parametrizing VAE for stablity."
    model.multiscale_lowres_separate_branch = False
    model.multiscale_retain_spatial_dims = True
    model.monitor = 'val_psnr'  # {'val_loss','val_psnr'}

    training = config.training
    training.lr = 0.001
    training.lr_scheduler_patience = 90
    training.max_epochs = 2400
    training.batch_size = 32
    training.num_workers = 4
    training.val_repeat_factor = None
    training.train_repeat_factor = None
    training.val_fraction = 0.1
    training.test_fraction = 0.1
    training.earlystop_patience = 300
    training.precision = 16

    return config
