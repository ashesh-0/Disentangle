from disentangle.configs.default_config import get_default_config
from disentangle.core.data_type import DataType
from disentangle.core.loss_type import LossType
from disentangle.core.model_type import ModelType
from disentangle.core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 128
    data.data_type = DataType.BioSR_MRC
    # data.channel_1 = 0
    # data.channel_2 = 1
    data.ch1_fname = 'ER/GT_all.mrc'
    data.ch2_fname = 'Microtubules/GT_all.mrc'

    data.trainig_datausage_fraction = 0.03
    # data.training_validtarget_fraction = 0.03
    # data.validtarget_random_fraction = 0.7
    # data.validtarget_random_fraction_final = 0.9
    # data.validtarget_random_fraction_stepepoch = 0.005

    data.sampler_type = SamplerType.DefaultSampler
    data.threshold = 0.02
    data.deterministic_grid = False
    data.normalized_input = True
    data.input_is_sum = False
    data.clip_percentile = 0.995

    data.channelwise_quantile = True
    # If this is set to true, then one mean and stdev is used for both channels. Otherwise, two different
    # meean and stdev are used.
    data.use_one_mu_std = True
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
    loss.mixed_rec_weight = 1

    loss.kl_weight = 1
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 0.0
    # loss.ch1_recons_w = 1
    # loss.ch2_recons_w = 5

    model = config.model
    model.model_type = ModelType.LadderVae
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
    model.decoder.conv2d_bias = True
    model.reconstruction_mode = True

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
    # predict_logvar takes one of the four values: [None,'global','channelwise','pixelwise', 'ch_invariant_pixelwise]
    model.predict_logvar = None
    model.logvar_lowerbound = -5  # -2.49 is log(1/12), from paper "Re-parametrizing VAE for stablity."
    model.multiscale_lowres_separate_branch = False
    model.multiscale_retain_spatial_dims = True
    model.monitor = 'val_psnr'  # {'val_loss','val_psnr'}
    model.non_stochastic_version = False
    model.enable_noise_model = False
    model.noise_model_ch1_fpath = None
    model.noise_model_ch1_fpath = None
    # model.pretrained_weights_path = '/home/ubuntu/ashesh/training/disentangle/2310/D7-M3-S0-L0/2/BaselineVAECL_best.ckpt'

    training = config.training
    training.lr = 0.001 / 2
    training.lr_scheduler_patience = int(30 / data.trainig_datausage_fraction)
    training.max_epochs = int(200 / data.trainig_datausage_fraction)
    training.batch_size = 32
    training.num_workers = 1
    training.val_repeat_factor = None
    training.train_repeat_factor = None
    training.val_fraction = 0.1
    training.test_fraction = 0.1

    training.earlystop_patience = int(100 / data.trainig_datausage_fraction)
    training.precision = 16
    training.check_val_every_n_epoch = int(1 / (data.trainig_datausage_fraction * 2))

    return config
