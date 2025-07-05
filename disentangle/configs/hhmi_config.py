from disentangle.configs.default_config import get_default_config
from disentangle.core.data_type import DataType
from disentangle.core.loss_type import LossType
from disentangle.core.model_type import ModelType
from disentangle.core.sampler_type import SamplerType
from disentangle.data_loader.nikola_7D_rawdata_loader import NikolaChannelList


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 64
    data.data_type = DataType.MultiTiffSameSizeDset
    # Dataset V1:Ch5 (peroxisomes), Ch0 (mitochondria), Ch3 (nuclei), Ch2 (lysosomes), Ch1 (cell boundary), and Ch4 (lipid droplets).
    # Dataset V2:Ch0 (mitochondria), Ch1 (cell boundary), Ch2 (lysosomes), Ch3 (nuclei), Ch4 (lipid droplets), and Ch5 (peroxisomes).
    # For Dataset V1, we need: [4,3,5] & [1,3,2]
    # In Dataset V2 we have used: [1,2,4] & [0,2,3]
    data.channel_idx_list = [4,3,5]
    data.train_fnames = ['Composite_region3_6am_CF_L3L2.tif', 
                        'Composite_region5_6am_L3L1.tif',
                        'Composite_region2_6AM_CF_L3L3.tif',  
                        'Composite_region4_6am_L3L2.tif'
                        ]
    data.val_fnames = ['Composite_region1_CF_L3L3.tif']
    data.test_fnames = ['Composite_region7_CF_12noon_L3L4.tif'
    ]
    data.depth3D = 5
    data.mode_3D = True


    # data.trainig_datausage_fraction = 0.1
    data.num_channels = len(data.channel_idx_list)
    data.target_idx_list = list(range(len(data.channel_idx_list)))

    data.enable_gaussian_noise = False
    # data.trainig_datausage_fraction = 0.02
    data.poisson_noise_factor = -1
    # data.validtarget_random_fraction = 1.0
    # data.training_validtarget_fraction = 0.2
    config.data.synthetic_gaussian_scale = 100
    # if True, then input has 'identical' noise as the target. Otherwise, noise of input is independently sampled.
    config.data.input_has_dependant_noise = True

    data.sampler_type = SamplerType.DefaultSampler
    data.threshold = 0.02
    # data.grid_size = 1
    data.deterministic_grid = False
    data.normalized_input = True
    data.clip_percentile = 1

    data.channelwise_quantile = False
    # If this is set to true, then one mean and stdev is used for both channels. Otherwise, two different
    # meean and stdev are used.
    data.use_one_mu_std = True
    data.train_aug_rotate = True
    data.randomized_channels = False
    data.multiscale_lowres_count = None
    data.padding_mode = 'reflect'
    data.padding_value = None
    # If this is set to True, then target channels will be normalized from their separate mean.
    # otherwise, target will be normalized just the same way as the input, which is determined by use_one_mu_std
    data.target_separate_normalization = True
    data.input_is_sum = False
    loss = config.loss
    loss.loss_type = LossType.DenoiSplitMuSplit
    # this is not uSplit.
    loss.kl_loss_formulation = 'denoisplit_usplit'
    loss.restricted_kl = False

    # loss.mixed_rec_weight = 1
    loss.usplit_w = 0.1
    loss.denoisplit_w = 1 - loss.usplit_w

    loss.kl_weight = 1.0
    loss.reconstruction_weight = 1.0
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 1.0

    model = config.model
    model.model_type = ModelType.LadderVae
    model.mode_3D = data.mode_3D
    model.decoder.mode_3D = model.mode_3D

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

    #False
    config.model.decoder.conv2d_bias = True

    model.skip_nboundary_pixels_from_loss = None
    model.num_targets = len(data.target_idx_list)
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
    model.predict_logvar = 'pixelwise'  #'channelwise'
    model.logvar_lowerbound = -5  # -2.49 is log(1/12), from paper "Re-parametrizing VAE for stablity."
    model.multiscale_lowres_separate_branch = False
    model.multiscale_retain_spatial_dims = True
    model.monitor = 'val_loss'  # {'val_loss','val_psnr'}

    model.enable_noise_model = True
    model.noise_model_type = 'gmm'
    # model.noise_model_ch1_fpath = '/group/jug/ashesh/training/noise_model/2505/2/GMMNoiseModel_n2v_denoising-subset_Ch0__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    # model.noise_model_ch2_fpath = '/group/jug/ashesh/training/noise_model/2505/3/GMMNoiseModel_n2v_denoising-subset_Ch1__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    # model.noise_model_ch3_fpath = '/group/jug/ashesh/training/noise_model/2505/4/GMMNoiseModel_n2v_denoising-subset_Ch2__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    # model.noise_model_ch4_fpath = '/group/jug/ashesh/training/noise_model/2505/5/GMMNoiseModel_n2v_denoising-subset_Ch3__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    # model.noise_model_ch5_fpath = '/group/jug/ashesh/training/noise_model/2505/6/GMMNoiseModel_n2v_denoising-subset_Ch4__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    # model.noise_model_ch6_fpath = '/group/jug/ashesh/training/noise_model/2505/7/GMMNoiseModel_n2v_denoising-subset_Ch5__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'

    # model.noise_model_ch1_fpath = '/group/jug/ashesh/training/noise_model/2505/3/GMMNoiseModel_n2v_denoising-subset_Ch1__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    # model.noise_model_ch2_fpath = '/group/jug/ashesh/training/noise_model/2505/5/GMMNoiseModel_n2v_denoising-subset_Ch3__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    # model.noise_model_ch3_fpath = '/group/jug/ashesh/training/noise_model/2505/4/GMMNoiseModel_n2v_denoising-subset_Ch2__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'

    model.noise_model_ch1_fpath = '/group/jug/ashesh/training/noise_model/2505/6/GMMNoiseModel_n2v_denoising-subset_Ch4__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    model.noise_model_ch2_fpath = '/group/jug/ashesh/training/noise_model/2505/5/GMMNoiseModel_n2v_denoising-subset_Ch3__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    model.noise_model_ch3_fpath = '/group/jug/ashesh/training/noise_model/2505/7/GMMNoiseModel_n2v_denoising-subset_Ch5__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'



    model.noise_model_learnable = False

    # model.noise_model_ch1_fpath = fname_format.format('2307/58', 'actin')
    # model.noise_model_ch2_fpath = fname_format.format('2307/59', 'mito')
    model.non_stochastic_version = False

    training = config.training
    training.lr = 0.001
    training.lr_scheduler_patience = 30
    training.max_epochs = 400
    training.batch_size = 16
    training.num_workers = 4
    training.val_repeat_factor = None
    training.train_repeat_factor = None
    training.val_fraction = 0.1
    training.test_fraction = 0.1
    training.earlystop_patience = 200
    training.precision = 16
    training.limit_train_batches = 2000
    return config
