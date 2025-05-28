from disentangle.configs.default_config import get_default_config
from disentangle.core.data_type import DataType
from disentangle.core.loss_type import LossType
from disentangle.core.model_type import ModelType
from disentangle.core.sampler_type import SamplerType
from finetunesplit.asymmetric_transforms import TransformEnum


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 28
    data.data_type = DataType.MNIST
    data.num_channels = 2
    data.sampler_type = SamplerType.DefaultSampler
    data.input_is_sum = False
    data.normalized_input = True
    data.multiscale_lowres_count = None
    
    data.ch0_labels_list = [0, 1]
    data.ch1_labels_list = [3,4]
    # data.ch0_transforms_params = [{'name':TransformEnum.PatchShuffle,'patch_size':28, 'grid_size':14}]
    data.ch0_transforms_params = [{'name':TransformEnum.Identity,},{'name': TransformEnum.Scale, 'min_scale': 0.2, 'max_scale': 1}]
    data.ch1_transforms_params = [{'name':TransformEnum.Translate,'max_fraction':1.0},]
    # data.ch1_transforms_params = [{'name':TransformEnum.Identity,}]
    # data.ch1_transforms_params = [{'name':TransformEnum.DeepInV, 'aug_theta_max':10,'aug_theta_z_max':90,'aug_shift_max':0.0, 'padding': 'zeros'},
    #                               {'name': TransformEnum.HFlip}, {'name': TransformEnum.VFlip}]
    # data.ch2_transforms_params = [{'name':TransformEnum.PatchShuffle,'patch_size':28, 'grid_size':14}]


    loss = config.loss
    loss.loss_type = LossType.Elbo
    # this is not uSplit.
    loss.kl_loss_formulation = 'usplit'
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
    model.z_dims = [128]

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
    model.num_targets = data.num_channels
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

    model.enable_noise_model = False
    model.non_stochastic_version = True

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
    training.limit_train_batches = 2000
    return config
