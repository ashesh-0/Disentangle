"""
Configuration file for the VAE model with critic
"""
import ml_collections
from disentangle.configs.default_config import get_default_config
from disentangle.core.data_type import DataType
from disentangle.core.loss_type import LossType
from disentangle.core.model_type import ModelType
from disentangle.core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 128
    data.data_type = DataType.OptiMEM100_014
    data.channel_1 = 0
    data.channel_2 = 2
    data.sampler_type = SamplerType.DefaultSampler
    data.threshold = 0.02
    data.deterministic_grid = True
    data.normalized_input = True

    loss = config.loss
    loss.loss_type = LossType.ElboWithCritic
    loss.kl_weight = 0.01
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 0.0
    loss.critic_loss_weight = 0.01

    model = config.model
    model.model_type = ModelType.LadderVAECritic
    model.z_dims = [128, 128, 128]
    model.blocks_per_layer = 5
    model.nonlin = 'elu'
    model.merge_type = 'residual'
    model.batchnorm = True
    model.stochastic_skip = True
    model.n_filters = 64
    model.dropout = 0.2
    model.learn_top_prior = True
    model.img_shape = None
    model.res_block_type = 'bacdbacd'
    model.gated = True
    model.no_initial_downscaling = True
    model.analytical_kl = True
    model.mode_pred = False
    model.var_clip_max = 5
    # Discriminator params
    model.critic = ml_collections.ConfigDict()
    model.critic.ndf = 64
    model.critic.netD = 'n_layers'
    model.critic.layers_D = 4
    model.critic.norm = 'batch'

    training = config.training
    training.lr = 0.001
    training.lr_scheduler_patience = 1000
    training.max_epochs = 20000
    training.batch_size = 8
    training.num_workers = 4
    training.val_repeat_factor = 50
    training.train_repeat_factor = 10
    training.val_fraction = 0.2
    config.training.earlystop_patience = 5000
    return config
