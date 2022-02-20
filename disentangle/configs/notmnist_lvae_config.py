from disentangle.configs.default_config import get_default_config
from disentangle.core.data_type import DataType
from disentangle.core.loss_type import LossType
from disentangle.core.model_type import ModelType
from disentangle.core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.data_type = DataType.NotMNIST
    data.img_dsample = 1
    data.image_size = 28 // data.img_dsample
    data.label1 = 'A'
    data.label2 = 'B'
    data.sampler_type = SamplerType.SingleImgSampler
    data.mean_val = 44.8525
    data.std_val = 73.395
    data.return_img_labels = False

    loss = config.loss
    loss.loss_type = LossType.Elbo
    loss.kl_weight = 1
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 0.0

    model = config.model
    model.model_type = ModelType.LadderVae
    model.z_dims = [128, 128, 128]
    model.blocks_per_layer = 3
    model.nonlin = 'elu'
    model.merge_type = 'residual'
    model.batchnorm = True
    model.stochastic_skip = True
    model.n_filters = 64
    model.dropout = 0.0
    model.learn_top_prior = True
    model.img_shape = None
    model.res_block_type = 'bacdbacd'
    model.gated = True
    model.no_initial_downscaling = True
    model.analytical_kl = True
    model.mode_pred = False

    training = config.training
    training.lr = 0.001
    training.lr_scheduler_patience = 30
    training.max_epochs = 1000
    training.batch_size = 16
    training.num_workers = 4
    return config
