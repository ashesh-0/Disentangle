import glob
import logging
import os
import pickle

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from disentangle.core.data_split_type import DataSplitType
from disentangle.core.data_type import DataType
from disentangle.core.loss_type import LossType
from disentangle.core.metric_monitor import MetricMonitor
from disentangle.core.model_type import ModelType
from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from disentangle.data_loader.multi_channel_determ_tiff_dloader_randomized import MultiChDeterministicTiffRandDloader
from disentangle.data_loader.multi_channel_tiff_dloader import MultiChTiffDloader
from disentangle.data_loader.multiscale_mc_tiff_dloader import MultiScaleTiffDloader
from disentangle.data_loader.notmnist_dloader import NotMNISTNoisyLoader
from disentangle.data_loader.pavia2_3ch_dloader import Pavia2ThreeChannelDloader
from disentangle.data_loader.places_dloader import PlacesLoader
from disentangle.data_loader.semi_supervised_dloader import SemiSupDloader
from disentangle.data_loader.single_channel.multi_dataset_dloader import SingleChannelMultiDatasetDloader
from disentangle.nets.model_utils import create_model
from disentangle.training_utils import ValEveryNSteps


def create_dataset(config, datadir, raw_data_dict=None, skip_train_dataset=False):
    if config.data.data_type == DataType.NotMNIST:
        train_img_files_pkl = os.path.join(datadir, 'train_fnames.pkl')
        val_img_files_pkl = os.path.join(datadir, 'val_fnames.pkl')

        datapath = os.path.join(datadir, 'noisy', 'Noise50')

        assert config.model.model_type in [ModelType.LadderVae]
        assert raw_data_dict is None
        label1 = config.data.label1
        label2 = config.data.label2
        train_data = None if skip_train_dataset else NotMNISTNoisyLoader(datapath, train_img_files_pkl, label1, label2)
        val_data = NotMNISTNoisyLoader(datapath, val_img_files_pkl, label1, label2)
    elif config.data.data_type == DataType.Places365:
        train_datapath = os.path.join(datadir, 'Noise-1', 'train')
        val_datapath = os.path.join(datadir, 'Noise-1', 'val')
        assert config.model.model_type in [ModelType.LadderVae, ModelType.LadderVaeTwinDecoder]
        assert raw_data_dict is None
        label1 = config.data.label1
        label2 = config.data.label2
        img_dsample = config.data.img_dsample
        train_data = None if skip_train_dataset else PlacesLoader(
            train_datapath, label1, label2, img_dsample=img_dsample)
        val_data = PlacesLoader(val_datapath, label1, label2, img_dsample=img_dsample)
    elif config.data.data_type == DataType.SemiSupBloodVesselsEMBL:
        datapath = datadir
        normalized_input = config.data.normalized_input
        use_one_mu_std = config.data.use_one_mu_std
        train_aug_rotate = config.data.train_aug_rotate
        enable_random_cropping = config.data.deterministic_grid is False
        train_data_kwargs = {}
        val_data_kwargs = {}

        train_data_kwargs['enable_random_cropping'] = enable_random_cropping
        val_data_kwargs['enable_random_cropping'] = False

        if 'multiscale_lowres_count' in config.data and config.data.multiscale_lowres_count is not None:
            padding_kwargs = {'mode': config.data.padding_mode}
            if 'padding_value' in config.data and config.data.padding_value is not None:
                padding_kwargs['constant_values'] = config.data.padding_value

            train_data = None if skip_train_dataset else SingleChannelMultiDatasetDloader(
                config.data,
                datapath,
                datasplit_type=DataSplitType.Train,
                val_fraction=config.training.val_fraction,
                test_fraction=config.training.test_fraction,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=train_aug_rotate,
                num_scales=config.data.multiscale_lowres_count,
                padding_kwargs=padding_kwargs,
                **train_data_kwargs)

            max_val = train_data.get_max_val()
            val_data = SingleChannelMultiDatasetDloader(
                config.data,
                datapath,
                datasplit_type=DataSplitType.Val,
                val_fraction=config.training.val_fraction,
                test_fraction=config.training.test_fraction,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=False,  # No rotation aug on validation
                max_val=max_val,
                num_scales=config.data.multiscale_lowres_count,
                padding_kwargs=padding_kwargs,
                **val_data_kwargs,
            )

        else:
            train_data = None if skip_train_dataset else SingleChannelMultiDatasetDloader(
                config.data,
                datapath,
                datasplit_type=DataSplitType.Train,
                val_fraction=config.training.val_fraction,
                test_fraction=config.training.test_fraction,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=train_aug_rotate,
                **train_data_kwargs)

            max_val = train_data.get_max_val()
            val_data = SingleChannelMultiDatasetDloader(
                config.data,
                datapath,
                datasplit_type=DataSplitType.Val,
                val_fraction=config.training.val_fraction,
                test_fraction=config.training.test_fraction,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=False,  # No rotation aug on validation
                max_val=max_val,
                **val_data_kwargs,
            )

        # For normalizing, we should be using the training data's mean and std.
        mean_val, std_val = train_data.compute_mean_std()
        train_data.set_mean_std(mean_val, std_val)
        val_data.set_mean_std(mean_val, std_val)

    elif config.data.data_type in [
            DataType.OptiMEM100_014, DataType.CustomSinosoid, DataType.CustomSinosoidThreeCurve, DataType.Prevedel_EMBL,
            DataType.AllenCellMito, DataType.SeparateTiffData, DataType.Pavia2VanillaSplitting, DataType.ShroffMitoEr
    ]:
        if config.data.data_type == DataType.OptiMEM100_014:
            datapath = os.path.join(datadir, 'OptiMEM100x014.tif')
        elif config.data.data_type == DataType.Prevedel_EMBL:
            datapath = os.path.join(datadir, 'MS14__z0_8_sl4_fr10_p_10.1_lz510_z13_bin5_00001.tif')
        elif config.data.data_type in [
                DataType.CustomSinosoid, DataType.CustomSinosoidThreeCurve, DataType.AllenCellMito,
                DataType.SeparateTiffData, DataType.Pavia2VanillaSplitting, DataType.ShroffMitoEr
        ]:
            # we create different filenames for different data configs.
            datapath = datadir

        normalized_input = config.data.normalized_input
        use_one_mu_std = config.data.use_one_mu_std
        train_aug_rotate = config.data.train_aug_rotate
        enable_random_cropping = config.data.deterministic_grid is False
        lowres_supervision = config.model.model_type == ModelType.LadderVAEMultiTarget
        if 'multiscale_lowres_count' in config.data and config.data.multiscale_lowres_count is not None:
            padding_kwargs = {'mode': config.data.padding_mode}
            if 'padding_value' in config.data and config.data.padding_value is not None:
                padding_kwargs['constant_values'] = config.data.padding_value

            train_data = None if skip_train_dataset else MultiScaleTiffDloader(
                config.data,
                datapath,
                datasplit_type=DataSplitType.Train,
                val_fraction=config.training.val_fraction,
                test_fraction=config.training.test_fraction,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=train_aug_rotate,
                enable_random_cropping=enable_random_cropping,
                num_scales=config.data.multiscale_lowres_count,
                lowres_supervision=lowres_supervision,
                padding_kwargs=padding_kwargs,
                allow_generation=True)
            max_val = train_data.get_max_val()

            val_data = MultiScaleTiffDloader(
                config.data,
                datapath,
                datasplit_type=DataSplitType.Val,
                val_fraction=config.training.val_fraction,
                test_fraction=config.training.test_fraction,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=False,  # No rotation aug on validation
                enable_random_cropping=False,
                # No random cropping on validation. Validation is evaluated on determistic grids
                num_scales=config.data.multiscale_lowres_count,
                lowres_supervision=lowres_supervision,
                padding_kwargs=padding_kwargs,
                allow_generation=False,
                max_val=max_val,
            )

        else:
            train_data_kwargs = {'allow_generation': True}
            val_data_kwargs = {'allow_generation': False}
            if config.model.model_type in [ModelType.LadderVaeSepEncoder, ModelType.LadderVaeSepEncoderSingleOptim]:
                data_class = SemiSupDloader
                # mixed_input_type = None,
                # supervised_data_fraction = 0.0,
                train_data_kwargs['mixed_input_type'] = config.data.mixed_input_type
                train_data_kwargs['supervised_data_fraction'] = config.data.supervised_data_fraction
                val_data_kwargs['mixed_input_type'] = config.data.mixed_input_type
                val_data_kwargs['supervised_data_fraction'] = 1.0
            else:
                train_data_kwargs['enable_random_cropping'] = enable_random_cropping
                val_data_kwargs['enable_random_cropping'] = False
                data_class = (MultiChDeterministicTiffRandDloader
                              if config.data.randomized_channels else MultiChDeterministicTiffDloader)

            train_data = None if skip_train_dataset else data_class(config.data,
                                                                    datapath,
                                                                    datasplit_type=DataSplitType.Train,
                                                                    val_fraction=config.training.val_fraction,
                                                                    test_fraction=config.training.test_fraction,
                                                                    normalized_input=normalized_input,
                                                                    use_one_mu_std=use_one_mu_std,
                                                                    enable_rotation_aug=train_aug_rotate,
                                                                    **train_data_kwargs)

            max_val = train_data.get_max_val()
            val_data = data_class(
                config.data,
                datapath,
                datasplit_type=DataSplitType.Val,
                val_fraction=config.training.val_fraction,
                test_fraction=config.training.test_fraction,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=False,  # No rotation aug on validation
                max_val=max_val,
                **val_data_kwargs,
            )

        # For normalizing, we should be using the training data's mean and std.
        mean_val, std_val = train_data.compute_mean_std()
        train_data.set_mean_std(mean_val, std_val)
        val_data.set_mean_std(mean_val, std_val)
    elif config.data.data_type == DataType.Pavia2:
        normalized_input = config.data.normalized_input
        use_one_mu_std = config.data.use_one_mu_std
        train_aug_rotate = config.data.train_aug_rotate
        enable_random_cropping = config.data.deterministic_grid is False
        train_data_kwargs = {'allow_generation': False}
        val_data_kwargs = {'allow_generation': False}
        train_data_kwargs['enable_random_cropping'] = enable_random_cropping
        val_data_kwargs['enable_random_cropping'] = False
        datapath = datadir
        train_data = None if skip_train_dataset else Pavia2ThreeChannelDloader(
            config.data,
            datapath,
            datasplit_type=DataSplitType.Train,
            val_fraction=config.training.val_fraction,
            test_fraction=config.training.test_fraction,
            normalized_input=normalized_input,
            use_one_mu_std=use_one_mu_std,
            enable_rotation_aug=train_aug_rotate,
            **train_data_kwargs)

        max_val = train_data.get_max_val()
        val_data = Pavia2ThreeChannelDloader(
            config.data,
            datapath,
            datasplit_type=DataSplitType.Val,
            val_fraction=config.training.val_fraction,
            test_fraction=config.training.test_fraction,
            normalized_input=normalized_input,
            use_one_mu_std=use_one_mu_std,
            enable_rotation_aug=False,  # No rotation aug on validation
            max_val=max_val,
            **val_data_kwargs,
        )

        # For normalizing, we should be using the training data's mean and std.
        mean_val, std_val = train_data.compute_mean_std()
        train_data.set_mean_std(mean_val, std_val)
        val_data.set_mean_std(mean_val, std_val)
    return train_data, val_data


def create_model_and_train(config, data_mean, data_std, logger, checkpoint_callback, train_loader, val_loader,
                           weights_summary):
    # tensorboard previous files.
    for filename in glob.glob(config.workdir + "/events*"):
        os.remove(filename)

    # checkpoints
    for filename in glob.glob(config.workdir + "/*.ckpt"):
        os.remove(filename)

    model = create_model(config, data_mean, data_std)
    if config.model.model_type == ModelType.LadderVaeStitch2Stage:
        assert config.training.pre_trained_ckpt_fpath and os.path.exists(config.training.pre_trained_ckpt_fpath)

    if config.training.pre_trained_ckpt_fpath:
        print('Starting with pre-trained model', config.training.pre_trained_ckpt_fpath)
        checkpoint = torch.load(config.training.pre_trained_ckpt_fpath)
        _ = model.load_state_dict(checkpoint['state_dict'], strict=False)

    # print(model)
    estop_monitor = config.model.get('monitor', 'val_loss')
    estop_mode = MetricMonitor(estop_monitor).mode()

    callbacks = [
        EarlyStopping(monitor=estop_monitor,
                      min_delta=1e-6,
                      patience=config.training.earlystop_patience,
                      verbose=True,
                      mode=estop_mode),
        checkpoint_callback,
    ]
    if 'val_every_n_steps' in config.training and config.training.val_every_n_steps is not None:
        callbacks.append(ValEveryNSteps(config.training.val_every_n_steps))

    logger.experiment.config.update(config.to_dict())
    # wandb.init(config=config)
    if torch.cuda.is_available():
        # profiler = pl.profiler.AdvancedProfiler(output_filename=os.path.join(config.workdir, 'advance_profile.txt'))
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=config.training.max_epochs,
            gradient_clip_val=config.training.grad_clip_norm_value,
            # gradient_clip_algorithm=config.training.gradient_clip_algorithm,
            logger=logger,
            # fast_dev_run=10,
            #  profiler=profiler,
            # overfit_batches=20,
            callbacks=callbacks,
            weights_summary=weights_summary,
            precision=config.training.precision)
    else:
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            logger=logger,
            gradient_clip_val=config.training.grad_clip_norm_value,
            gradient_clip_algorithm=config.training.gradient_clip_algorithm,
            callbacks=callbacks,
            # fast_dev_run=10,
            # overfit_batches=10,
            weights_summary=weights_summary,
            precision=config.training.precision)
    trainer.fit(model, train_loader, val_loader)


def train_network(train_loader, val_loader, data_mean, data_std, config, model_name, logdir):
    ckpt_monitor = config.model.get('monitor', 'val_loss')
    ckpt_mode = MetricMonitor(ckpt_monitor).mode()
    checkpoint_callback = ModelCheckpoint(
        monitor=ckpt_monitor,
        dirpath=config.workdir,
        filename=model_name + '_best',
        save_last=True,
        save_top_k=1,
        mode=ckpt_mode,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = model_name + "_last"
    logger = WandbLogger(name=os.path.join(config.hostname, config.exptname),
                         save_dir=logdir,
                         project="Disentanglement")
    # logger = TensorBoardLogger(config.workdir, name="", version="", default_hp_metric=False)
    weights_summary = None
    pl.utilities.distributed.log.setLevel(logging.ERROR)
    posterior_collapse_count = 0
    collapse_flag = True
    while collapse_flag and posterior_collapse_count < 20:
        collapse_flag = create_model_and_train(config,
                                               data_mean,
                                               data_std,
                                               logger,
                                               checkpoint_callback,
                                               train_loader,
                                               val_loader,
                                               weights_summary=weights_summary)
        if collapse_flag is None:
            print('CTRL+C inturrupt. Ending')
            return

        if collapse_flag:
            posterior_collapse_count = posterior_collapse_count + 1

    if collapse_flag:
        print("Posterior collapse limit reached, attempting training with KL annealing turned on!")
        while collapse_flag:
            config.loss.kl_annealing = True
            collapse_flag = create_model_and_train(config,
                                                   data_mean,
                                                   data_std,
                                                   logger,
                                                   checkpoint_callback,
                                                   train_loader,
                                                   val_loader,
                                                   weights_summary=weights_summary)
            if collapse_flag is None:
                print('CTRL+C inturrupt. Ending')
                return
