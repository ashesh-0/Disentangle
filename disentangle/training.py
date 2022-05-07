import glob
import logging
import os
import pickle

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from disentangle.core.data_type import DataType
from disentangle.core.loss_type import LossType
from disentangle.core.model_type import ModelType
from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from disentangle.data_loader.multi_channel_tiff_dloader import MultiChTiffDloader
from disentangle.data_loader.notmnist_dloader import NotMNISTNoisyLoader
from disentangle.data_loader.places_dloader import PlacesLoader
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
    elif config.data.data_type == DataType.OptiMEM100_014:
        datapath = os.path.join(datadir, 'OptiMEM100x014.tif')
        if 'deterministic_grid' in config.data and config.data.deterministic_grid is True:
            train_data = None if skip_train_dataset else MultiChDeterministicTiffDloader(
                config.data.image_size,
                datapath,
                config.data.channel_1,
                config.data.channel_2,
                is_train=True,
                val_fraction=config.training.val_fraction)
            val_data = MultiChDeterministicTiffDloader(config.data.image_size,
                                                       datapath,
                                                       config.data.channel_1,
                                                       config.data.channel_2,
                                                       is_train=False,
                                                       val_fraction=config.training.val_fraction)

        else:
            normalized_input = config.data.get('normalized_input', False)
            train_data = None if skip_train_dataset else MultiChTiffDloader(
                config.data.image_size,
                datapath,
                config.data.channel_1,
                config.data.channel_2,
                thresh=config.data.threshold,
                is_train=True,
                val_fraction=config.training.val_fraction,
                repeat_factor=config.training.train_repeat_factor,
                normalized_input=normalized_input)
            val_data = MultiChTiffDloader(config.data.image_size,
                                          datapath,
                                          config.data.channel_1,
                                          config.data.channel_2,
                                          thresh=config.data.threshold,
                                          is_train=False,
                                          val_fraction=config.training.val_fraction,
                                          repeat_factor=config.training.val_repeat_factor,
                                          normalized_input=normalized_input)
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
    # print(model)
    callbacks = [
        EarlyStopping(monitor='val_loss',
                      min_delta=1e-6,
                      patience=config.training.earlystop_patience,
                      verbose=True,
                      mode='min'),
        checkpoint_callback,
    ]
    if 'val_every_n_steps' in config.training and config.training.val_every_n_steps is not None:
        callbacks.append(ValEveryNSteps(config.training.val_every_n_steps))

    if torch.cuda.is_available():
        # profiler = pl.profiler.AdvancedProfiler(output_filename=os.path.join(config.workdir, 'advance_profile.txt'))
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=config.training.max_epochs,
            gradient_clip_val=config.training.grad_clip_norm_value,
            gradient_clip_algorithm=config.training.gradient_clip_algorithm,
            logger=logger,
            #  profiler=profiler,
            callbacks=callbacks,
            weights_summary=weights_summary)
    else:
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            logger=logger,
            gradient_clip_val=config.training.grad_clip_norm_value,
            gradient_clip_algorithm=config.training.gradient_clip_algorithm,
            callbacks=callbacks,
            #  fast_dev_run=100,
            # overfit_batches=10,
            weights_summary=weights_summary)
    trainer.fit(model, train_loader, val_loader)


def train_network(train_loader, val_loader, data_mean, data_std, config, model_name, log_info=False):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.workdir,
        filename=model_name + '_best',
        save_last=True,
        save_top_k=1,
        mode='min',
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = model_name + "_last"
    logger = TensorBoardLogger(config.workdir, name="", version="", default_hp_metric=False)
    weights_summary = "top" if log_info else None
    if not log_info:
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
