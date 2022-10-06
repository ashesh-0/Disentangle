"""
Here, the idea is to load the data from different data dtypes into a single interface.
"""
from disentangle.core.data_type import DataType
from disentangle.data_loader.multi_channel_train_val_data import train_val_data as _loadOptiMEM100
from disentangle.data_loader.sinosoid_dloader import train_val_data as _loadsinosoid
from typing import Union


def get_train_val_data(data_config, fpath, is_train: Union[None, bool], val_fraction=None, allow_generation=None):
    """
    Ensure that the shape of data should be N*H*W*C: N is number of data points. H,W are the image dimensions.
    C is the number of channels.
    """
    if data_config.data_type == DataType.OptiMEM100_014:
        return _loadOptiMEM100(fpath, data_config, is_train, val_fraction=val_fraction)
    elif data_config.data_type == DataType.CustomSinosoid:
        return _loadsinosoid(fpath, data_config, is_train, val_fraction=val_fraction, allow_generation=allow_generation)
    else:
        raise NotImplementedError(f'{DataType.name(data_config.data_type)} is not implemented')
