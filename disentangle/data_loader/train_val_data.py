"""
Here, the idea is to load the data from different data dtypes into a single interface.
"""
from disentangle.core.data_type import DataType
from disentangle.data_loader.multi_channel_train_val_data import train_val_data as _load_tiff_train_val
from disentangle.data_loader.sinosoid_dloader import train_val_data as _loadsinosoid
from disentangle.data_loader.allencell_rawdata_loader import get_train_val_data as _loadallencellmito
from disentangle.data_loader.two_tiff_rawdata_loader import get_train_val_data as _loadseparatetiff
from typing import Union


def get_train_val_data(data_config, fpath, is_train: Union[None, bool], val_fraction=None, allow_generation=None):
    """
    Ensure that the shape of data should be N*H*W*C: N is number of data points. H,W are the image dimensions.
    C is the number of channels.
    """
    if data_config.data_type == DataType.OptiMEM100_014:
        return _load_tiff_train_val(fpath, data_config, is_train, val_fraction=val_fraction)
    elif data_config.data_type == DataType.CustomSinosoid:
        return _loadsinosoid(fpath, data_config, is_train, val_fraction=val_fraction, allow_generation=allow_generation)
    elif data_config.data_type == DataType.Prevedel_EMBL:
        return _load_tiff_train_val(fpath, data_config, is_train, val_fraction=val_fraction)
    elif data_config.data_type == DataType.AllenCellMito:
        return _loadallencellmito(fpath, data_config, is_train, val_fraction)
    elif data_config.data_type == DataType.SeparateTiffData:
        return _loadseparatetiff(fpath, data_config, is_train, val_fraction)
    else:
        raise NotImplementedError(f'{DataType.name(data_config.data_type)} is not implemented')
