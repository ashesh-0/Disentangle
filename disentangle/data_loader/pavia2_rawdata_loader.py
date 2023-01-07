"""
It has 4 channels: Nucleus, Nucleus, Actin, Tubulin
It has 3 sets: Only CYAN, ONLY MAGENTA, MIXED.
It has 2 versions: denoised and raw data.
"""
import os
import numpy as np
from nd2reader import ND2Reader
from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples


def load_nd2(fpaths):
    """
    Load .nd2 images.
    """
    images = []
    for fpath in fpaths:
        with ND2Reader(fpath) as img:
            # channels are the last dimension.
            img = np.concatenate([x[..., None] for x in img], axis=-1)
            images.append(img[None])
    # number of images is the first dimension.
    return np.concatenate(images, axis=0)


class Pavia2DataSetType:
    JustCYAN = '001'
    JustMAGENTA = '010'
    MIXED = '100'


class Pavia2DataSetVersion:
    DD = 'DenoisedDeconvolved'
    RAW = 'Raw data'


def get_mixed_fnames(version):
    if version == Pavia2DataSetVersion.RAW:
        return [
            'HaCaT005.nd2', 'HaCaT009.nd2', 'HaCaT013.nd2', 'HaCaT016.nd2', 'HaCaT019.nd2', 'HaCaT029.nd2',
            'HaCaT037.nd2', 'HaCaT041.nd2', 'HaCaT044.nd2', 'HaCaT051.nd2', 'HaCaT054.nd2', 'HaCaT059.nd2',
            'HaCaT066.nd2', 'HaCaT071.nd2', 'HaCaT006.nd2', 'HaCaT011.nd2', 'HaCaT014.nd2', 'HaCaT017.nd2',
            'HaCaT020.nd2', 'HaCaT031.nd2', 'HaCaT039.nd2', 'HaCaT042.nd2', 'HaCaT045.nd2', 'HaCaT052.nd2',
            'HaCaT056.nd2', 'HaCaT063.nd2', 'HaCaT067.nd2', 'HaCaT007.nd2', 'HaCaT012.nd2', 'HaCaT015.nd2',
            'HaCaT018.nd2', 'HaCaT027.nd2', 'HaCaT034.nd2', 'HaCaT040.nd2', 'HaCaT043.nd2', 'HaCaT046.nd2',
            'HaCaT053.nd2', 'HaCaT058.nd2', 'HaCaT065.nd2', 'HaCaT068.nd2'
        ]


def get_justcyan_fnames(version):
    if version == Pavia2DataSetVersion.RAW:
        return [
            'HaCaT023.nd2', 'HaCaT024.nd2', 'HaCaT026.nd2', 'HaCaT032.nd2', 'HaCaT033.nd2', 'HaCaT036.nd2',
            'HaCaT048.nd2', 'HaCaT049.nd2', 'HaCaT057.nd2', 'HaCaT060.nd2', 'HaCaT062.nd2'
        ]


def get_justmagenta_fnames(version):
    if version == Pavia2DataSetVersion.RAW:
        return [
            'HaCaT008.nd2', 'HaCaT021.nd2', 'HaCaT025.nd2', 'HaCaT030.nd2', 'HaCaT038.nd2', 'HaCaT050.nd2',
            'HaCaT061.nd2', 'HaCaT069.nd2', 'HaCaT010.nd2', 'HaCaT022.nd2', 'HaCaT028.nd2', 'HaCaT035.nd2',
            'HaCaT047.nd2', 'HaCaT055.nd2', 'HaCaT064.nd2', 'HaCaT070.nd2'
        ]


def load_data(datadir, dset_type, dset_version=Pavia2DataSetVersion.RAW):
    if dset_type == Pavia2DataSetType.JustCYAN:
        datadir = os.path.join(datadir, 'ONLY_CYAN')
        fnames = get_justcyan_fnames(dset_version)
    elif dset_type == Pavia2DataSetType.JustMAGENTA:
        datadir = os.path.join(datadir, 'ONLY_MAGENTA')
        fnames = get_justmagenta_fnames(dset_version)
    elif dset_type == Pavia2DataSetType.MIXED:
        datadir = os.path.join(datadir, 'MIXED')
        fnames = get_mixed_fnames(dset_version)

    fpaths = [os.path.join(datadir, x) for x in fnames]
    data = load_nd2(fpaths)
    return data


def train_val_test_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    dtypes = data_config.dset_types
    data = {}
    for dset_type in [Pavia2DataSetType.MIXED, Pavia2DataSetType.JustMAGENTA, Pavia2DataSetType.JustCYAN]:
        if int(dtypes) & int(dset_type):
            data[dset_type] = load_data(datadir, dset_type)

    assert len(data) > 0
    for key in data.keys():
        train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(data[key]))

        if datasplit_type == DataSplitType.Train:
            data[key] = data[key][train_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Val:
            data[key] = data[key][val_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Test:
            data[key] = data[key][test_idx].astype(np.float32)
        else:
            raise Exception("invalid datasplit")
    return data