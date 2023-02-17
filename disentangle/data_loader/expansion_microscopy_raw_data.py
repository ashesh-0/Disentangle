from cv2 import imread
from disentangle.core.custom_enum import Enum
from czifile import imread as imread_czi
import os
import numpy as np


class SubDatasetType:
    ChannelA = 'Mitotracker'
    ChannelB = 'Tubulin'
    ChannelApiuB = 'Sum'
    ChannelAeB = 'And'


def sub_directory(subdset_type):
    if subdset_type == SubDatasetType.ChannelA:
        return "MDCK_MitoDeepRed639"
    elif subdset_type == SubDatasetType.ChannelB:
        return "MDCK_AlphaBetaTub488"
    elif subdset_type == SubDatasetType.ChannelApiuB:
        return "MDCK_MitoDeepRed639_AlphaBetaTub647_DAPI405"
    elif subdset_type == SubDatasetType.ChannelAeB:
        return "MDCK_MitoDeepRed639_AlphaBetaTub488"
    else:
        raise ValueError('Invalid subsset_type:', subdset_type)


def get_fnames(subdset_type):
    if subdset_type == SubDatasetType.ChannelB:
        return [
            'Experiment-177-Airyscan Processing-49.czi',
            'Experiment-178-Airyscan Processing-51.czi',
            'Experiment-179-Airyscan Processing-53.czi',
            'Experiment-180-Airyscan Processing-55.czi',
            'Experiment-181-Airyscan Processing-58.czi',
            'Experiment-182-Airyscan Processing-60.czi',
            'Experiment-183-Airyscan Processing-62.czi',
        ][:2]
    elif subdset_type == SubDatasetType.ChannelA:
        return [
            'Experiment-124-Airyscan Processing-01.czi',
            'Experiment-126-Airyscan Processing-05.czi',
            'Experiment-127-Airyscan Processing-07.czi',
            'Experiment-128-Airyscan Processing-09.czi',
            'Experiment-129-Airyscan Processing-11.czi',
            'Experiment-130-Airyscan Processing-13.czi',
            'Experiment-132-Airyscan Processing-15.czi',
            'Experiment-133-Airyscan Processing-18.czi',
            'Experiment-134-Airyscan Processing-21.czi',
            'Experiment-135-Airyscan Processing-23.czi',
            'Experiment-136-Airyscan Processing-25.czi',
        ][:2]
    elif subdset_type == SubDatasetType.ChannelAeB:
        return [
            # 'Experiment-140-Airyscan Processing-03.czi', # this has an issue. There is a straight line 
            # across which we see a clear change in intensity
            'Experiment-155-Airyscan Processing-06.czi',
            'Experiment-156-Airyscan Processing-12.czi',
            'Experiment-157-Airyscan Processing-10.czi',
            'Experiment-159-Airyscan Processing-14.czi',
            'Experiment-160-Airyscan Processing-15.czi',
            'Experiment-162-Airyscan Processing-18.czi',
            'Experiment-163-Airyscan Processing-20.czi',
            'Experiment-164-Airyscan Processing-22.czi',
            'Experiment-165-Airyscan Processing-24.czi',
            'Experiment-166-Airyscan Processing-26.czi',
        ]
    elif subdset_type == SubDatasetType.ChannelApiuB:
        return [
            'Experiment-167-Airyscan Processing-28.czi',
            'Experiment-168-Airyscan Processing-30.czi',
            'Experiment-169-Airyscan Processing-32.czi',
            'Experiment-170-Airyscan Processing-34.czi',
            'Experiment-171-Airyscan Processing-36.czi',
            'Experiment-172-Airyscan Processing-39.czi',
            'Experiment-173-Airyscan Processing-41.czi',
            'Experiment-174-Airyscan Processing-43.czi',
            'Experiment-175-Airyscan Processing-45.czi',
            'Experiment-176-Airyscan Processing-47.czi',
        ][:2]
    else:
        raise ValueError("Invalid subdset_type:", subdset_type)


def load_czi(fpaths):
    imgs = []
    for fpath in fpaths:
        img = imread_czi(fpath)
        print(fpath, img.shape)
        # the first dimension of img stored in imgs will have dim of 1, where the contenation will happen
        imgs.append(img)
    return imgs


def load_data(datadir, subset_dtype):
    direc = os.path.join(datadir, sub_directory(subset_dtype))
    fnames = get_fnames(subset_dtype)
    fpaths = [os.path.join(direc, fname) for fname in fnames]
    data = load_czi(fpaths)
    for i in range(len(data)):
        assert data[i].shape[-1] == 1
        assert data[i].shape[0] == 1
        data[i] = data[i][0, ..., 0]
    return data