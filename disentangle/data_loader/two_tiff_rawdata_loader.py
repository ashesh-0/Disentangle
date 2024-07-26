import os

import numpy as np

from disentangle.core.data_split_type import DataSplitType, get_datasplit_tuples
from disentangle.core.data_type import DataType
from disentangle.core.tiff_reader import load_tiff


def get_train_val_data(dirname, data_config, datasplit_type, val_fraction, test_fraction):
    # actin-60x-noise2-highsnr.tif  mito-60x-noise2-highsnr.tif
    fpath1 = os.path.join(dirname, data_config.ch1_fname)
    fpath2 = os.path.join(dirname, data_config.ch2_fname)
    fpaths = [fpath1, fpath2]
    fpath0 = ''
    if 'ch_input_fname' in data_config:
        fpath0 = os.path.join(dirname, data_config.ch_input_fname)
        fpaths = [fpath0] + fpaths

    print(f'Loading from {dirname} Channels: '
          f'{fpath1},{fpath2}, inp:{fpath0} Mode:{DataSplitType.name(datasplit_type)}')

    data = np.concatenate([load_tiff(fpath)[..., None] for fpath in fpaths], axis=3)
    if data_config.data_type == DataType.PredictedTiffData:
        assert len(data.shape) == 5 and data.shape[-1] == 1
        data = data[..., 0].copy()
    # data = data[::3].copy()
    # NOTE: This was not the correct way to do it. It is so because the noise present in the input was directly related
    # to the noise present in the channels and so this is not the way we would get the data.
    # We need to add the noise independently to the input and the target.

    # if data_config.get('poisson_noise_factor', False):
    #     data = np.random.poisson(data)
    # if data_config.get('enable_gaussian_noise', False):
    #     synthetic_scale = data_config.get('synthetic_gaussian_scale', 0.1)
    #     print('Adding Gaussian noise with scale', synthetic_scale)
    #     noise = np.random.normal(0, synthetic_scale, data.shape)
    #     data = data + noise

    if datasplit_type == DataSplitType.All:
        return data.astype(np.float32)

    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(data), starting_test=True)
    if datasplit_type == DataSplitType.Train:
        return data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        return data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        return data[test_idx].astype(np.float32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from disentangle.configs.twotiff_config import get_config
    from disentangle.core.data_type import DataType
    from disentangle.core.loss_type import LossType
    from disentangle.core.model_type import ModelType
    from disentangle.core.sampler_type import SamplerType

    config = get_config()
    config.data.enable_gaussian_noise = False
    # config.data.synthetic_gaussian_scale = 1000
    data = get_train_val_data('/group/jug/ashesh/data/ventura_gigascience/', config.data, DataSplitType.Test,
                              config.training.val_fraction, config.training.test_fraction)

    _, ax = plt.subplots(figsize=(6, 3), ncols=2)
    ax[0].imshow(data[0, ..., 0])
    ax[1].imshow(data[0, ..., 1])

    import json

    from disentangle.core.tiff_reader import load_tiff, save_tiff
    schema = '/group/jug/ashesh/data/paper_stats/Test_P128_G32_M50_Sk0/kth_{}/pred_training_pre_eccv_disentangle_2402_D7-M3-S0-L0_108_1.tif'
    pred_data = []
    for i in range(10):
        fpath = schema.format(i)
        pred_data.append(load_tiff(fpath))
        json_fpath = fpath.replace('.tif', '.json')
        # load the json
        with open(json_fpath,'rb') as f:
            json_data = json.load(f)
            assert float(json_fpath['factor']) == 1.0
            pred_data[-1] += float(json_data['offset'])
        
    pred_data = np.concatenate(pred_data, axis=0)
    save_tiff('/group/jug/ashesh/downloads/Actin_HighSNR.tif', data[...,0])
    save_tiff('/group/jug/ashesh/downloads/Mito_HighSNR.tif', data[...,1])
    save_tiff('/group/jug/ashesh/downloads/Actin_pred.tif', pred_data[...,0])
    save_tiff('/group/jug/ashesh/downloads/Mito_pred.tif', pred_data[...,1])