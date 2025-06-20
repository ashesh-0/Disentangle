import json
import os

import numpy as np
import torch
import torch.nn as nn

from disentangle.core.data_type import DataType
from disentangle.core.model_type import ModelType
from disentangle.nets.gmm_nnbased_noise_model import DeepGMMNoiseModel
from disentangle.nets.gmm_noise_model import GaussianMixtureNoiseModel
from disentangle.nets.hist_gmm_noise_model import HistGMMNoiseModel
from disentangle.nets.hist_noise_model import HistNoiseModel


class DisentNoiseModel(nn.Module):

    def __init__(self, *nmodels):
        super().__init__()
        # self.nmodels = nmodels
        for i, nmodel in enumerate(nmodels):
            if nmodel is not None:
                self.add_module(f'nmodel_{i}', nmodel)

        self._nm_cnt = 0
        for nmodel in nmodels:
            if nmodel is not None:
                self._nm_cnt += 1

        print(f'[{self.__class__.__name__}] Nmodels count:{self._nm_cnt}')

    def likelihood(self, obs, signal):
        if obs.shape[1] == 1:
            assert signal.shape[1] == 1
            assert self.n2model is None
            return self.nmodel_0.likelihood(obs, signal)
        assert obs.shape[1] == self._nm_cnt, f'{obs.shape[1]} != {self._nm_cnt}'
        ll_list = []
        for ch_idx in range(obs.shape[1]):
            nmodel = getattr(self, f'nmodel_{ch_idx}')
            ll_list.append(nmodel.likelihood(obs[:, ch_idx:ch_idx + 1], signal[:, ch_idx:ch_idx + 1]))
        return torch.cat(ll_list, dim=1)


def last2path(fpath):
    return os.path.join(*fpath.split('/')[-2:])


def get_nm_config(noise_model_fpath):
    config_fpath = os.path.join(os.path.dirname(noise_model_fpath), 'config.json')
    with open(config_fpath, 'r') as f:
        noise_model_config = json.load(f)
    return noise_model_config


def nm_config_sanity_check_target_idx_list(config):
    if config.data.get('disable_checks', True):
        return
    
    def get_channel(fname):
        assert isinstance(fname, list)
        assert len(fname) == 2
        assert fname[1] == ''
        fname = fname[0]
        if config.data.data_type == DataType.NicolaData:
            token = fname.replace('.tif', '').split('_')[-1]
            assert token.startswith('channel')
            return int(token[len('channel'):])
        elif config.data.data_type == DataType.TavernaSox2GolgiV2:
            return fname.replace('.tif', '').replace('sox2golgiv2', '').strip('_')

    def get_dset_type(fname):
        """
        Everything except the channel token.
        """
        assert isinstance(fname, list)
        assert len(fname) == 2
        assert fname[1] == ''
        fname = fname[0]
        fname = fname.replace('.tif', '')
        # uSplit_14022025_lowSNR_channel0
        tokens = fname.split('_')
        token = tokens[-2]
        # lowSNR
        return token.replace('SNR', '')

    fname_list = []
    ch_list = []
    dsettype_list = []
    for ch_idx in range(len(config.data.target_idx_list)):
        nm = config.model[f'noise_model_ch{ch_idx+1}_fpath']
        nm_config = get_nm_config(nm)
        fname = nm_config['fname']
        dsettype = get_dset_type(fname)
        dsettype_list.append(dsettype)
        ch = get_channel(fname)
        if ch is None:
            print(f'Warning: {ch_idx} is None, skipping validation.')
            continue
        ch_list.append(ch)
        fname_list.append(fname)
        assert config.data.channel_idx_list[config.data.target_idx_list[
            ch_idx]] == ch, f'{config.data.channel_idx_list[config.data.target_idx_list[ch_idx]]} != {ch}'

    assert len(set(dsettype_list)) == 1, f'{dsettype_list} should be just one'
    if 'dset_type' in config.data and config.data.get('disable_checks', False) is False:
        assert dsettype == config.data.dset_type, f'{dsettype} != {config.data.dset_type}'

    # nm1 = config.model.noise_model_ch1_fpath
    # nm2 = config.model.noise_model_ch2_fpath
    # nm1_config = get_nm_config(nm1)
    # nm2_config = get_nm_config(nm2)

    # fname1 = nm1_config['fname']
    # fname2 = nm2_config['fname']
    # ch1 = get_channel(fname1)
    # ch2 = get_channel(fname2)

    # assert len(config.data.target_idx_list) == 2
    # assert config.data.target_idx_list[0] == ch1, f'{config.data.target_idx_list[0]} != {ch1}'
    # assert config.data.target_idx_list[1] == ch2, f'{config.data.target_idx_list[1]} != {ch2}'
    # # check low, mid, high, verylow
    # dsettype1 = get_dset_type(fname1)
    # dsettype2 = get_dset_type(fname2)
    # assert dsettype1 == dsettype2, f'{dsettype1} != {dsettype2}'
    # assert dsettype1 == config.data.dset_type, f'{dsettype1} != {config.data.dset_type}'


def noise_model_config_sanity_check(noise_model_fpath, config, channel_key=None):
    if 'target_idx_list' in config.data and config.data.target_idx_list is not None:
        return nm_config_sanity_check_target_idx_list(config)

    config_fpath = os.path.join(os.path.dirname(noise_model_fpath), 'config.json')
    with open(config_fpath, 'r') as f:
        noise_model_config = json.load(f)

    # make sure that the amount of noise is consistent.
    if 'add_gaussian_noise_std' in noise_model_config:
        # data.enable_gaussian_noise = False
        # config.data.synthetic_gaussian_scale = 1000
        assert 'enable_gaussian_noise' in config.data
        assert config.data.enable_gaussian_noise == True, 'Gaussian noise is not enabled'

        assert 'synthetic_gaussian_scale' in config.data
        assert noise_model_config[
            'add_gaussian_noise_std'] == config.data.synthetic_gaussian_scale, f'{noise_model_config["add_gaussian_noise_std"]} != {config.data.synthetic_gaussian_scale}'

    cfg_poisson_noise_factor = config.data.get('poisson_noise_factor', -1)
    nm_poisson_noise_factor = noise_model_config.get('poisson_noise_factor', -1)
    assert cfg_poisson_noise_factor == nm_poisson_noise_factor, f'{nm_poisson_noise_factor} != {cfg_poisson_noise_factor}'

    if 'train_pure_noise_model' in noise_model_config and noise_model_config['train_pure_noise_model']:
        print('Pure noise model is being used now.')
        return
    # make sure that the same file is used for noise model and data.
    if channel_key is not None and channel_key in noise_model_config:
        fname = noise_model_config['fname']
        if '' in fname:
            fname.remove('')
        assert len(fname) == 1
        fname = fname[0]
        cur_data_fpath = os.path.join(config.datadir, config.data[channel_key])
        nm_data_fpath = os.path.join(noise_model_config['datadir'], fname)
        if cur_data_fpath != nm_data_fpath:
            print(f'Warning: {cur_data_fpath} != {nm_data_fpath}')
            assert last2path(cur_data_fpath) == last2path(nm_data_fpath), f'{cur_data_fpath} != {nm_data_fpath}'
        # assert cur_data_fpath == nm_data_fpath, f'{cur_data_fpath} != {nm_data_fpath}'
    else:
        print(f'Warning: channel_key is not found in noise model config: {channel_key}')

    if config.data.data_type == DataType.Pavia3SeqData:

        cond_str = {'Balanced': 'Cond_1', 'MediumSkew': 'Cond_2', 'HighSkew': 'Cond_3'}[config.data.alpha_level]
        power_str = {'High': 'Main', 'Medium': 'Divided_2', 'Low': 'Divided_4'}[config.data.power_level]
        fname = noise_model_config['fname'][0]
        # 'Cond_1-Main.tif'
        if 'Deconvolved' not in noise_model_config['fname'][0]:
            # this is old code.
            assert fname.replace('.tif', '') == f'{cond_str}-{power_str}', f'{fname} != {cond_str}-{power_str}'
        else:
            assert cond_str.lower().replace('_', '') in fname.split('_'), f'{cond_str} not in {fname}'
            assert power_str.lower().replace('_', '') in fname.replace('div', 'divided'), f'{power_str} not in {fname}'

        # 0/1
        channel_idx = noise_model_config['channel_idx'][0]
        if channel_key == 'ch1_fname':
            assert channel_idx == 0
        elif channel_key == 'ch2_fname':
            assert channel_idx == 1
        else:
            raise ValueError(f'Invalid channel_key: {channel_key}')
    elif config.data.data_type == DataType.TavernaSox2Golgi:
        fname = os.path.basename(noise_model_fpath)
        if channel_key == 'ch1_fname':
            assert config.data.channel_1 in fname
            assert config.data.channel_2 not in fname
        elif channel_key == 'ch2_fname':
            assert config.data.channel_2 in fname
            assert config.data.channel_1 not in fname


def get_noise_model(config):
    if 'enable_noise_model' in config.model and config.model.enable_noise_model:
        nmodels = []
        if config.model.model_type == ModelType.Denoiser:
            if config.model.noise_model_type == 'hist':
                if config.model.denoise_channel == 'Ch1':
                    print(f'Noise model Ch1: {config.model.noise_model_ch1_fpath}')
                    hist1 = np.load(config.model.noise_model_ch1_fpath)
                    nmodel1 = HistNoiseModel(hist1)
                    nmodel2 = None
                    nmodels = [nmodel1, nmodel2]
                elif config.model.denoise_channel == 'Ch2':
                    print(f'Noise model Ch2: {config.model.noise_model_ch2_fpath}')
                    hist2 = np.load(config.model.noise_model_ch2_fpath)
                    nmodel1 = HistNoiseModel(hist2)
                    nmodel2 = None
                    nmodels = [nmodel1, nmodel2]
                elif config.model.denoise_channel == 'input':
                    print(f'Noise model Ch1: {config.model.noise_model_ch1_fpath}')
                    hist1 = np.load(config.model.noise_model_ch1_fpath)
                    nmodel1 = HistNoiseModel(hist1)
                    nmodel2 = None
                    nmodels = [nmodel1, nmodel2]
            elif config.model.noise_model_type == 'gmm':
                if config.model.denoise_channel == 'Ch1':
                    nmodel_fpath = config.model.noise_model_ch1_fpath
                    print(f'Noise model Ch1: {nmodel_fpath}')
                    nmodel1 = GaussianMixtureNoiseModel(params=np.load(nmodel_fpath))
                    noise_model_config_sanity_check(nmodel_fpath, config, 'ch1_fname')
                    nmodel2 = None
                    nmodels = [nmodel1, nmodel2]
                elif config.model.denoise_channel == 'Ch2':
                    nmodel_fpath = config.model.noise_model_ch2_fpath
                    print(f'Noise model Ch2: {nmodel_fpath}')
                    nmodel1 = GaussianMixtureNoiseModel(params=np.load(nmodel_fpath))
                    noise_model_config_sanity_check(nmodel_fpath, config, 'ch2_fname')
                    nmodel2 = None
                    nmodels = [nmodel1, nmodel2]
                elif config.model.denoise_channel == 'input':
                    nmodel_fpath = config.model.noise_model_ch1_fpath
                    print(f'Noise model input: {nmodel_fpath}')
                    nmodel1 = GaussianMixtureNoiseModel(params=np.load(nmodel_fpath))
                    noise_model_config_sanity_check(nmodel_fpath, config)
                    nmodel2 = None
                    nmodels = [nmodel1, nmodel2]
            else:
                raise ValueError(f'Invalid denoise_channel: {config.model.denoise_channel}')
        elif config.model.noise_model_type == 'hist':
            print(f'Noise model Ch1: {config.model.noise_model_ch1_fpath}')
            print(f'Noise model Ch2: {config.model.noise_model_ch2_fpath}')

            hist1 = np.load(config.model.noise_model_ch1_fpath)
            nmodel1 = HistNoiseModel(hist1)
            hist2 = np.load(config.model.noise_model_ch2_fpath)
            nmodel2 = HistNoiseModel(hist2)
            nmodels = [nmodel1, nmodel2]
        elif config.model.noise_model_type == 'histgmm':
            print(f'Noise model Ch1: {config.model.noise_model_ch1_fpath}')
            print(f'Noise model Ch2: {config.model.noise_model_ch2_fpath}')

            noise_model_config_sanity_check(config.model.noise_model_ch1_fpath, config, 'ch1_fname')
            noise_model_config_sanity_check(config.model.noise_model_ch2_fpath, config, 'ch2_fname')

            hist1 = np.load(config.model.noise_model_ch1_fpath)
            nmodel1 = HistGMMNoiseModel(hist1)
            nmodel1.fit()

            hist2 = np.load(config.model.noise_model_ch2_fpath)
            nmodel2 = HistGMMNoiseModel(hist2)
            nmodel2.fit()
            nmodels = [nmodel1, nmodel2]
        elif config.model.noise_model_type == 'deepgmm':
            print(f'Noise model Ch1: {config.model.noise_model_ch1_fpath}')
            print(f'Noise model Ch2: {config.model.noise_model_ch2_fpath}')

            nmodel1 = DeepGMMNoiseModel(params=np.load(config.model.noise_model_ch1_fpath))
            nmodel2 = DeepGMMNoiseModel(params=np.load(config.model.noise_model_ch2_fpath))
            nmodel1.make_learnable()
            nmodel2.make_learnable()
            nmodels = [nmodel1, nmodel2]
        elif config.model.noise_model_type == 'gmm':
            print(f'Noise model Ch1: {config.model.noise_model_ch1_fpath}')
            print(f'Noise model Ch2: {config.model.noise_model_ch2_fpath}')

            nmodel1 = GaussianMixtureNoiseModel(params=np.load(config.model.noise_model_ch1_fpath))
            nmodel2 = GaussianMixtureNoiseModel(params=np.load(config.model.noise_model_ch2_fpath))

            noise_model_config_sanity_check(config.model.noise_model_ch1_fpath, config, 'ch1_fname')
            noise_model_config_sanity_check(config.model.noise_model_ch2_fpath, config, 'ch2_fname')
            nmodels = [nmodel1, nmodel2]
            if 'num_targets' in config.model and config.model.num_targets > 2:
                for i in range(3, config.model.num_targets+1):
                    nm_fpath = config.model[f'noise_model_ch{i}_fpath']
                    print(f'Noise model Ch{i}: {nm_fpath}')
                    nmodel_ch = GaussianMixtureNoiseModel(params=np.load(nm_fpath))
                    nmodels.append(nmodel_ch)

        if config.model.get('noise_model_learnable', False):
            for nmodel in nmodels:
                if nmodel is not None:
                    nmodel.make_learnable()

        return DisentNoiseModel(*nmodels)
    return None


if __name__ == '__main__':
    from disentangle.configs.nikola_7D_config import get_config
    config = get_config()
    noise_model_config_sanity_check(config.model.noise_model_ch1_fpath, config)
