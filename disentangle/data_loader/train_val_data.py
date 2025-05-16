"""
Here, the idea is to load the data from different data dtypes into a single interface.
"""
from copy import deepcopy
from typing import Union

from disentangle.config_utils import get_configdir_from_saved_predictionfile, load_config
from disentangle.core.data_split_type import DataSplitType
from disentangle.core.data_type import DataType
from disentangle.data_loader.allencell_rawdata_loader import get_train_val_data as _loadallencellmito
from disentangle.data_loader.care3D_rawdata_loader import get_train_val_data as _loadcare3D
from disentangle.data_loader.dao_3ch_rawdata_loader import get_train_val_data as _loaddao3ch
from disentangle.data_loader.elisa3D_rawdata_loader import get_train_val_data as _loadelisa3D
from disentangle.data_loader.embl_semisup_rawdata_loader import get_train_val_data as _loadembl2_semisup
from disentangle.data_loader.exp_microscopy_rawdata_loader import get_train_val_data as _loadexp_microscopy
from disentangle.data_loader.ht_iba1_ki67_rawdata_loader import get_train_val_data as _load_ht_iba1_ki67
from disentangle.data_loader.multi_channel_train_val_data import train_val_data as _load_tiff_train_val
from disentangle.data_loader.multicrops_dset_rawdata_loader import get_train_val_data as _loadmulticropdset
from disentangle.data_loader.nikola_7D_rawdata_loader import get_train_val_data as _loadnikola7D
from disentangle.data_loader.pavia2_rawdata_loader import get_train_val_data as _loadpavia2
from disentangle.data_loader.pavia2_rawdata_loader import get_train_val_data_vanilla as _loadpavia2_vanilla
from disentangle.data_loader.pavia3_rawdata_loader import get_train_val_data as _loadpavia3
from disentangle.data_loader.raw_mrc_dloader import get_train_val_data as _loadmrc
from disentangle.data_loader.schroff_rawdata_loader import get_train_val_data as _loadschroff_mito_er
from disentangle.data_loader.silviolabcshl_rawdata_loader import get_train_val_data as _loadsilviolabcshl
from disentangle.data_loader.sinosoid_dloader import train_val_data as _loadsinosoid
from disentangle.data_loader.sinosoid_threecurve_dloader import train_val_data as _loadsinosoid3curve
from disentangle.data_loader.sox2golgi_rawdata_loader import get_train_val_data as _loadsox2golgi
from disentangle.data_loader.sox2golgi_v2_rawdata_loader import get_train_val_data as _loadsox2golgi_v2
from disentangle.data_loader.tiff_raw_dloader import get_train_val_data as _loadmultitiff
from disentangle.data_loader.two_tiff_rawdata_loader import get_train_val_data as _loadseparatetiff


def get_train_val_data(data_config,
                       fpath,
                       datasplit_type: DataSplitType,
                       val_fraction=None,
                       test_fraction=None,
                       allow_generation=None,
                       ignore_specific_datapoints=None):
    """
    Ensure that the shape of data should be N*H*W*C: N is number of data points. H,W are the image dimensions.
    C is the number of channels.
    """
    assert isinstance(datasplit_type, int), f'datasplit_type should be an integer, but is {datasplit_type}'
    if data_config.data_type == DataType.OptiMEM100_014:
        return _load_tiff_train_val(fpath,
                                    data_config,
                                    datasplit_type,
                                    val_fraction=val_fraction,
                                    test_fraction=test_fraction)
    elif data_config.data_type == DataType.CustomSinosoid:
        return _loadsinosoid(fpath,
                             data_config,
                             datasplit_type,
                             val_fraction=val_fraction,
                             test_fraction=test_fraction,
                             allow_generation=allow_generation)

    elif data_config.data_type == DataType.CustomSinosoidThreeCurve:
        return _loadsinosoid3curve(fpath,
                                   data_config,
                                   datasplit_type,
                                   val_fraction=val_fraction,
                                   test_fraction=test_fraction,
                                   allow_generation=allow_generation)

    elif data_config.data_type == DataType.Prevedel_EMBL:
        return _load_tiff_train_val(fpath,
                                    data_config,
                                    datasplit_type,
                                    val_fraction=val_fraction,
                                    test_fraction=test_fraction)
    elif data_config.data_type == DataType.AllenCellMito:
        return _loadallencellmito(fpath, data_config, datasplit_type, val_fraction, test_fraction)
    elif data_config.data_type in [DataType.SeparateTiffData, DataType.PredictedTiffData]:
        if data_config.data_type == DataType.PredictedTiffData:
            cfg1 = load_config(get_configdir_from_saved_predictionfile(data_config.ch1_fname))
            cfg2 = load_config(get_configdir_from_saved_predictionfile(data_config.ch2_fname))
            cfg3 = load_config(get_configdir_from_saved_predictionfile(data_config.ch_input_fname))
            msg = ''
            if 'poisson_noise_factor' in cfg1.data or 'poisson_noise_factor' in cfg2.data or 'poisson_noise_factor' in cfg3.data:
                msg = f'p1:{cfg1.data.poisson_noise_factor} p2:{cfg2.data.poisson_noise_factor} p3:{cfg3.data.poisson_noise_factor}'
                assert cfg1.data.poisson_noise_factor == cfg2.data.poisson_noise_factor == cfg3.data.poisson_noise_factor, msg

            if 'enable_gaussian_noise' in cfg1.data or 'enable_gaussian_noise' in cfg2.data or 'enable_gaussian_noise' in cfg3.data:
                assert cfg1.data.enable_gaussian_noise == cfg2.data.enable_gaussian_noise == cfg3.data.enable_gaussian_noise
                if cfg1.data.enable_gaussian_noise:
                    msg = f'g1:{cfg1.data.synthetic_gaussian_scale} g2:{cfg2.data.synthetic_gaussian_scale} g3:{cfg3.data.synthetic_gaussian_scale}'
                    assert cfg1.data.synthetic_gaussian_scale == cfg2.data.synthetic_gaussian_scale == cfg3.data.synthetic_gaussian_scale, msg

        return _loadseparatetiff(fpath, data_config, datasplit_type, val_fraction, test_fraction)
    elif data_config.data_type == DataType.Pavia2:
        return _loadpavia2(fpath, data_config, datasplit_type, val_fraction=val_fraction, test_fraction=test_fraction)
    elif data_config.data_type == DataType.Pavia2VanillaSplitting:
        return _loadpavia2_vanilla(fpath,
                                   data_config,
                                   datasplit_type,
                                   val_fraction=val_fraction,
                                   test_fraction=test_fraction)
    elif data_config.data_type == DataType.SemiSupBloodVesselsEMBL:
        return _loadembl2_semisup(fpath,
                                  data_config,
                                  datasplit_type,
                                  val_fraction=val_fraction,
                                  test_fraction=test_fraction)

    elif data_config.data_type == DataType.ShroffMitoEr:
        return _loadschroff_mito_er(fpath,
                                    data_config,
                                    datasplit_type,
                                    val_fraction=val_fraction,
                                    test_fraction=test_fraction)
    elif data_config.data_type == DataType.HTIba1Ki67:
        return _load_ht_iba1_ki67(fpath,
                                  data_config,
                                  datasplit_type,
                                  val_fraction=val_fraction,
                                  test_fraction=test_fraction)
    elif data_config.data_type == DataType.BioSR_MRC:
        return _loadmrc(fpath, data_config, datasplit_type, val_fraction=val_fraction, test_fraction=test_fraction)
    elif data_config.data_type == DataType.TavernaSox2Golgi:
        return _loadsox2golgi(fpath,
                              data_config,
                              datasplit_type,
                              val_fraction=val_fraction,
                              test_fraction=test_fraction)
    elif data_config.data_type == DataType.TavernaSox2GolgiV2:
        return _loadsox2golgi_v2(fpath,
                                 data_config,
                                 datasplit_type,
                                 val_fraction=val_fraction,
                                 test_fraction=test_fraction)
    elif data_config.data_type in [DataType.Dao3Channel, DataType.Dao3ChannelWithInput]:
        # if data_config.data_type == DataType.Dao3ChannelWithInput:
        # assert 'target_idx_list' in data_config, 'target_idx_list should be provided for Dao3ChannelWithInput'
        # assert 'input_idx' in data_config, 'input_idx should be provided for Dao3ChannelWithInput'

        return _loaddao3ch(fpath, data_config, datasplit_type, val_fraction=val_fraction, test_fraction=test_fraction)
    elif data_config.data_type in [DataType.ExpMicroscopyV1, DataType.ExpMicroscopyV2, DataType.ExpMicroscopyV3]:
        return _loadexp_microscopy(fpath,
                                     data_config,
                                     datasplit_type,
                                     val_fraction=val_fraction,
                                     test_fraction=test_fraction)
    elif data_config.data_type == DataType.Pavia3SeqData:
        return _loadpavia3(fpath, data_config, datasplit_type, val_fraction=val_fraction, test_fraction=test_fraction)
    elif data_config.data_type == DataType.NicolaData:
        return _loadnikola7D(fpath, data_config, datasplit_type, val_fraction=val_fraction, test_fraction=test_fraction)
    elif data_config.data_type == DataType.SilvioLabCSHLData:
        return _loadsilviolabcshl(fpath,
                                  data_config,
                                  datasplit_type,
                                  val_fraction=val_fraction,
                                  test_fraction=test_fraction)
    elif data_config.data_type == DataType.MultiCropDset:
        return _loadmulticropdset(fpath,
                                  data_config,
                                 datasplit_type,
                                 val_fraction=val_fraction,
                                 test_fraction=test_fraction)
    elif data_config.data_type == DataType.Elisa3DData:
        return _loadelisa3D(fpath,
                           data_config,
                           datasplit_type,
                           val_fraction=val_fraction,
                           test_fraction=test_fraction)
    elif data_config.data_type == DataType.Care3D:
        return _loadcare3D(fpath,
                          data_config,
                          datasplit_type,
                          val_fraction=val_fraction,
                          test_fraction=test_fraction)
    elif data_config.data_type == DataType.MultiTiffSameSizeDset:
        return _loadmultitiff(fpath,data_config,datasplit_type)
    
    elif data_config.data_type == DataType.SimilarityExperiment:
        import numpy as np
        from skimage.transform import resize

        import ml_collections as ml
        from disentangle.data_loader.multifile_raw_dloader import MultiChannelData

        new_config = ml.ConfigDict(data_config)
        new_config.data_type = data_config.raw_data_type
        data = get_train_val_data(new_config,
                       fpath,
                       datasplit_type,
                       val_fraction=val_fraction,
                       test_fraction=test_fraction,
                       allow_generation=allow_generation,
                       ignore_specific_datapoints=ignore_specific_datapoints)
        if isinstance(data, MultiChannelData):
            relevant_channel_data = np.concatenate([x[...,data_config.relevant_channel_position] for x in data._data], axis=0)
            del data
        else:
            assert isinstance(data, np.ndarray), f'Expected np.ndarray, but got {type(data)}'
            relevant_channel_data = data[...,data_config.relevant_channel_position]
        # upscale it by data_config.upscale_factor
        factor1 = data_config.ch1_factor
        factor2 = data_config.ch2_factor
        ch1_data = resize(relevant_channel_data, (relevant_channel_data.shape[0], int(relevant_channel_data.shape[1] * factor1), 
                                                  int(relevant_channel_data.shape[2] * factor1)), anti_aliasing=True)
        
        ch2_data = resize(relevant_channel_data, (relevant_channel_data.shape[0], int(relevant_channel_data.shape[1] * factor2), 
                                                  int(relevant_channel_data.shape[2] * factor2)), anti_aliasing=True)

        return (ch1_data.astype(np.float32), ch2_data.astype(np.float32))
    else:
        raise NotImplementedError(f'{DataType.name(data_config.data_type)} is not implemented')
