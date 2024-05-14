"""
Here, the idea is to load the data from different data dtypes into a single interface.
"""
from typing import Union

from disentangle.config_utils import get_configdir_from_saved_predictionfile, load_config
from disentangle.core.data_split_type import DataSplitType
from disentangle.core.data_type import DataType
from disentangle.data_loader.allencell_rawdata_loader import get_train_val_data as _loadallencellmito
from disentangle.data_loader.dao_3ch_rawdata_loader import get_train_val_data as _loaddao3ch
from disentangle.data_loader.derain100H_rawdata_loader import get_train_val_data as _loadderain100H
from disentangle.data_loader.exp_microscopyv2_rawdata_loader import get_train_val_data as _loadexp_microscopyv2
from disentangle.data_loader.haze4k_rawdata_loader import get_train_val_data as _loadhaze4k
from disentangle.data_loader.ht_iba1_ki67_rawdata_loader import get_train_val_data as _load_ht_iba1_ki67
from disentangle.data_loader.multi_channel_train_val_data import train_val_data as _load_tiff_train_val
# from disentangle.data_loader.pqrsinstitute_semisup_rawdata_loader import \
# get_train_val_data as _loadpqrsinstitute2_semisup
from disentangle.data_loader.raw_mrc_dloader import get_train_val_data as _loadmrc
from disentangle.data_loader.schroff_rawdata_loader import get_train_val_data as _loadschroff_mito_er
from disentangle.data_loader.sinosoid_dloader import train_val_data as _loadsinosoid
from disentangle.data_loader.sinosoid_threecurve_dloader import train_val_data as _loadsinosoid3curve
from disentangle.data_loader.sox2golgi_rawdata_loader import get_train_val_data as _loadsox2golgi
from disentangle.data_loader.sox2golgi_v2_rawdata_loader import get_train_val_data as _loadsox2golgi_v2
from disentangle.data_loader.two_tiff_rawdata_loader import get_train_val_data as _loadseparatetiff

# from disentangle.data_loader.xyzinstitute2_rawdata_loader import get_train_val_data as _loadxyzinstitute2
# from disentangle.data_loader.xyzinstitute2_rawdata_loader import \
# get_train_val_data_vanilla as _loadxyzinstitute2_vanilla


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

    elif data_config.data_type == DataType.Prevedel_pqrsinstitute:
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
    elif data_config.data_type == DataType.xyzinstitute2:
        return _loadxyzinstitute2(fpath,
                                  data_config,
                                  datasplit_type,
                                  val_fraction=val_fraction,
                                  test_fraction=test_fraction)
    elif data_config.data_type == DataType.xyzinstitute2VanillaSplitting:
        return _loadxyzinstitute2_vanilla(fpath,
                                          data_config,
                                          datasplit_type,
                                          val_fraction=val_fraction,
                                          test_fraction=test_fraction)
    elif data_config.data_type == DataType.SemiSupBloodVesselspqrsinstitute:
        return _loadpqrsinstitute2_semisup(fpath,
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
    elif data_config.data_type == DataType.Dao3Channel:
        return _loaddao3ch(fpath, data_config, datasplit_type, val_fraction=val_fraction, test_fraction=test_fraction)
    elif data_config.data_type == DataType.ExpMicroscopyV2:
        return _loadexp_microscopyv2(fpath,
                                     data_config,
                                     datasplit_type,
                                     val_fraction=val_fraction,
                                     test_fraction=test_fraction)
    elif data_config.data_type == DataType.Derain100H:
        return _loadderain100H(fpath,
                               data_config,
                               datasplit_type,
                               val_fraction=val_fraction,
                               test_fraction=test_fraction)
    elif data_config.data_type == DataType.Dehaze4K:
        return _loadhaze4k(fpath, data_config, datasplit_type, val_fraction=val_fraction, test_fraction=test_fraction)
    else:
        raise NotImplementedError(f'{DataType.name(data_config.data_type)} is not implemented')
