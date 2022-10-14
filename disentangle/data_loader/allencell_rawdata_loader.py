import os
from disentangle.core.tiff_reader import load_tiffs


def get_train_val_datafiles(dirname, is_train, val_fraction):
    fnames = [
        'AICS-11_0.ome.tif',
        'AICS-11_1.ome.tif',
        'AICS-11_2.ome.tif',
        'AICS-11_3.ome.tif',
        'AICS-11_4.ome.tif',
        'AICS-11_5.ome.tif',
        'AICS-11_6.ome.tif',
        'AICS-11_7.ome.tif',
        'AICS-11_8.ome.tif',
        'AICS-11_9.ome.tif',
        # 'AICS-11_10.ome.tif', 'AICS-11_11.ome.tif', 'AICS-11_12.ome.tif', 'AICS-11_13.ome.tif', 'AICS-11_14.ome.tif',
        # 'AICS-11_15.ome.tif', 'AICS-11_16.ome.tif', 'AICS-11_17.ome.tif', 'AICS-11_18.ome.tif', 'AICS-11_19.ome.tif',
        # 'AICS-11_20.ome.tif', 'AICS-11_21.ome.tif', 'AICS-11_22.ome.tif', 'AICS-11_23.ome.tif', 'AICS-11_24.ome.tif',
        # 'AICS-11_25.ome.tif', 'AICS-11_26.ome.tif', 'AICS-11_27.ome.tif', 'AICS-11_28.ome.tif', 'AICS-11_29.ome.tif',
        # 'AICS-11_30.ome.tif', 'AICS-11_31.ome.tif', 'AICS-11_32.ome.tif', 'AICS-11_33.ome.tif', 'AICS-11_34.ome.tif',
        # 'AICS-11_35.ome.tif', 'AICS-11_36.ome.tif', 'AICS-11_37.ome.tif', 'AICS-11_38.ome.tif', 'AICS-11_39.ome.tif',
        # 'AICS-11_40.ome.tif', 'AICS-11_41.ome.tif', 'AICS-11_42.ome.tif', 'AICS-11_43.ome.tif', 'AICS-11_44.ome.tif',
        # 'AICS-11_45.ome.tif', 'AICS-11_46.ome.tif', 'AICS-11_47.ome.tif', 'AICS-11_48.ome.tif', 'AICS-11_49.ome.tif',
        # 'AICS-11_50.ome.tif', 'AICS-11_51.ome.tif', 'AICS-11_52.ome.tif', 'AICS-11_53.ome.tif', 'AICS-11_54.ome.tif',
        # 'AICS-11_55.ome.tif', 'AICS-11_56.ome.tif', 'AICS-11_57.ome.tif', 'AICS-11_58.ome.tif'
    ]
    val_count = int(val_fraction * len(fnames))

    val_names = fnames[-val_count:]
    train_names = fnames[:-val_count]

    if is_train:
        return [os.path.join(dirname, fname) for fname in train_names]
    else:
        return [os.path.join(dirname, fname) for fname in val_names]


def get_train_val_data(dirname, data_config, is_train, val_fraction):
    fpaths = get_train_val_datafiles(dirname, is_train, val_fraction)
    print(f'Loading {dirname} with Channels {data_config.channel_1},{data_config.channel_2}, is_train:{is_train}')
    return load_tiffs(fpaths)[..., [data_config.channel_1, data_config.channel_2]]
