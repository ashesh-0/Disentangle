"""
For the purpose of doing OOD detection, we need to extract features from the training dataset. 
There are three primary locations where the features can be extracted from:
1. bu_values.
2. mu of Z.
3. sigma of Z.

Then we have different hierarchy levels.
An additional idea is to observe how the boundary uncertainty is.

To save the features, we need a way to also know which features corresponds to which data point. The simplest approach is to save the (normalized) patch as well. 

We follow the following directory structure:

model_name/patches/patch_i.npy
model_name/features/bu_values/hierarchy_K/bu_values_i_K.npy
model_name/features/mu_Z/hierarchy_K/mu_Z_i_K.npy
model_name/features/sigma_Z/hierarchy_K/sigma_Z_i_K.npy
"""
import os
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

import ml_collections
from disentangle.analysis.checkpoint_utils import get_best_checkpoint
from disentangle.config_utils import load_config
from disentangle.core.data_split_type import DataSplitType
from disentangle.core.data_type import DataType
from disentangle.data_loader.patch_index_manager import TilingMode
from disentangle.training import create_dataset, create_model


def extract_raw_feature_one_batch(model, x:torch.Tensor) -> dict:
    """
    Extract features from the model.
    
    Args:
        model: The model to extract features from.
        x: NxCxHxW
    
    Returns:
        A dictionary containing the extracted features.
    """
    _, td_data =model(x, return_bu_values=True)
    
    return {
        'bu_values': [x.detach().cpu().numpy() for x in td_data['bu_values']],
        'mu_Z': [x.get().detach().cpu().numpy() for x in td_data['q_mu']],
        'logvar_Z': [x.get().detach().cpu().numpy() for x in td_data['q_lv']],
    }

def extract_feature_one_batch(model, x:torch.Tensor) -> dict:
    data_dict = extract_raw_feature_one_batch(model, x)
    summarized_dict = {}
    for key in data_dict.keys():
        avg_arr = [x.mean(axis=(2,3)) for x in data_dict[key]]
        std_arr = [x.std(axis=(2,3)) for x in data_dict[key]]
        feature = [np.concatenate([avg,std],axis=-1) for avg,std in zip(avg_arr,std_arr)]
        summarized_dict[key] = feature
    return summarized_dict

def get_feature_fpaths(feature_str, outputdir, num_hierarchies):
    assert feature_str in ['bu_values', 'mu_Z', 'logvar_Z'], "feature_str must be one of ['bu_values', 'mu_Z', 'logvar_Z']"
    feature_fpaths = [os.path.join(outputdir, f'{feature_str}/hierarchy_{k}/{feature_str}_{k}.mmap') for k in range(num_hierarchies)]
    return feature_fpaths

def get_multi_hierarchy_mmaps(feature_str, outputdir, num_hierarchies, channel_count, num_inputs):
    feature_fpaths = get_feature_fpaths(feature_str, outputdir, num_hierarchies)
    # create the directories if they do not exist
    for fpath in feature_fpaths:
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
    
    feature_data = [np.memmap(fpath, dtype=np.float32, mode='w+', shape=(num_inputs, channel_count)) for fpath in feature_fpaths]
    # save the shape 
    shape_fpath = os.path.join(outputdir, feature_str, f'shape_{feature_str}.txt')
    with open(shape_fpath, 'w') as f:
        f.write(f'{num_inputs},{channel_count}\n')
    
    return  feature_data

def get_feature_shape_fname(feature_str):
    return f'shape_{feature_str}.txt'

def get_input_shape_fname():
    """
    Returns the name of the file that contains the shape of the input data.
    """
    return get_feature_shape_fname('input')


def extract_and_save_features(model,dset,  outputdir, num_epochs=1, num_hierarchies=4, bu_values_channels=64,
                                mu_Z_channels=128,
                                sigma_Z_channels=128, 
                               batch_size=64, num_workers=4, shuffle=False):
    """
    Extract features from the dataset. and save them to disk.
    
    Args:
        model: The model to extract features from.
        dset: The dataset to extract features from.
        num_epochs: The number of epochs to run the extraction for.
    """
    os.makedirs(outputdir, exist_ok=True)
    
    print('---------------')
    print(f'Extracting features from {len(dset)} data points for {num_epochs} epochs and saving to {outputdir}')
    print('---------------')

    model.eval()  # Set the model to evaluation mode
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # input
    inpC,inpH,inpW = dset[0][0].shape
    input_fpath = os.path.join(outputdir, 'patches.mmap')
    num_data_points = len(dset) * num_epochs

    input_data = np.memmap(input_fpath, dtype=np.float32, mode='w+', shape=(num_data_points, inpC,inpH, inpW))
    # save the shape 
    shape_fpath = os.path.join(outputdir, get_input_shape_fname())
    with open(shape_fpath, 'w') as f:
        f.write(f'{num_data_points},{inpC},{inpH},{inpW}\n')

    
    # bu_values
    bu_values_data = get_multi_hierarchy_mmaps('bu_values', outputdir, num_hierarchies, bu_values_channels*2, num_data_points)
    mu_data = get_multi_hierarchy_mmaps('mu_Z', outputdir, num_hierarchies, mu_Z_channels*2, num_data_points)
    sigma_data = get_multi_hierarchy_mmaps('logvar_Z', outputdir, num_hierarchies, sigma_Z_channels*2, num_data_points)

    cnt = 0
    for _ in range(num_epochs):
        for batch in tqdm(dloader):
            inp, _ = batch
            input_data[cnt:cnt+inp.shape[0], ...] = inp.numpy()
            inp = inp.to(model.device)
            features = extract_feature_one_batch(model, inp)
            # breakpoint()  # For debugging purposes, remove in production.
            for k in range(num_hierarchies):
                bu_values_data[k][cnt:cnt+inp.shape[0], ...] = features['bu_values'][k]
                mu_data[k][cnt:cnt+inp.shape[0], ...] = features['mu_Z'][k]
                sigma_data[k][cnt:cnt+inp.shape[0], ...] = features['logvar_Z'][k]
            cnt += inp.shape[0]
            # breakpoint()  # For debugging purposes, remove in production.
    
    for key in ['bu_values', 'mu_Z', 'logvar_Z']:
        print(key, features[key][0].shape)

def boilerplate(ckpt_dir, data_dir, test_datapath):
    config = load_config(ckpt_dir)
    padding_kwargs = {
        "mode": config.data.get("padding_mode", "constant"),
    }

    if padding_kwargs["mode"] == "constant":
        padding_kwargs["constant_values"] = config.data.get("padding_value", 0)

    dloader_kwargs = {
        "overlapping_padding_kwargs": padding_kwargs,
        "tiling_mode": TilingMode.ShiftBoundary,
    }
    if test_datapath is not None:
        print(f"Using test dataset: {test_datapath}")
        data_dir = os.path.dirname(test_datapath)
        config = ml_collections.ConfigDict(config)
        with config.unlocked():
            config.data.data_type = DataType.MultiTiffSameSizeDset
            config.data.train_fnames = [os.path.basename(test_datapath)]
            config.data.val_fnames = [os.path.basename(test_datapath)]
            config.data.test_fnames = [os.path.basename(test_datapath)]

    train_dset, val_dset = create_dataset(
        config,
        data_dir,
        eval_datasplit_type=DataSplitType.Val,
        kwargs_dict=dloader_kwargs,
    )

    # For normalizing, we should be using the training data's mean and std.
    mean_dict, std_dict = train_dset.compute_mean_std()
    train_dset.set_mean_std(mean_dict, std_dict)
    val_dset.set_mean_std(mean_dict, std_dict)

    # load model
    model = create_model(config, deepcopy(mean_dict), deepcopy(std_dict))
    ckpt_fpath = get_best_checkpoint(ckpt_dir)
    checkpoint = torch.load(ckpt_fpath)

    _ = model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    _ = model.cuda()

    outputdict = {'config': config, 'train_dset': train_dset, 'val_dset': val_dset, 'model': model}
    if test_datapath is not None:
        outputdict['test_dset'] = train_dset
    return outputdict

def get_output_modeldir(output_dir, ckpt_dir, test_datapath=None, indistr_val=False):
    """
    ckpt_dir: /group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/1
    """
    if ckpt_dir[-1] == '/':
        ckpt_dir = ckpt_dir[:-1]
    modeldir = '_'.join(ckpt_dir.split("/")[-3:])
    resultsdir = os.path.join(output_dir, modeldir)
    if test_datapath is not None:
        test_fname = os.path.basename(test_datapath).replace('.tif', '').replace('.tiff', '')
        resultsdir = os.path.join(resultsdir, test_fname)
    elif indistr_val:
        resultsdir = os.path.join(resultsdir, 'indistribution_val')
    os.makedirs(resultsdir, exist_ok=True)
    # breakpoint()  # For debugging purposes, remove in production.
    return resultsdir

if __name__ == '__main__':
    # OOD: python disentangle/scripts/feature_extractor_for_ood.py --ckpt_dir=/group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/10 --data_dir=/group/jug/ashesh/data/nikola_data/20240531/ --output_dir=/group/jug/ashesh/EnsDeLyon/OOD/ --test_datapath=/group/jug/ashesh/EnsDeLyon/OOD_data/TavernaSox2GolgiV2/TavernaSox2GolgiV2_Test_W0.1.tif
    # In distribution: python disentangle/scripts/feature_extractor_for_ood.py --ckpt_dir=/group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/10 --data_dir=/group/jug/ashesh/data/nikola_data/20240531/ --output_dir=/group/jug/ashesh/EnsDeLyon/OOD/
    import argparse
    parser = argparse.ArgumentParser(description='Extract features from the dataset.')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='The directory containing the model checkpoint.')
    parser.add_argument('--data_dir', type=str, required=True, help='The directory containing the dataset.')
    parser.add_argument('--output_dir', type=str, required=True, help='The directory to save the extracted features.', default='/group/jug/ashesh/EnsDeLyon/OOD/')
    parser.add_argument('--num_epochs', type=int, default=1, help='The number of epochs to run the extraction for.')
    # a flag 
    parser.add_argument('--use_indistribution_val', action='store_true', help='If set, will use the test dataset for feature extraction instead of the training dataset.')
    # batch size 
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for feature extraction.')
    parser.add_argument('--test_datapath', type=str, default=None, help='Path to the test dataset. If provided, will use this dataset for feature extraction instead of the training dataset.')
    args = parser.parse_args()

    boilerplate_output = boilerplate(args.ckpt_dir, args.data_dir, args.test_datapath)
    # breakpoint()  # For debugging purposes, remove in production.
    dset = None
    if args.test_datapath is None:
        if args.use_indistribution_val:
            print("Using in-distribution validation dataset for feature extraction.")
            dset = boilerplate_output['val_dset']
        else:
            print("Using training dataset for feature extraction.")
            dset = boilerplate_output['train_dset']
    else:
        print("Using test dataset for feature extraction.")
        dset  = boilerplate_output['test_dset']

    extract_and_save_features(
        boilerplate_output['model'],
        dset,
        get_output_modeldir(args.output_dir, args.ckpt_dir, args.test_datapath, indistr_val=args.use_indistribution_val),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )