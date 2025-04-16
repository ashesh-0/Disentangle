import argparse
import os
from datetime import datetime

import papermill as pm


def get_notebook_fpath(directory, notebook_name: str, param_str:str='') -> str:
    """
    A model will have multiple runs. Each run will have a different version.
    """
    assert notebook_name[-6:] == '.ipynb', 'Notebook name should end with .ipynb'
    notebook_with_param = notebook_name[:-6] + '_' + param_str
    fname =  notebook_with_param + '_{now}_{version}.ipynb'
    idx = 0 
    timesubdir = datetime.now().strftime("%Y%m%d") 
    dirpath = os.path.join(directory, notebook_with_param, timesubdir)
    os.makedirs(dirpath, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d.%H.%M")
    while os.path.exists(os.path.join(dirpath, fname.format(version=idx,now=now))):
        now = datetime.now().strftime("%Y%m%d.%H.%M")
        idx += 1
    
    return os.path.join(dirpath, fname.format(version=idx, now=now))

def parse_bool(elem):
    elem = elem.lower()
    assert elem in ['true', 'false'], f'Invalid value {elem}. Must be true or false'
    return elem == 'true'


if __name__ == '__main__':
    # python evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/4
    parser = argparse.ArgumentParser(description='Run a notebook')
    parser.add_argument('--notebook', type=str, help='Notebook to run', default='/home/ashesh.ashesh/code/Disentangle/disentangle/notebooks/SelfSupervisionExperiment.ipynb')
    parser.add_argument('--outputdir', type=str, help='Output notebook directory', default='/group/jug/ashesh/EnsDeLyon/notebook_results/')
    parser.add_argument('--ckpt_dir', type=str, help='Checkpoint to use. eg. /group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/4')
    parser.add_argument('--save_to_file', type=parse_bool, help='Save to file', default=None)

    # params which affect the performance.
    parser.add_argument('--mmse_count', type=int, help='Number of mmse values to generate', default=None)
    parser.add_argument('--custom_image_size', type=int, help='Custom image size', default=None)
    parser.add_argument('--skip_pixels', type=int, help='Skip pixels', default=None)
    parser.add_argument('--use_first_k_images', type=int, help='Use first k images', default=None)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=None)
    parser.add_argument('--aug_theta_max', type=float, help='Augmentation theta max', default=None)
    parser.add_argument('--aug_theta_z_max', type=float, help='Augmentation theta z max', default=None)
    parser.add_argument('--aug_shift_max', type=float, help='Augmentation shift max', default=None)
    parser.add_argument('--max_step_count_step1', type=int, help='Max step count step 1', default=None)
    parser.add_argument('--max_step_count_step2', type=int, help='Max step count step 2', default=None)
    parser.add_argument('--lr_step1', type=float, help='Learning rate step 1', default=None)
    parser.add_argument('--lr_step2', type=float, help='Learning rate step 2', default=None)
    parser.add_argument('--k_augmentations', type=int, help='Number of augmentations', default=None)
    parser.add_argument('--optimaization_mode', type=str, help='Optimization mode', default=None)
    parser.add_argument('--k_moment_value', type=int, help='K moment value', default=None)
    parser.add_argument('--enable_mixing_aug', type=parse_bool, help='Enable mixing augmentation', default=None)
    args = parser.parse_args()

    values_dict = {
            'ckpt_dir': args.ckpt_dir,
            'outputdir': args.outputdir,
            'k_moment_value': 2, 
            'optimaization_mode': 'twostep', 
            'k_augmentations': 8, 
            'aug_theta_max': 0, 
            'aug_theta_z_max': 0, 
            'aug_shift_max': 0, 
            'max_step_count_step1': 200000, 
            'max_step_count_step2': 200000, 
            'lr_step1': 1e-3, 
            'lr_step2': 1e-5,
            'use_first_k_images': 1,
            'skip_pixels': 4,
            'custom_image_size': None,
            'mmse_count': 2,
            'batch_size': 128,
            'enable_mixing_aug': False,
            'save_to_file': True,
    }
    updated_keys = []
    skip_name_keys = ['ckpt_dir', 'outputdir', 'save_to_file']
    for key in values_dict.keys():
        if getattr(args, key) is not None:
            values_dict[key] = args.__dict__[key]
            print(f'Updating {key} to {values_dict[key]}')
            if  key not in skip_name_keys:
                updated_keys.append(key)
    
    
    updated_keys = sorted(updated_keys)
    param_str = '-'.join([f'{key}-{values_dict[key]}' for key in updated_keys])
    
    assert values_dict['optimaization_mode'] in ['twostep', 'onestep','norm_and_network'], f'Invalid optimization mode {values_dict["optimaization_mode"]}. Must be twostep or onestep'
    
    
    notebook_fpath = get_notebook_fpath(args.outputdir, os.path.basename(args.notebook), param_str=param_str)

    # get a year-month-day hour-minute formatted string
    # param_str = f"Elem-{args.elem_size}_MMSE-{args.mmse_count}_Skip-{args.skip_percentile}"
    os.makedirs(os.path.dirname(notebook_fpath), exist_ok=True)
    # save the configuration
    # convert args to dict
    args_dict = vars(args)

    print(notebook_fpath)
    pm.execute_notebook(
        args.notebook,
        notebook_fpath,
        parameters = values_dict,
    )
    
