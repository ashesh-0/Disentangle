import argparse
import os
from datetime import datetime

import papermill as pm


def get_notebook_fpath(directory, notebook_name: str) -> str:
    """
    A model will have multiple runs. Each run will have a different version.
    """
    assert notebook_name[-6:] == '.ipynb', 'Notebook name should end with .ipynb'
    fname = notebook_name[:-6] + '_{now}_{version}.ipynb'
    idx = 0 
    now = datetime.now().strftime("%Y%m%d.%H.%M.%S")
    while os.path.exists(os.path.join(directory, fname.format(version=idx,now=now))):
        now = datetime.now().strftime("%Y%m%d.%H.%M")
        idx += 1
    
    return os.path.join(directory, fname.format(version=idx, now=now))



if __name__ == '__main__':
    # python evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/4
    parser = argparse.ArgumentParser(description='Run a notebook')
    parser.add_argument('--notebook', type=str, help='Notebook to run', default='/home/ashesh.ashesh/code/Disentangle/disentangle/notebooks/SelfSupervisionExperiment.ipynb')
    parser.add_argument('--outputdir', type=str, help='Output notebook directory', default='/group/jug/ashesh/EnsDeLyon/notebook_results/')
    parser.add_argument('--ckpt_dir', type=str, help='Checkpoint to use. eg. /group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/4')
    parser.add_argument('--mmse_count', type=int, help='Number of mmse values to generate', default=2)
    
    parser.add_argument('--image_size_for_grid_centers', type=int, help='Image size for grid centers', default=8)
    parser.add_argument('--custom_image_size', type=int, help='Custom image size', default=16)
    parser.add_argument('--skip_pixels', type=int, help='Skip pixels', default=4)

    parser.add_argument('--use_first_k_images', type=int, help='Use first k images', default=1)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
    parser.add_argument('--aug_theta_max', type=float, help='Augmentation theta max', default=0)
    parser.add_argument('--aug_theta_z_max', type=float, help='Augmentation theta z max', default=0)
    parser.add_argument('--aug_shift_max', type=float, help='Augmentation shift max', default=0)
    parser.add_argument('--max_step_count_step1', type=int, help='Max step count step 1', default=200000)
    parser.add_argument('--max_step_count_step2', type=int, help='Max step count step 2', default=200000)
    parser.add_argument('--lr_step1', type=float, help='Learning rate step 1', default=1e-3)
    parser.add_argument('--lr_step2', type=float, help='Learning rate step 2', default=1e-5)
    parser.add_argument('--k_augmentations', type=int, help='Number of augmentations', default=8)
    parser.add_argument('--optimaization_mode', type=str, help='Optimization mode', default='twostep')
    parser.add_argument('--k_moment_value', type=int, help='K moment value', default=8)
    args = parser.parse_args()

    notebook_fpath = get_notebook_fpath(args.outputdir, os.path.basename(args.notebook))
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
        parameters = {
            'ckpt_dir': args.ckpt_dir,
            'mmse_count': args.mmse_count,
            'image_size_for_grid_centers': args.image_size_for_grid_centers,
            'outputdir': args.outputdir,
            'custom_image_size': args.custom_image_size,
            'use_first_k_images': args.use_first_k_images,
            'batch_size': args.batch_size,
            'aug_theta_max': args.aug_theta_max,
            'aug_theta_z_max': args.aug_theta_z_max,
            'aug_shift_max': args.aug_shift_max,
            'max_step_count_step1': args.max_step_count_step1,
            'max_step_count_step2': args.max_step_count_step2,
            'skip_pixels': args.skip_pixels,
            'lr_step1': args.lr_step1,
            'lr_step2': args.lr_step2,
            'k_augmentations': args.k_augmentations,
            'optimaization_mode': args.optimaization_mode,
            'k_moment_value': args.k_moment_value,
            # 'allowed_keys': allowed_keys,
            }
    )
    
