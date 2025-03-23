import argparse
import os
from datetime import datetime

import papermill as pm

if __name__ == '__main__':
    # python evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/4
    parser = argparse.ArgumentParser(description='Run a notebook')
    parser.add_argument('--notebook', type=str, help='Notebook to run', default='/home/ashesh.ashesh/code/Disentangle/disentangle/notebooks/CalibrationCoverage.ipynb')
    parser.add_argument('--outputdir', type=str, help='Output notebook directory', default='/group/jug/ashesh/EnsDeLyon/notebook_results/')
    parser.add_argument('--ckpt_dir', type=str, help='Checkpoint to use. eg. /group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/4')
    parser.add_argument('--mmse_count', type=int, help='Number of mmse values to generate', default=10)
    parser.add_argument('--elem_size', type=int, help='Number of timesteps to use', default=10)
    parser.add_argument('--tag_time_flag', type=bool, help='Tag time flag', default=False)
    args = parser.parse_args()

    # get a year-month-day hour-minute formatted string
    param_str = f"Elem-{args.elem_size}_MMSE-{args.mmse_count}"
    ckpt_dir = args.ckpt_dir
    model_token = '-'.join(ckpt_dir.strip('/').split('/')[-3:])
    outputdir = os.path.join(args.outputdir, model_token)
    fname = os.path.basename(args.notebook)
    fname = fname.replace('.ipynb','')
    if args.tag_time_flag:
        now = datetime.now().strftime("%Y%m%d.%H.%M")
        fname = f"{fname}_{param_str}_{now}.ipynb"
    else:
        fname = f"{fname}_{param_str}.ipynb"

    output_fpath = os.path.join(outputdir, fname)
    output_config_fpath = os.path.join(outputdir,'config', fname.replace('.ipynb','.txt'))
    os.makedirs(os.path.dirname(output_config_fpath), exist_ok=True)
    # save the configuration
    # convert args to dict
    args_dict = vars(args)
    # save as json
    with open(output_config_fpath, 'w') as f:
        f.write(str(args_dict))

    print(output_fpath, '\n', output_config_fpath)
    pm.execute_notebook(
        args.notebook,
        output_fpath,
        parameters = {
            'ckpt_dir': args.ckpt_dir,
            'mmse_count': args.mmse_count,
            'elem_size': args.elem_size,
            'outputdir': outputdir,
            }
    )
    
