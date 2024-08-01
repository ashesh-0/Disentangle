import os
import shutil
import argparse
from disentangle.data_loader.pavia3_rawdata_loader import load_one_file as load_pavia3_file
from disentangle.core.tiff_reader import save_tiff
import numpy as np

def convert_directory_structure(input_dir, output_dir):
    """
    Convert directory structure from
    /group/jug/ashesh/data/pavia3_sequential/Cond_1/Main/1_001.nd2 to /group/jug/ashesh/data/pavia3_sequential/Cond_1-Main/1_001.nd2
    """
    for condition_subdir in os.listdir(input_dir):
        if not os.path.isdir(os.path.join(input_dir, condition_subdir)):
            continue
        for power_subdir in os.listdir(os.path.join(input_dir, condition_subdir)):
            if not os.path.isdir(os.path.join(input_dir, condition_subdir, power_subdir)):
                continue
            for file in os.listdir(os.path.join(input_dir, condition_subdir, power_subdir)):
                if not file.endswith('.nd2'):
                    continue
                input_file_path = os.path.join(input_dir, condition_subdir, power_subdir, file)
                
                output_file_path = os.path.join(output_dir, condition_subdir + '-' + power_subdir, file)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                shutil.copy(input_file_path, output_file_path)
                print('[Copied] ',input_file_path, '\t', output_file_path)

def squish_to_one_file(input_dir, outputdir):
    """
    Squish all files in input_dir to one file
    """
    assert os.path.isdir(input_dir)
    data_list = []
    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        data = load_pavia3_file(fpath)
        print(data.dtype, data.shape, data[...,0].mean().round(2), '+-', 
              data[...,0].std().round(2), '\t', 
              data[...,1].mean().round(2), '+-', 
              data[...,1].std().round(2))
        data_list.append(data)
    data = np.concatenate(data_list, axis=0)
    
    if input_dir.endswith('/'):
        input_dir = input_dir[:-1]
    output_file = os.path.join(outputdir, os.path.basename(input_dir) + '.tif')
    print(output_file, data.dtype, data.shape)
    save_tiff(output_file, data)


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--squish', action='store_true')
    args = parser.parse_args()
    if args.squish:
        squish_to_one_file(args.input_dir, args.output_dir)
    else:
        convert_directory_structure(args.input_dir, args.output_dir)