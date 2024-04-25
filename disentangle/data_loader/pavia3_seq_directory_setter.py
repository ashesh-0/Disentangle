import os
import shutil
import argparse

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
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    convert_directory_structure(args.input_dir, args.output_dir)