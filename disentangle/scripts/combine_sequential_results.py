import argparse
import os

import numpy as np
from tqdm import tqdm

from disentangle.core.tiff_reader import load_tiff, save_tiff

if __name__ == '__main__':
    data_dir = '/group/jug/ashesh/data/paper_stats/All_P128_G64_M50_Sk32/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()
    ckpt_ = args.ckpt
    assert os.path.isdir(ckpt_)

    fname = 'pred_disentangle_' + '_'.join(ckpt_.strip('/').split('/')[-3:]) + '.tif'
    data_dict = {}
    for k in tqdm(range(1000)):
        datafpath = os.path.join(data_dir, f'kth_{k}', fname)
        if not os.path.exists(datafpath):
            continue
        print(datafpath)
        data_dict[k] = load_tiff(datafpath)

    max_id = np.max(list(data_dict.keys()))
    full_data = np.concatenate([data_dict[k] for k in range(max_id + 1)], axis=0)
    output_fpath = os.path.join(data_dir, fname)
    save_tiff(output_fpath, full_data)
    print(f'Saved to {output_fpath}')
