import nd2
import argparse
import os

from nis2pyr.reader import read_nd2file
from disentangle.core.tiff_reader import save_tiff

def load_7D(fpath):    
    print(f'Loading from {fpath}')
    with nd2.ND2File(fpath) as nd2file:
        # Stdout: ND2 dimensions: {'P': 20, 'C': 19, 'Y': 1608, 'X': 1608}; RGB: False; datatype: uint16; legacy: False
        data = read_nd2file(nd2file)
    return data

def load_one_fpath(fpath, channel_idx):
    data = load_7D(fpath)
    # data.shape: (1, 20, 1, 19, 1608, 1608, 1) 
    data = data[0, :, 0, :, :, :, 0]
    # data.shape: (20, 19, 1608, 1608)
    # Here, 20 are different locations and 19 are different channels.
    data = data[:, channel_idx,...]
    # data.shape: (20, 1608, 1608)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str, help='Path to the ND2 file')
    parser.add_argument('channel_idx', type=int, help='Channel index to load')
    parser.add_argument('--outputdir', type=str, default='.')
    args = parser.parse_args()
    data = load_one_fpath(args.fpath, args.channel_idx)
    print(data.shape)
    fname = os.path.basename(args.fpath).replace('.nd2', '.tif')
    tokens = fname.split('.')
    ext = tokens[-1]
    fname = '.'.join(tokens[:-1]) + f'_channel{args.channel_idx}.{ext}'
    print(fname)
    outputpath = os.path.join(args.outputdir, fname)
    print(f'Saving to {outputpath}')
    save_tiff(outputpath, data)