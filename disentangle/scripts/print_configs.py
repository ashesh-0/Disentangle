import argparse
import os

from disentangle.config_utils import load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    assert os.path.exists(args.config)
    print(load_config(args.config))
