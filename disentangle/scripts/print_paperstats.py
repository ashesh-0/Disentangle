import argparse
import os
import pickle
from time import sleep

from disentangle.analysis.results_handler import PaperResultsHandler


def rnd(obj):
    return f'{obj:.3f}'


def show(ckpt_dir, results_dir, only_test=True, skip_last_pixels=None):
    if ckpt_dir[-1] == '/':
        ckpt_dir = ckpt_dir[:-1]
    if results_dir[-1] == '/':
        results_dir = results_dir[:-1]

    fname = PaperResultsHandler.get_fname(ckpt_dir)
    print(ckpt_dir)
    for dir in sorted(os.listdir(results_dir)):
        if only_test and dir[:4] != 'Test':
            continue
        if skip_last_pixels is not None:
            sktoken = dir.split('_')[-1]
            assert sktoken[:2] == 'Sk'
            if int(sktoken[2:]) != skip_last_pixels:
                continue

        fpath = os.path.join(results_dir, dir, fname)
        # print(fpath)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                out = pickle.load(f)

            print(dir)
            if 'rmse' in out:
                print('RMSE', ' '.join([rnd(x) for x in out['rmse']]))
            if 'psnr' in out:
                print('PSNR', ' '.join([rnd(x) for x in out['psnr']]))
            if 'rangeinvpsnr' in out:
                print('RangeInvPSNR', ' '.join([rnd(x) for x in out['rangeinvpsnr']]))
            if 'ssim' in out:
                print('SSIM', ' '.join(rnd(x) for x in out['ssim']))
            if 'ms_ssim' in out:
                print('MS-SSIM', ' '.join(rnd(x) for x in out['ms_ssim']))
            print('')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('ckpt_dir', type=str)
    # parser.add_argument('results_dir', type=str)
    # parser.add_argument('--skip_last_pixels', type=int)
    # args = parser.parse_args()

    # ckpt_dir = '/home/ubuntu.ubuntu/training/disentangle/2210/D3-M3-S0-L0/117'
    # results_dir = '/home/ubuntu.ubuntu/data/paper_stats/'
    ckpt_dirs = [
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/93',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/88',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/109/',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/125',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/94',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/89',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/128',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/95',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/87',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/130',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/92',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/90',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/115',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/104',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/96',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/126',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/105',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/97',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/127',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/106',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/98',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/129',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/107',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/99',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/135',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/114',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/101',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/133',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/113',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/100',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/132',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/117',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/103',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/120',
        # '/home/ubuntu.ubuntu/training/disentangle/2402/D16-M23-S0-L0/102',
    ]

    for ckpt_dir in ckpt_dirs:
        show(ckpt_dir, '/group/ubuntu/ubuntu/data/paper_stats/', only_test=True, skip_last_pixels=44)
        sleep(1)

    # show(args.ckpt_dir, args.results_dir, only_test=True, skip_last_pixels=args.skip_last_pixels)
