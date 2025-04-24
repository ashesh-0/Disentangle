"""
We want to learn
1. the optimal mixing of the normalized target which yields a closest approximation to the input. 
2. Having found it, we next want to find the optimal linear transformation (mu and sigma) of the target which yields a closest approximation to the input.
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from disentangle.analysis.stitch_prediction import stitch_predictions
from disentangle.core.psnr import RangeInvariantPsnr


def get_best_mixing(normalized_tar_arr, normalized_inp_arr, plot=False):
    """
    Since the network is trained to predict normalized target distribution, we work here with normalized target patches.
    """
    mean_psnr_arr = []
    std_psnr_arr = []
    t_values = np.arange(0.0,1.0, 0.02) 
    for t in tqdm(t_values):
        inp_tiled = [(t *normalized_tar_arr[i][...,0] + (1-t)*normalized_tar_arr[i][...,1]) for i in range(len(normalized_tar_arr))]
        # print(inp_tiled[0].shape, normalized_inp_arr[0].shape)
        psnr_values = [RangeInvariantPsnr(normalized_inp_arr[i]*1.0, inp_tiled[i]).item() for i in range(len(normalized_inp_arr))]
        mean_psnr_arr.append(np.mean(psnr_values))
        std_psnr_arr.append(np.std(psnr_values))

    best_idx = np.argmax(mean_psnr_arr)
    best_t_estimate = t_values[best_idx]
    print(f'Best t value: {best_t_estimate}')
    if plot:
        plt.plot(t_values, mean_psnr_arr)
    return best_t_estimate, mean_psnr_arr[best_idx]

def get_forward_operator_parameters(dset, normalized_tar_patches, normalized_inp_patches, plot=False):
    """
    if [c1,c2] = model(x), and t is the optimal mixing, then 
    (t*c1 + (1-t)*c2)*sigma + mu will be the closest approximation to x.
    We return mu, sigma and t.
    
    """
    tar =  stitch_predictions(normalized_tar_patches, dset)
    inp =  stitch_predictions(normalized_inp_patches, dset)
    inp = [x[...,0] for x in inp]

    mixing_t,_ = get_best_mixing(tar, inp, plot=plot)

    # Now we need to find the best mu and sigma
    estimated_inp_patches = normalized_tar_patches[:,0]*mixing_t + normalized_tar_patches[:,1]*(1-mixing_t)
    N = estimated_inp_patches.shape[0]
    mu_est = np.mean(estimated_inp_patches)
    sigma_est = estimated_inp_patches.reshape(N, -1).std(axis=1).mean()

    mu_act = np.mean(normalized_inp_patches)
    sigma_act = normalized_inp_patches.reshape(N, -1).std(axis=1).mean()

    mu = mu_act - mu_est*(sigma_act/sigma_est)
    sigma = sigma_act/sigma_est    
    return mixing_t, mu, sigma
    # ((est_inp - mu_est)/sigma_est)*sigma_act + mu_act
    # = est_inp*sigma_act/sigma_est + mu_act - mu_est*(sigma_act/sigma_est)