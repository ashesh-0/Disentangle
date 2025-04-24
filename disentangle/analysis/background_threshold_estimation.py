import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_normalized_target_patches(dset, target_normalizer_fun, num_workers=4, batch_size = 32):
    # Note
    dloader = DataLoader(dset, pin_memory=False, num_workers=num_workers, shuffle=False, batch_size=batch_size)
    tar_patches = []
    for batch in tqdm(dloader):
        _, tar = batch[:2]
        tar_normalized = target_normalizer_fun(tar)
        tar_patches.append(tar_normalized.cpu().numpy())
    tar_patches = np.concatenate(tar_patches, axis=0)
    return tar_patches


def get_background_thresholds(val_dset,target_normalizer_fun, num_workers=4, batch_size=32, skip_percentile=50):
    
    tar_patches = get_normalized_target_patches(val_dset, target_normalizer_fun,
                                                num_workers=num_workers, batch_size=batch_size)
    background_thresholds = [np.percentile(tar_patches[:,i], skip_percentile) for i in range(tar_patches.shape[1])]
    # print(background_thresholds)
    return background_thresholds

def background_patch_detection_func(data, ch_idx):
    threshold = background_thresholds[ch_idx]
    return data.reshape(data.shape[0],-1).mean(axis=-1) < threshold
