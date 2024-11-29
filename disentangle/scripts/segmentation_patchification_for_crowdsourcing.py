import numpy as np


def patchify_one_frame(SgI: np.ndarray, SgGT: np.ndarray, SgP: np.ndarray, GT: np.ndarray, patch_size=128):    
    """
    Inputs:
        SgI: Segmentation output for the superimposed input frame.
        SgGT: Segmentation output for the GT frame.
        SgP: Segmentation output for the Prediction.
        I: Superimposed input frame.
        G: GT frame.
        P: Prediction frame.
    Outputs:
        A list of patch tuples (G patch, SgI patch, SgGT patch, SgP patch, patch_location)
    """
    assert len(SgI.shape) == 2
    assert SgI.shape == SgGT.shape == SgP.shape == GT.shape, f"Shape Mismatch: SgI{SgI.shape}, SgGT{SgGT.shape}, SgP{SgP.shape}, I{I.shape}, G{G.shape}, P{P.shape}"
    h = np.random.randint(0, SgI.shape[1]-patch_size)
    w = np.random.randint(0, SgI.shape[2]-patch_size)
    output = {
        'GT': GT[:, h:h+patch_size, w:w+patch_size],
        'SgI': SgI[h:h+patch_size, w:w+patch_size],
        'SgGT': SgGT[h:h+patch_size, w:w+patch_size],
        'SgP': SgP[h:h+patch_size, w:w+patch_size],
        'patch_location': [h, w]
    }
    return output


def patchify_one_file_set(SgI_fpath, SgGT_fpath, SgP_fpath, GT_fpath, patch_size=128):
    SgI = np.load(SgI_fpath)
    SgGT = np.load(SgGT_fpath)
    SgP = np.load(SgP_fpath)
    GT = np.load(GT_fpath)
    assert len(SgI.shape) == 3
    assert len(SgGT.shape) == 3
    assert len(SgP.shape) == 3
    assert len(GT.shape) == 3
    num_patches_per_frame = (GT.shape[-1] // patch_size)**2

    assert SgI.shape[0] == SgGT.shape[0] == SgP.shape[0] == GT.shape[0]
    assert SgI.shape[1] == SgGT.shape[1] == SgP.shape[1] == GT.shape[1]
    assert SgI.shape[2] == SgGT.shape[2] == SgP.shape[2] == GT.shape[2]
    patches = {
        'GT': [],
        'SgI': [],
        'SgGT': [],
        'SgP': [],
        'patch_location': []
    }
    for frame_idx in range(SgI.shape[0]):
        for _ in range(num_patches_per_frame):
            one_patch = patchify_one_frame(SgI[frame_idx], SgGT[frame_idx], SgP[frame_idx], GT[frame_idx], patch_size)
            patches['GT'].append(one_patch['GT'])
            patches['SgI'].append(one_patch['SgI'])
            patches['SgGT'].append(one_patch['SgGT'])
            patches['SgP'].append(one_patch['SgP'])
            patches['patch_location'].append([frame_idx] + one_patch['patch_location'])
    return patches

