"""
We assume that the feature files are stored in a memory-mapped format. (run feature_extractor.py first)
"""

import os

import numpy as np

import disentangle.loss.ood_metrics as metrics
import faiss


def load_normalized_features(raw_feature_fpath:str, shape:tuple):
    fname = os.path.basename(raw_feature_fpath)
    norm_feature_fpath = os.path.join(os.path.dirname(raw_feature_fpath), f'norm_{fname}')
    if os.path.exists(norm_feature_fpath):
        print(f'Loading normalized features from {norm_feature_fpath}')
        return np.memmap(norm_feature_fpath, dtype=float, mode='r', shape=shape)
    else:
        normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
        print(f'Normalizing features from {raw_feature_fpath} and saving to {norm_feature_fpath}')
        data = np.memmap(raw_feature_fpath, dtype=float, mode='r', shape=shape)
        norm_data = np.memmap(norm_feature_fpath, dtype=float, mode='w+', shape=shape)
        norm_data[:] = normalizer(data)
        return norm_data

def get_shape(feature_fpath: str):
    """
    Get the shape of the feature file.
    """
    datadir = os.path.dirname(feature_fpath)
    fname = 'shape_' + os.path.basename(feature_fpath)
    shape_fpath = os.path.join(datadir, fname)
    return tuple(np.load(shape_fpath, allow_pickle=True))

def evaluate(indistribution_feature_fpath, ood_feature_fpath):
    in_feat = load_normalized_features(indistribution_feature_fpath, shape=get_shape(indistribution_feature_fpath))
    ood_feat = load_normalized_features(ood_feature_fpath, shape=get_shape(ood_feature_fpath))

    ALPHA = 1.00
    for K in [100]:
        rand_ind = np.random.choice(len(in_feat), int(len(in_feat) * ALPHA), replace=False)
        index = faiss.IndexFlatL2(in_feat.shape[1])
        index.add(in_feat[rand_ind])

        ################### Using KNN distance Directly ###################
        D, _ = index.search(in_feat, K, )
        scores_in = -D[:,-1]

        all_results = []
        D, _ = index.search(ood_feat, K)
        scores_ood_test = -D[:,-1]
        results = metrics.cal_metric(scores_in, scores_ood_test)
        all_results.append(results)
        metrics.print_all_results(all_results,['OOD dataset name'], 'KNN')
        print()


