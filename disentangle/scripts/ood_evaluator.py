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
        return np.memmap(norm_feature_fpath, dtype=np.float32, mode='r', shape=shape)
    else:
        normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
        print(f'Normalizing features from {raw_feature_fpath} and saving to {norm_feature_fpath}')
        data = np.memmap(raw_feature_fpath, dtype=np.float32, mode='r', shape=shape)
        norm_data = np.memmap(norm_feature_fpath, dtype=np.float32, mode='w+', shape=shape)
        norm_data[:] = normalizer(data)
        return norm_data

def get_shape(feature_fpath: str):
    """
    Get the shape of the feature file.
    /group/jug/ashesh/EnsDeLyon/OOD/2406_D25-M3-S0-L8_10/logvar_Z/hierarchy_2/logvar_Z_2.mmap
    """
    datadir = os.path.dirname(os.path.dirname(feature_fpath))
    fname = 'shape_' + os.path.basename(datadir) + '.txt'
    shape_fpath = os.path.join(datadir, fname)
    with open(shape_fpath, 'r') as f:
        shape_str = f.readline().strip().split(',')
        shape = tuple([int(x) for x in shape_str])
    return shape

def evaluate(indistribution_feature_fpath, ood_feature_fpath):
    in_feat = load_normalized_features(indistribution_feature_fpath, shape=get_shape(indistribution_feature_fpath))
    ood_feat = load_normalized_features(ood_feature_fpath, shape=get_shape(ood_feature_fpath))
    
    in_feat = np.asarray(in_feat, dtype=np.float32)
    ood_feat = np.asarray(ood_feat, dtype=np.float32)
    # breakpoint()

    ALPHA = 1.0
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate OOD detection using KNN distance')
    parser.add_argument('--ind', type=str,
                        help='Path to the in-distribution feature file')
    parser.add_argument('--ood', type=str,
                        help='Path to the out-of-distribution feature file')

    args = parser.parse_args()

    indistribution_feature_fpath = args.ind
    ood_feature_fpath = args.ood

    evaluate(indistribution_feature_fpath, ood_feature_fpath)