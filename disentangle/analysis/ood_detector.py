import disentangle.loss.ood_metrics as metrics
import faiss
from disentangle.scripts.ood_evaluator import get_shape, load_normalized_features


class OODDetector:
    """
    This class loads the in-distribution features and computes the k_nearest neighbors distance.
    For evaluation, given features, it computes the score based on the distance to the nearest neighbors in the in-distribution set.
    """
    def __init__(self, indistribution_fpath, k_nearest=50):
        self.indistribution_fpath = indistribution_fpath
        self.in_feat = load_normalized_features(self.indistribution_fpath, shape=get_shape(self.indistribution_fpath))
        self.index = faiss.IndexFlatL2(self.in_feat.shape[1])
        self.index.add(self.in_feat)
        self.K = k_nearest
        self._indis_scores = self.get_score(self.in_feat)
        print(f'Loaded in-distribution features from {self.indistribution_fpath} with shape {self.in_feat.shape} and added to index.')
    

    def get_score(self, features):
        D_in, _ = self.index.search(features, self.K)
        scores_in = -D_in[:,-1]
        return scores_in

    def OOD_detector_score(self, features):
        scores = self.get_score(features)
        results = metrics.cal_metric(self._indis_scores, scores)
        return results
    
