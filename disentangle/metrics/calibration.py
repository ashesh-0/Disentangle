"""
Here, we define the calibration metric. This metric measures the calibration of the model's predictions. A model is well-calibrated if the predicted probabilities are close to the true probabilities. We use the Expected Calibration Error (ECE) to measure the calibration of the model. The ECE is defined as the expected value of the difference between the predicted and true probabilities, where the expectation is taken over the bins of the predicted probabilities. The ECE is a scalar value that ranges from 0 to 1, where 0 indicates perfect calibration and 1 indicates the worst calibration. We also provide a function to plot the reliability diagram, which is a visual representation of the calibration of the model.
"""

import numpy as np
from scipy import stats

from disentangle.analysis.paper_plots import get_first_index, get_last_index


class Calibration:

    def __init__(self, num_bins=15):
        self._bins = num_bins
        self._bin_boundaries = None

    def logvar_to_std(self, logvar):
        return np.exp(logvar / 2)

    def compute_bin_boundaries(self, predict_std):
        min_std = np.min(predict_std)
        max_std = np.max(predict_std)
        return np.linspace(min_std, max_std, self._bins + 1)

    def compute_stats(self, pred, pred_std, target):
        """
        Args:
            pred: np.ndarray, shape (n, h, w, c)
            pred_std: np.ndarray, shape (n, h, w, c)
            target: np.ndarray, shape (n, h, w, c)
        """
        self._bin_boundaries = {}
        stats_dict = {}
        for ch_idx in range(pred.shape[-1]):
            stats_dict[ch_idx] = {'bin_count': [], 'rmv': [], 'rmse': [], 'bin_boundaries': None, 'bin_matrix': [], 'rmse_err': []}
            pred_ch = pred[..., ch_idx]
            std_ch = pred_std[..., ch_idx]
            target_ch = target[..., ch_idx]
            boundaries = self.compute_bin_boundaries(std_ch)
            stats_dict[ch_idx]['bin_boundaries'] = boundaries
            bin_matrix = np.digitize(std_ch.reshape(-1), boundaries)
            bin_matrix = bin_matrix.reshape(std_ch.shape)
            stats_dict[ch_idx]['bin_matrix'] = bin_matrix
            error = (pred_ch - target_ch)**2
            for bin_idx in range(1, 1+self._bins):
                bin_mask = bin_matrix == bin_idx
                bin_error = error[bin_mask]
                bin_size = np.sum(bin_mask)
                bin_error = np.sqrt(np.sum(bin_error) / bin_size) if bin_size > 0 else None
                stderr = np.std(error[bin_mask]) / np.sqrt(bin_size) if bin_size > 0 else None
                rmse_stderr = np.sqrt(stderr) if stderr is not None else None

                bin_var = np.mean((std_ch[bin_mask]**2))
                stats_dict[ch_idx]['rmse'].append(bin_error)
                stats_dict[ch_idx]['rmse_err'].append(rmse_stderr)
                stats_dict[ch_idx]['rmv'].append(np.sqrt(bin_var))
                stats_dict[ch_idx]['bin_count'].append(bin_size)
        return stats_dict


def get_calibrated_factor_for_stdev(pred, pred_std, target, q_s=0.00001, q_e=0.99999, num_bins=30):
    calib = Calibration(num_bins=num_bins)
    stats_dict = calib.compute_stats(pred, pred_std, target)
    outputs = {}
    for ch_idx in stats_dict.keys():
        y = stats_dict[ch_idx]['rmse']
        x = stats_dict[ch_idx]['rmv']
        count = stats_dict[ch_idx]['bin_count']

        first_idx = get_first_index(count, q_s)
        last_idx = get_last_index(count, q_e)
        x = x[first_idx:-last_idx]
        y = y[first_idx:-last_idx]
        slope, intercept, *_ = stats.linregress(x,y)
        output = {'scalar':slope, 'offset':intercept}
        outputs[ch_idx] = output
    return outputs
