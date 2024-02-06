"""
Here, we define the calibration metric. This metric measures the calibration of the model's predictions. A model is well-calibrated if the predicted probabilities are close to the true probabilities. We use the Expected Calibration Error (ECE) to measure the calibration of the model. The ECE is defined as the expected value of the difference between the predicted and true probabilities, where the expectation is taken over the bins of the predicted probabilities. The ECE is a scalar value that ranges from 0 to 1, where 0 indicates perfect calibration and 1 indicates the worst calibration. We also provide a function to plot the reliability diagram, which is a visual representation of the calibration of the model.
"""
import numpy as np


class Calibration:

    def __init__(self, num_bins=15, mode='pixelwise'):
        self._bins = num_bins
        self._bin_boundaries = None
        self._mode = mode
        assert mode in ['pixelwise', 'patchwise']
        self._boundary_mode = 'quantile'
        assert self._boundary_mode in ['quantile', 'uniform']
        # self._bin_boundaries = {}

    def logvar_to_std(self, logvar):
        return np.exp(logvar / 2)

    def compute_bin_boundaries(self, predict_logvar):
        if self._boundary_mode == 'quantile':
            boundaries = np.quantile(self.logvar_to_std(predict_logvar), np.linspace(0, 1, self._bins + 1))
            return boundaries
        else:
            min_logvar = np.min(predict_logvar)
            max_logvar = np.max(predict_logvar)
            min_std = self.logvar_to_std(min_logvar)
            max_std = self.logvar_to_std(max_logvar)
        return np.linspace(min_std, max_std, self._bins + 1)

    def compute_stats(self, pred, pred_logvar, target):
        """
        Args:
            pred: np.ndarray, shape (n, h, w, c)
            pred_logvar: np.ndarray, shape (n, h, w, c)
            target: np.ndarray, shape (n, h, w, c)
        """
        self._bin_boundaries = {}
        stats = {}
        for ch_idx in range(pred.shape[-1]):
            stats[ch_idx] = {'bin_count': [], 'rmv': [], 'rmse': [], 'bin_boundaries': None, 'bin_matrix': []}
            pred_ch = pred[..., ch_idx]
            logvar_ch = pred_logvar[..., ch_idx]
            std_ch = self.logvar_to_std(logvar_ch)
            print(std_ch.shape)
            target_ch = target[..., ch_idx]
            if self._mode == 'pixelwise':
                boundaries = self.compute_bin_boundaries(logvar_ch)
                stats[ch_idx]['bin_boundaries'] = boundaries
                bin_matrix = np.digitize(std_ch.reshape(-1), boundaries)
                bin_matrix = bin_matrix.reshape(std_ch.shape)
                stats[ch_idx]['bin_matrix'] = bin_matrix
                error = (pred_ch - target_ch)**2
                for bin_idx in range(self._bins):
                    bin_mask = bin_matrix == bin_idx
                    bin_error = error[bin_mask]
                    bin_size = np.sum(bin_mask)
                    bin_error = np.sqrt(np.sum(bin_error) / bin_size) if bin_size > 0 else None
                    bin_var = np.mean((std_ch[bin_mask]**2))
                    stats[ch_idx]['rmse'].append(bin_error)
                    stats[ch_idx]['rmv'].append(np.sqrt(bin_var))
                    stats[ch_idx]['bin_count'].append(bin_size)
            else:
                raise NotImplementedError(f'Patchwise mode is not implemented yet.')
        return stats
