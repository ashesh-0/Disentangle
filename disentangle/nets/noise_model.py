from disentangle.nets.hist_noise_model import HistNoiseModel
import numpy as np

def get_noise_model(model_config):
    if 'noise_model_path' in model_config and model_config.noise_model_path is not None:
        hist = np.load(model_config.noise_model_path)
        return HistNoiseModel(hist)
    return None