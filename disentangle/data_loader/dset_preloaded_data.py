"""
This dataset takes as input a numpy array of shape (N, H, W, C) as a preloaded dataset.
"""
from disentangle.data_loader.vanilla_dloader import MultiChDloader


class PreloadedDset(MultiChDloader):
    def __init__(self, data, *args, **kwargs):
        self._preloaded_data = data
        super().__init__(*args, **kwargs)
    
    def load_data(self, data_config, datasplit_type, val_fraction=None, test_fraction=None, allow_generation=None):
        self._data = self._preloaded_data.copy()
        self._loaded_data_preprocessing(data_config)
