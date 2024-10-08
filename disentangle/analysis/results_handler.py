import json
import os
import pickle

from disentangle.core.data_split_type import DataSplitType
from disentangle.core.tiff_reader import save_tiff


class PaperResultsHandler:

    def __init__(
        self,
        output_dir,
        eval_datasplit_type,
        patch_size,
        grid_size,
        mmse_count,
        skip_last_pixels,
        predict_kth_frame=None,
        multiplicative_factor=1,
        train_calibration=False,
        epistemic_uncertainty_data_collection=False,
        eval_calibration=False,
        override_kwargs = None,
    ):
        self._dtype = eval_datasplit_type
        self._outdir = output_dir
        self._patchN = patch_size
        self._gridN = grid_size if isinstance(grid_size, int) else '-'.join(map(str, grid_size))
        self._mmseN = mmse_count
        self._skiplN = skip_last_pixels
        self._predict_kth_frame = predict_kth_frame
        self._multiplicative_factor = multiplicative_factor
        self._train_calibration = train_calibration
        self._eval_calibration = eval_calibration
        self._override_kwargs = override_kwargs
        self._epistemic_uncertainty_data_collection = epistemic_uncertainty_data_collection

    def dirpath(self, dtype=None):
        if dtype is None:
            dtype = self._dtype
        path = os.path.join(
            self._outdir,
            f'{DataSplitType.name(dtype)}_P{self._patchN}_G{self._gridN}_M{self._mmseN}_Sk{self._skiplN}')
        if self._multiplicative_factor != 1:
            path += '_F'
        return path

    @staticmethod
    def get_fname(ckpt_fpath):
        assert ckpt_fpath[-1] != '/'
        basename = '_'.join(ckpt_fpath.split("/")[4:]) + '.pkl'
        basename = 'stats_' + basename
        return basename

    @staticmethod
    def get_calib_fname(ckpt_fpath):
        return PaperResultsHandler.get_fname(ckpt_fpath).replace('stats_', 'calib_')

    @staticmethod
    def get_calib_stats_fname(ckpt_fpath):
        return PaperResultsHandler.get_fname(ckpt_fpath).replace('stats_', 'calib_stats_')
    
    @staticmethod
    def get_pred_fname(ckpt_fpath, postfix=''):
        assert ckpt_fpath[-1] != '/'
        basename = '_'.join(ckpt_fpath.split("/")[4:])
        if postfix != '':
            basename = basename + '_' + postfix
        basename += '.tif'
        basename = 'pred_' + basename
        return basename

    def override_kwargs_subdir(self):
        dict_ = json.loads(self._override_kwargs)
        dict_str = ''
        for key in sorted(dict_.keys()):
            dict_str+= f'{key}-{dict_[key]}'
        return dict_str

    def get_output_dir(self, dtype=None):
        outdir = self.dirpath(dtype=dtype)
        if self._epistemic_uncertainty_data_collection:
            outdir = os.path.join(outdir, 'epistemic_uncertainty_data_collection')
            os.makedirs(outdir, exist_ok=True)
        
        if self._predict_kth_frame is not None:
            os.makedirs(outdir, exist_ok=True)
            outdir = os.path.join(outdir, f'kth_{self._predict_kth_frame}')

        if not os.path.isdir(outdir):
            os.mkdir(outdir)


        if self._override_kwargs is not None:
            key = self.override_kwargs_subdir()
            outdir = os.path.join(outdir, key)
            
            if not os.path.isdir(outdir):
                os.mkdir(outdir)

        return outdir

    def get_calib_factors_path(self, ckpt_fpath):
        """ Returns the path to the calibration factors file.
        """
        outdir = self.get_output_dir(DataSplitType.Val)
        return os.path.join(outdir, self.get_calib_fname(ckpt_fpath))


    def get_output_fpath(self, ckpt_fpath):
        outdir = self.get_output_dir()
        if self._train_calibration:
            output_fpath = os.path.join(outdir, self.get_calib_fname(ckpt_fpath))
        elif self._eval_calibration:
            output_fpath = os.path.join(outdir, self.get_calib_stats_fname(ckpt_fpath))
        else:
            output_fpath = os.path.join(outdir, self.get_fname(ckpt_fpath))
        return output_fpath

    def save(self, ckpt_fpath, ckpt_stats):
        output_fpath = self.get_output_fpath(ckpt_fpath)
        with open(output_fpath, 'wb') as f:
            pickle.dump(ckpt_stats, f)
        print(f'[{self.__class__.__name__}] Saved to {output_fpath}')
        return output_fpath

    def get_pred_fpath(self, ckpt_fpath, overwrite):
        suitable_fpath_notfound = True
        postfix = '1'
        while suitable_fpath_notfound:
            fname = self.get_pred_fname(ckpt_fpath, postfix=postfix)
            fpath = os.path.join(self.get_output_dir(), fname)
            suitable_fpath_notfound = os.path.exists(fpath) and not overwrite
            postfix = str(int(postfix) + 1)
        return fpath

    def dump_predictions(self, ckpt_fpath, predictions, hparam_dict, overwrite=True):
        fpath = self.get_pred_fpath(ckpt_fpath, overwrite)
        save_tiff(fpath, predictions)
        print(f'Written {predictions.shape} to {fpath}')
        hparam_fpath = fpath.replace('.tif', '.json')
        with open(hparam_fpath, 'w') as f:
            json.dump(hparam_dict, f)

    def load(self, output_fpath):
        assert os.path.exists(output_fpath)
        with open(output_fpath, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    output_dir = '.'
    patch_size = 23
    grid_size = 16
    mmse_count = 1
    skip_last_pixels = 0

    saver = PaperResultsHandler(output_dir, 1, patch_size, grid_size, mmse_count, skip_last_pixels)
    fpath = saver.save('/home/ashesh.ashesh/training/disentangle/2210/D7-M3-S0-L0/82', {'a': [1, 2], 'b': [3]})

    print(saver.load(fpath))
