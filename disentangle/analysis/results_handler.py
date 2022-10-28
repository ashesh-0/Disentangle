import pickle
import os

class PaperResultsHandler:
    def __init__(
        self,
        output_dir,
patch_size,
grid_size,
mmse_count,
skip_last_pixels,):
        self._outdir = output_dir
        self._patchN = patch_size
        self._gridN = grid_size
        self._mmseN = mmse_count
        self._skiplN = skip_last_pixels
    
    def dirpath(self):
        return os.path.join(self._outdir,f'P{self._patchN}_G{self._gridN}_M{self._mmseN}_Sk{self._skiplN}')

    def save(self, ckpt_fpath, ckpt_stats):
        outdir = self.dirpath()
        # os.mkdir(outdir,)
        basename = 'stats_' + os.path.basename(ckpt_fpath)
        output_fpath = os.path.join(outdir,basename)
        with open(output_fpath,'wb') as f:
            pickle.dump(ckpt_stats,f)
        print(f'[{self.__class__.__name__}] Saved to {output_fpath}')
        return output_fpath

    def load(self,output_fpath):
        assert os.path.exists(output_fpath)
        with open(output_fpath,'rb') as f:
            return pickle.load(f)

if __name__ == '__main__':
    output_dir = '.'
    patch_size=23
    grid_size=16
    mmse_count=1
    skip_last_pixels=0

    saver = PaperResultsHandler(        output_dir,
patch_size,
grid_size,
mmse_count,
skip_last_pixels)
    fpath = saver.save('/home/ashesh/asdfdfh.ckpt',{'a':[1,2],'b':[3]})

    print(saver.load(fpath))