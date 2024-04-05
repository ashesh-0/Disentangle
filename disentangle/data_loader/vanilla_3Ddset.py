from disentangle.data_loader.vanilla_dloader import MultiChDloader
from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.patch_index_manager import GridAlignement, GridIndexManager
from typing import Tuple, Union
import numpy as np

class MultiCh3DDset(MultiChDloader):
    """
    Data loader which returns 3D patches. The depth of the patch is specified by the `depth` parameter in the data_config.
    """
    def __init__(self,
                 data_config,
                 fpath: str,
                 datasplit_type: DataSplitType = None,
                 val_fraction=None,
                 test_fraction=None,
                 normalized_input=None,
                 enable_rotation_aug: bool = False,
                 enable_random_cropping: bool = False,
                 use_one_mu_std=None,
                 allow_generation=False,
                 max_val=None,
                 grid_alignment=GridAlignement.LeftTop,
                 overlapping_padding_kwargs=None,
                 print_vars=True):
        self._depth = data_config.depth
        super().__init__(data_config, fpath, datasplit_type=datasplit_type, val_fraction=val_fraction, test_fraction=test_fraction,
                         normalized_input=normalized_input, enable_rotation_aug=enable_rotation_aug, enable_random_cropping=enable_random_cropping,
                         use_one_mu_std=use_one_mu_std, allow_generation=allow_generation, max_val=max_val, grid_alignment=grid_alignment,
                         overlapping_padding_kwargs=overlapping_padding_kwargs, print_vars=print_vars)
        # depth could be time/z-stack.
        

    def _load_img(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the channels and also the respective noise channels.
        """
        if isinstance(index, int) or isinstance(index, np.int64):
            idx = index
        else:
            idx = index[0]

        tidx = self.idx_manager.get_t(idx)
        imgs = self._data[tidx:tidx + self._depth]
        loaded_imgs = [imgs[None, ..., i] for i in range(imgs.shape[-1])]
        noise = []
        if self._noise_data is not None and not self._disable_noise:
            noise = [
                self._noise_data[tidx:tidx+self._depth][None, ..., i] for i in range(self._noise_data.shape[-1])
            ]
        return tuple(loaded_imgs), tuple(noise)

    def _rotate(self, img_tuples, noise_tuples):
        img_kwargs = {}
        for i in range(len(img_tuples)):
            for j in range(self._depth):
                img_kwargs[f'img{i}_{j}'] = img_tuples[i][0,j]
        noise_kwargs = {}
        for i in range(len(noise_tuples)):
            for j in range(self._depth):
                noise_kwargs[f'noise{i}_{j}'] = noise_tuples[i][0,j]
            
        keys = list(img_kwargs.keys()) + list(noise_kwargs.keys())
        self._rotation_transform.add_targets({k: 'image' for k in keys})
        print(keys)
        rot_dic = self._rotation_transform(image=img_tuples[0][0], **img_kwargs, **noise_kwargs)
        print('rotated', img_kwargs.keys(), noise_kwargs.keys())
        rotated_img_tuples = []
        for i in range(len(img_tuples)):
            rotated_img_tuples.append(np.concatenate([rot_dic[f'img{i}_{j}'][None,None] for j in range(self._depth)], axis=1))

        rotated_noise_tuples = []
        for i in range(len(noise_tuples)):
            rotated_noise_tuples.append(np.concatenate([rot_dic[f'noise{i}_{j}'][None,None] for j in range(self._depth)], axis=1))

        # img_tuples = [rot_dic[f'img{i}'][None] for i in range(len(img_tuples))]
        # noise_tuples = [rot_dic[f'noise{i}'][None] for i in range(len(noise_tuples))]
        return rotated_img_tuples, rotated_noise_tuples

    def set_img_sz(self, image_size, grid_size):
        """
        If one wants to change the image size on the go, then this can be used.
        Args:
            image_size: size of one patch
            grid_size: frame is divided into square grids of this size. A patch centered on a grid having size `image_size` is returned.
        """

        self._img_sz = image_size
        self._grid_sz = grid_size
        shape = self._data.shape

        self.idx_manager = GridIndexManager((shape[0] - self._depth, *shape[1:]), self._grid_sz, self._img_sz, self._grid_alignment)
        self.set_repeat_factor()

    def __len__(self):
        return (self.N -self._depth) * self._repeat_factor



if __name__ == '__main__':
    from disentangle.configs.pavia_atn_config import get_config
    
    config = get_config()
    dset = MultiCh3DDset(
        config.data,
        '/group/jug/ashesh/data/microscopy/OptiMEM100x014_medium.tif',
        DataSplitType.Train,
        val_fraction=config.training.val_fraction,
        test_fraction=config.training.test_fraction,
        normalized_input=config.data.normalized_input,
        enable_rotation_aug=config.data.normalized_input,
        enable_random_cropping=False,#config.data.deterministic_grid is False,
        use_one_mu_std=config.data.use_one_mu_std,
        allow_generation=False,
        max_val=None,
        grid_alignment=GridAlignement.LeftTop,
        overlapping_padding_kwargs=None)

    mean, std = dset.compute_mean_std()
    dset.set_mean_std(mean, std)

    inp, target = dset[0]
    print(inp.shape, target.shape)

