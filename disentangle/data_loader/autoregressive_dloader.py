from typing import Tuple, Union

import numpy as np

from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from disentangle.data_loader.patch_index_manager import GridAlignement


class AutoRegressiveDloader(MultiChDeterministicTiffDloader):
    # def __init__(self, data_config,
    #              fpath: str,
    #              datasplit_type: DataSplitType = None,
    #              val_fraction=None,
    #              test_fraction=None,
    #              normalized_input=None,
    #              enable_rotation_aug: bool = False,
    #              enable_random_cropping: bool = False,
    #              use_one_mu_std=None,
    #              allow_generation=False,
    #              max_val=None,
    #              grid_alignment=GridAlignement.LeftTop,
    #              overlapping_padding_kwargs=None):
    #     super().__init__(data_config, fpath, datasplit_type=datasplit_type, val_fraction=val_fraction,
    #                      test_fraction=test_fraction,
    #                      normalized_input=normalized_input,
    #                      enable_rotation_aug=enable_rotation_aug,
    #                      enable_random_cropping=enable_random_cropping,
    #                      use_one_mu_std=use_one_mu_std,
    #                      allow_generation=allow_generation,
    #                      max_val=max_val,
    #                      grid_alignment=grid_alignment,
    #                      overlapping_padding_kwargs=overlapping_padding_kwargs)
    #     self._input_nbrs = data_config.input_nbrs

    def _find_neighboring_crops(self, index, img_tuples, h_start, w_start):
        """
        Given an index, it returns the neighboring crops of the image at that index.
        There can be 4 neighboring crops: left, right, top, bottom.
        """
        t = self.idx_manager.get_t(index)
        index = self.idx_manager.idx_from_hwt(h_start, w_start, t)
        top_nbr_idx = self.idx_manager.get_top_nbr_idx(index)
        if top_nbr_idx is None:
            # TODO: A better thing could be (background_value - mean)/std
            return [np.zeros((1, self._img_sz, self._img_sz))] * len(img_tuples)
        else:
            h_start, w_start, t_nbr = self.idx_manager.hwt_from_idx(top_nbr_idx)
            assert t_nbr == t, f't_nbr: {t_nbr}, t: {t}'
            return [self._crop_flip_img(img, h_start, w_start, False, False) for img in img_tuples]

    def _crop_imgs(self, index, *img_tuples: np.ndarray):
        h, w = img_tuples[0].shape[-2:]
        if self._img_sz is None:
            return (*img_tuples, {'h': [0, h], 'w': [0, w], 'hflip': False, 'wflip': False})

        if self._enable_random_cropping:
            h_start, w_start = self._get_random_hw(h, w)
        else:
            h_start, w_start = self._get_deterministic_hw(index)

        cropped_imgs = []
        for img in img_tuples:
            img = self._crop_flip_img(img, h_start, w_start, False, False)
            cropped_imgs.append(img)

        neighboring_imgs = self._find_neighboring_crops(index, img_tuples, h_start, w_start)

        return (tuple(cropped_imgs), (neighboring_imgs), {
            'h': [h_start, h_start + self._img_sz],
            'w': [w_start, w_start + self._img_sz],
            'hflip': False,
            'wflip': False,
        })

    def __getitem__(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        img_tuples, neighboring_imgs = self._get_img(index)
        if self._empty_patch_replacement_enabled:
            if np.random.rand() < self._empty_patch_replacement_probab:
                img_tuples = self.replace_with_empty_patch(img_tuples)

        if self._enable_rotation:
            # passing just the 2D input. 3rd dimension messes up things.
            raise NotImplementedError(
                'If we rotate then if the neigboring img was at top, then it will no longer be at top.')

        target = np.concatenate(img_tuples, axis=0)
        inp, alpha = self._compute_input(img_tuples)
        neighboring_inp, nbr_alpha = self._compute_input(neighboring_imgs)
        assert nbr_alpha == 0.5
        inp = np.concatenate([inp, neighboring_inp], axis=0)

        output = [inp, target]

        if self._return_alpha:
            output.append(alpha)

        if isinstance(index, int):
            return tuple(output)

        _, grid_size = index
        output.append(grid_size)
        return tuple(output)


if __name__ == '__main__':
    from disentangle.configs.microscopy_multi_channel_lvae_config import get_config
    config = get_config()
    dset = AutoRegressiveDloader(config.data,
                                 '/group/jug/ashesh/data/microscopy/OptiMEM100x014.tif',
                                 DataSplitType.Train,
                                 val_fraction=config.training.val_fraction,
                                 test_fraction=config.training.test_fraction,
                                 normalized_input=config.data.normalized_input,
                                 enable_rotation_aug=config.data.train_aug_rotate,
                                 enable_random_cropping=config.data.deterministic_grid is False,
                                 use_one_mu_std=config.data.use_one_mu_std,
                                 allow_generation=False,
                                 max_val=None,
                                 grid_alignment=GridAlignement.LeftTop,
                                 overlapping_padding_kwargs=None)

    mean, std = dset.compute_mean_std()
    dset.set_mean_std(mean, std)

    inp, target, alpha = dset[0]
