"""
Here, the input image is of multiple resolutions. Target image is the same.
"""
from typing import List, Tuple, Union

import numpy as np
from skimage.transform import resize

from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.patch_index_manager import TilingMode
from disentangle.data_loader.vanilla_dloader import MultiChDloader


class LCMultiChDloader(MultiChDloader):

    def __init__(
        self,
        data_config,
        fpath: str,
        datasplit_type: DataSplitType = None,
        val_fraction=None,
        test_fraction=None,
        normalized_input=None,
        enable_rotation_aug: bool = False,
        use_one_mu_std=None,
        num_scales: int = None,
        enable_random_cropping=False,
        padding_kwargs: dict = None,
        allow_generation: bool = False,
        lowres_supervision=None,
        max_val=None,
        tiling_mode=TilingMode.ShiftBoundary,
        overlapping_padding_kwargs=None,
        print_vars=True,
    ):
        """
        Args:
            num_scales: The number of resolutions at which we want the input. Note that the target is formed at the
                        highest resolution.
        """
        self._padding_kwargs = padding_kwargs  # mode=padding_mode, constant_values=constant_value
        self._uncorrelated_channel_probab =  data_config.get('uncorrelated_channel_probab', 0.5)

        if overlapping_padding_kwargs is not None:
            assert self._padding_kwargs == overlapping_padding_kwargs, 'During evaluation, overlapping_padding_kwargs should be same as padding_args. \
                It should be so since we just use overlapping_padding_kwargs when it is not None'

        else:
            overlapping_padding_kwargs = padding_kwargs

        super().__init__(data_config,
                         fpath,
                         datasplit_type=datasplit_type,
                         val_fraction=val_fraction,
                         test_fraction=test_fraction,
                         normalized_input=normalized_input,
                         enable_rotation_aug=enable_rotation_aug,
                         enable_random_cropping=enable_random_cropping,
                         use_one_mu_std=use_one_mu_std,
                         allow_generation=allow_generation,
                         max_val=max_val,
                         tiling_mode=tiling_mode,
                        #  grid_alignment=grid_alignment,
                         overlapping_padding_kwargs=overlapping_padding_kwargs,
                         print_vars=print_vars)
        self.num_scales = num_scales
        assert self.num_scales is not None
        self._scaled_data = [self._data]
        self._scaled_noise_data = [self._noise_data]

        assert isinstance(self.num_scales, int) and self.num_scales >= 1
        self._lowres_supervision = lowres_supervision
        assert isinstance(self._padding_kwargs, dict)
        assert 'mode' in self._padding_kwargs

        for _ in range(1, self.num_scales):
            shape = self._scaled_data[-1].shape
            # assert len(shape) == 4
            new_shape = (*shape[:-3], shape[-3] // 2, shape[-2] // 2, shape[-1])
            ds_data = resize(self._scaled_data[-1].astype(np.float32), new_shape).astype(self._scaled_data[-1].dtype)
            # NOTE: These asserts are important. the resize method expects np.float32. otherwise, one gets weird results.
            assert ds_data.max()/self._scaled_data[-1].max() < 5, 'Downsampled image should not have very different values'
            assert ds_data.max()/self._scaled_data[-1].max() > 0.2, 'Downsampled image should not have very different values'

            self._scaled_data.append(ds_data)
            # do the same for noise
            if self._noise_data is not None:
                noise_data = resize(self._scaled_noise_data[-1], new_shape)
                self._scaled_noise_data.append(noise_data)

    def reduce_data(self, t_list=None, h_start=None, h_end=None, w_start=None, w_end=None):
        assert t_list is not None
        assert h_start is None
        assert h_end is None
        assert w_start is None
        assert w_end is None

        self._data = self._data[t_list].copy()
        self._scaled_data = [self._scaled_data[i][t_list].copy() for i in range(len(self._scaled_data))]

        if self._noise_data is not None:
            self._noise_data = self._noise_data[t_list].copy()
            self._scaled_noise_data = [self._scaled_noise_data[i][t_list].copy() for i in range(len(self._scaled_noise_data))]

        self.N = len(t_list)
        self.set_img_sz(self._img_sz, self._grid_sz)
        print(f'[{self.__class__.__name__}] Data reduced. New data shape: {self._data.shape}')

    def _init_msg(self):
        msg = super()._init_msg()
        msg += f' Pad:{self._padding_kwargs}'
        if self._uncorrelated_channels:
            msg += f' UncorrChProbab:{self._uncorrelated_channel_probab}'
        return msg

    def _load_scaled_img(self, scaled_index, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx, _ = index
        
        # tidx = self.idx_manager.get_t(idx)
        patch_loc_list = self.idx_manager.get_patch_location_from_dataset_idx(idx)
        nidx = patch_loc_list[0]

        imgs = self._scaled_data[scaled_index][nidx]
        imgs = tuple([imgs[None,..., i] for i in range(imgs.shape[-1])])
        if self._noise_data is not None:
            noisedata = self._scaled_noise_data[scaled_index][nidx]
            noise = tuple([noisedata[None,..., i] for i in range(noisedata.shape[-1])])
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            # since we are using this lowres images for just the input, we need to add the noise of the input.
            assert self._lowres_supervision is None or self._lowres_supervision is False
            imgs = tuple([img + noise[0] * factor for img in imgs])
        return imgs

    def _crop_img(self, img: np.ndarray, patch_start_loc:Tuple):
        """
        Here, h_start, w_start could be negative. That simply means we need to pick the content from 0. So,
        the cropped image will be smaller than self._img_sz * self._img_sz
        """
        max_len_vals = list(self.idx_manager.data_shape[1:-1])
        max_len_vals[-2:]= img.shape[-2:]
        return self._crop_img_with_padding(img, patch_start_loc, max_len_vals=max_len_vals)

    def _get_img(self, index: int):
        """
        Returns the primary patch along with low resolution patches centered on the primary patch.
        """
        img_tuples, noise_tuples = self._load_img(index)
        assert self._img_sz is not None
        h, w = img_tuples[0].shape[-2:]
        if self._enable_random_cropping:
            patch_start_loc = self._get_random_hw(h, w)
            if self._5Ddata:
                patch_start_loc = (np.random.choice(img_tuples[0].shape[-3] - self._depth3D),) + patch_start_loc
        else:
            patch_start_loc = self._get_deterministic_loc(index)

        cropped_img_tuples = [self._crop_flip_img(img, patch_start_loc, False, False) for img in img_tuples]
        cropped_noise_tuples = [self._crop_flip_img(noise, patch_start_loc, False, False) for noise in noise_tuples]
        patch_start_loc = list(patch_start_loc)
        h_start, w_start = patch_start_loc[-2], patch_start_loc[-1]
        h_center = h_start + self._img_sz // 2
        w_center = w_start + self._img_sz // 2
        allres_versions = {i: [cropped_img_tuples[i]] for i in range(len(cropped_img_tuples))}
        for scale_idx in range(1, self.num_scales):
            scaled_img_tuples = self._load_scaled_img(scale_idx, index)

            h_center = h_center // 2
            w_center = w_center // 2

            h_start = h_center - self._img_sz // 2
            w_start = w_center - self._img_sz // 2
            patch_start_loc[-2:] = [h_start, w_start]
            scaled_cropped_img_tuples = [
                self._crop_flip_img(img, patch_start_loc, False, False) for img in scaled_img_tuples
            ]
            for ch_idx in range(len(img_tuples)):
                allres_versions[ch_idx].append(scaled_cropped_img_tuples[ch_idx])

        output_img_tuples = tuple([np.concatenate(allres_versions[ch_idx]) for ch_idx in range(len(img_tuples))])
        return output_img_tuples, cropped_noise_tuples

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        img_tuples, noise_tuples = self._get_img(index)
        if self._uncorrelated_channels:
            assert self._input_idx is None, 'Uncorrelated channels is not implemented when there is a separate input channel.'
            if np.random.rand() < self._uncorrelated_channel_probab:
                img_tuples_new = [None] * len(img_tuples)
                img_tuples_new[0] = img_tuples[0]
                for i in range(1, len(img_tuples)):
                    new_index = np.random.randint(len(self))
                    img_tuples_tmp, _ = self._get_img(new_index)
                    img_tuples_new[i] = img_tuples_tmp[i]
                img_tuples = img_tuples_new
                

        if self._is_train:
            if self._empty_patch_replacement_enabled:
                if np.random.rand() < self._empty_patch_replacement_probab:
                    img_tuples = self.replace_with_empty_patch(img_tuples)


        if self._enable_rotation:
            img_tuples, noise_tuples = self._rotate(img_tuples, noise_tuples)

        assert self._lowres_supervision != True
        # add noise to input
        if len(noise_tuples) > 0:
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            input_tuples = []
            for x in img_tuples:
                # NOTE: other LC levels already have noise added. So, we just need to add noise to the highest resolution.
                x = x.copy() # make a copy so that we don't modify the original image.
                x[0] = x[0] + noise_tuples[0] * factor
                input_tuples.append(x)
        else:
            input_tuples = img_tuples

        inp, alpha = self._compute_input(input_tuples)
        # assert self._alpha_weighted_target in [False, None]
        target_tuples = [img[:1] for img in img_tuples]
        # add noise to target.
        if len(noise_tuples) >= 1:
            target_tuples = [x + noise for x, noise in zip(target_tuples, noise_tuples[1:])]

        target = self._compute_target(target_tuples, alpha)

        output = [inp, target]

        if self._return_alpha:
            output.append(alpha)

        if isinstance(index, int):
            return tuple(output)

        _, grid_size = index
        output.append(grid_size)
        return tuple(output)


if __name__ == '__main__':
    # from disentangle.configs.microscopy_multi_channel_lvae_config import get_config
    import matplotlib.pyplot as plt

    # from disentangle.configs.elisa3D_config import get_config
    # from disentangle.configs.nikola_synthetic_noise_config import get_config
    from disentangle.configs.shroff_3D_config import get_config
    config = get_config()
    config.data.multiscale_lowres_count = 3
    config.data.image_size = 256
    config.data.patch_sampling_prior = 'center'
    padding_kwargs = {'mode': config.data.padding_mode}
    if 'padding_value' in config.data and config.data.padding_value is not None:
        padding_kwargs['constant_values'] = config.data.padding_value

    dset = LCMultiChDloader(config.data,
                            # '/group/jug/ashesh/data/Elisa3D/',
                            # '/group/jug/ashesh/data/nikola_data/20240531/',
                            '/group/jug/ashesh/data/shrofflab',
                            DataSplitType.Train,
                            val_fraction=config.training.val_fraction,
                            test_fraction=config.training.test_fraction,
                            normalized_input=config.data.normalized_input,
                            enable_rotation_aug=config.data.train_aug_rotate,
                            enable_random_cropping=True,
                            use_one_mu_std=config.data.use_one_mu_std,
                            allow_generation=False,
                            num_scales=config.data.multiscale_lowres_count,
                            max_val=None,
                            padding_kwargs=padding_kwargs,
                            overlapping_padding_kwargs=None)

    mean, std = dset.compute_mean_std()
    dset.set_mean_std(mean, std)

    inp, tar = dset[0]
    print(inp.shape, tar.shape)
    if config.data.get('depth3D', 1) > 1:
        inp = inp[:,0]
        tar = tar[:,0]
    print(inp.shape, tar.shape)
    _, ax = plt.subplots(figsize=(10, 2), ncols=5)
    ax[0].imshow(inp[0])
    ax[1].imshow(inp[1])
    ax[2].imshow(inp[2])
    ax[3].imshow(tar[0])
    ax[4].imshow(tar[1])


    inp, tar = dset[0]
    _, ax = plt.subplots(figsize=(10, 6), ncols=5,nrows=3)
    ax[0,0].imshow(inp[0,0])
    ax[0,1].imshow(inp[1,0])
    ax[0,2].imshow(inp[2,0])
    ax[0,3].imshow(tar[0,0])
    ax[0,4].imshow(tar[1,0])

    ax[1,0].imshow(inp[0,1])
    ax[1,1].imshow(inp[1,1])
    ax[1,2].imshow(inp[2,1])
    ax[1,3].imshow(tar[0,1])
    ax[1,4].imshow(tar[1,1])

    ax[2,0].imshow(inp[0,2])
    ax[2,1].imshow(inp[1,2])
    ax[2,2].imshow(inp[2,2])
    ax[2,3].imshow(tar[0,2])
    ax[2,4].imshow(tar[1,2])
