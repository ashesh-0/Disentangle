"""
Here, the input image is of multiple resolutions. Target image is the same.
"""
from typing import List, Tuple, Union

import numpy as np
from skimage.transform import resize
from disentangle.core.data_split_type import DataSplitType

from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from disentangle.core.data_type import DataType
from disentangle.data_loader.patch_index_manager import GridAlignement

class MultiScaleTiffDloader(MultiChDeterministicTiffDloader):

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
        grid_alignment = GridAlignement.LeftTop,
        overlapping_padding_kwargs = None,
    ):
        """
        Args:
            num_scales: The number of resolutions at which we want the input. Note that the target is formed at the
                        highest resolution.
        """
        self._padding_kwargs = padding_kwargs  # mode=padding_mode, constant_values=constant_value
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
                         grid_alignment=grid_alignment,
                         overlapping_padding_kwargs=overlapping_padding_kwargs)
        self.num_scales = num_scales
        assert self.num_scales is not None
        self._scaled_data = [self._data]
        assert isinstance(self.num_scales, int) and self.num_scales >= 1
        # self.enable_padding_while_cropping is used only for overlapping_dloader. This is a hack and at some point be
        # fixed properly
        self.enable_padding_while_cropping = False
        self._lowres_supervision = lowres_supervision
        assert isinstance(self._padding_kwargs, dict)
        assert 'mode' in self._padding_kwargs

        for _ in range(1, self.num_scales):
            shape = self._scaled_data[-1].shape
            assert len(shape) == 4
            new_shape = (shape[0], shape[1] // 2, shape[2] // 2, shape[3])
            ds_data = resize(self._scaled_data[-1], new_shape)
            self._scaled_data.append(ds_data)

    def _init_msg(self):
        msg = super()._init_msg()
        msg += f' Pad:{self._padding_kwargs}'
        return msg

    def _load_scaled_img(self, scaled_index, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx, _ = index
        imgs = self._scaled_data[scaled_index][idx % self.N]
        return tuple([imgs[None, :, :, i] for i in range(imgs.shape[-1])])

    def _crop_img(self, img: np.ndarray, h_start: int, w_start: int):
        """
        Here, h_start, w_start could be negative. That simply means we need to pick the content from 0. So,
        the cropped image will be smaller than self._img_sz * self._img_sz
        """
        h_end = h_start + self._img_sz
        w_end = w_start + self._img_sz
        h_start = max(0, h_start)
        w_start = max(0, w_start)
        new_img = img[..., h_start:h_end, w_start:w_end]
        return new_img

    def _get_img(self, index: int):
        """
        Loads an image.
        Crops the image such that cropped image has content.
        """
        img_tuples = self._load_img(index)
        assert self._img_sz is not None
        h, w = img_tuples[0].shape[-2:]
        if self._enable_random_cropping:
            h_start, w_start = self._get_random_hw(h, w)
        else:
            h_start, w_start = self._get_deterministic_hw(index)

        cropped_img_tuples = [self._crop_flip_img(img, h_start, w_start, False, False) for img in img_tuples]

        h_center = h_start + self._img_sz // 2
        w_center = w_start + self._img_sz // 2
        allres_versions = {i: [cropped_img_tuples[i]] for i in range(len(cropped_img_tuples))}
        for scale_idx in range(1, self.num_scales):
            # img1, img2 = self._load_scaled_img(scale_idx, index)
            scaled_img_tuples = self._load_scaled_img(scale_idx, index)

            h_center = h_center // 2
            w_center = w_center // 2
            # img1_padded = np.zeros_like(img1_versions[-1])
            # img2_padded = np.zeros_like(img2_versions[-1])
            h_start = h_center - self._img_sz // 2
            w_start = w_center - self._img_sz // 2

            scaled_cropped_img_tuples = [
                self._crop_flip_img(img, h_start, w_start, False, False) for img in scaled_img_tuples
            ]

            h_start = max(0, -h_start)
            w_start = max(0, -w_start)
            h_end = h_start + scaled_cropped_img_tuples[0].shape[1]
            w_end = w_start + scaled_cropped_img_tuples[0].shape[2]
            if self.enable_padding_while_cropping:
                for ch_idx in range(len(img_tuples)):
                    assert scaled_cropped_img_tuples[ch_idx].shape == allres_versions[ch_idx][-1].shape
                    # assert img2_cropped.shape == img2_versions[-1].shape
                    allres_versions[ch_idx].append(scaled_cropped_img_tuples[ch_idx])

            else:
                h_max, w_max = allres_versions[0][-1].shape[1:]
                padding = np.array([[0, 0], [h_start, h_max - h_end], [w_start, w_max - w_end]])

                for ch_idx in range(len(img_tuples)):
                    if ch_idx + 1 < len(img_tuples):
                        assert allres_versions[ch_idx + 1][-1].shape == allres_versions[ch_idx][-1].shape
                    allres_versions[ch_idx].append(
                        np.pad(scaled_cropped_img_tuples[ch_idx], padding, **self._padding_kwargs))

        output_img_tuples = tuple([np.concatenate(allres_versions[ch_idx]) for ch_idx in range(len(img_tuples))])
        return output_img_tuples

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        img_tuples = self._get_img(index)
        assert self._enable_rotation is False

        if self._lowres_supervision:
            target = np.concatenate([img[:, None] for img in img_tuples], axis=1)
        else:
            target = np.concatenate([img[:1] for img in img_tuples], axis=0)

        if self._normalized_input:
            img_tuples = self.normalize_img(*img_tuples)

        inp = 0
        for img in img_tuples:
            inp += img / (len(img_tuples))

        inp = inp.astype(np.float32)

        if isinstance(index, int):
            return inp, target

        _, grid_size = index
        return inp, target, grid_size
