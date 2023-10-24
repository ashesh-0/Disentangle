"""
This class manages the running average of the solution on training data.
"""
"""
This class manages the running average of the solution on training data.
"""
import os
from typing import List

import numpy as np
import torch
from PIL import Image

from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.patch_index_manager import GridAlignement, GridIndexManager


class Location:

    def __init__(self, topleft_h, topleft_w, time):
        self.h = topleft_h
        self.w = topleft_w
        self.t = time
        assert isinstance(self.h, int)
        assert isinstance(self.w, int)
        assert isinstance(self.t, int)

    def shift_up(self, shift):
        self.h -= shift

    def shift_down(self, shift):
        self.h += shift

    def shift_left(self, shift):
        self.w -= shift

    def shift_right(self, shift):
        self.w += shift


class LocationBasedSolutionRAManager:
    """
    It works with Location objects
    """

    def __init__(self,
                 data_shape,
                 skip_boundary_pixelcount,
                 dump_img_dir=None,
                 dropout=0.0,
                 enable_after_nepoch=-1) -> None:
        """
        data_shape: (T,H,W,C)
        we however wamt to work with (T,C,H,W)
        """
        data_shape = (data_shape[0], data_shape[-1], data_shape[1], data_shape[2])
        self._data = np.zeros(data_shape)
        assert len(data_shape) == 4
        self._skipN = skip_boundary_pixelcount
        self._dump_img_dir = dump_img_dir
        self._dropout = dropout
        self._enable_after_nepoch = enable_after_nepoch
        self._disabled = None

    def update_at_locations(self, batch_predictions, locations: List[Location]):
        H, W = batch_predictions.shape[2:]
        for i, location in enumerate(locations):
            # TODO: fix this. this is intended to not skip the boudnary pixels. right now this will always be self._skipN
            skiphs = self._skipN if location.h >= self._skipN else max(0, 0 - location.h)
            skipws = self._skipN if location.w >= self._skipN else max(0, 0 - location.w)

            # when we are at the boundary, then the slice selected in self._data will be smaller than the fixed size of batch_predictions
            # so, we need to skip the extra pixels. e stands for end.
            skiphe = max(0, location.h + H - self._skipN - self._data.shape[2])
            skipwe = max(0, location.w + W - self._skipN - self._data.shape[3])

            self._data[location.t, :, location.h + skiphs:location.h + H - self._skipN, location.w + skipws:location.w +
                       W - self._skipN] = batch_predictions[i][:, skiphs:H - self._skipN - skiphe,
                                                               skipws:W - self._skipN - skipwe]

    def is_valid_location(self, location, patch_size):
        T, _, H, W = self._data.shape
        return location.h >= 0 and location.h + patch_size <= H and location.w >= 0 and location.w + patch_size <= W and location.t >= 0 and location.t < T

    def update_disabled(self, cur_epoch):
        """
        With the objective of skipping the data for the first few epochs, here we update the self._disabled.
        """
        disabled = (cur_epoch is not None and cur_epoch < self._enable_after_nepoch)
        if self._disabled is None:
            self._disabled = disabled
        elif self._disabled != disabled:
            print(f'[{self.__class__.__name__}] Changing return data state to {disabled}')
            assert disabled is False, 'At later epochs, we should not disable returning the data'
            self._disabled = disabled

    def get_from_locations(self, locations, patch_size, cur_epoch=None, skipdata=False):
        output = []
        self.update_disabled(cur_epoch)
        skipdata = skipdata or self._disabled
        for location in locations:
            if self.is_valid_location(location, patch_size) and skipdata == False:
                output.append(self._data[location.t, :, location.h:location.h + patch_size,
                                         location.w:location.w + patch_size])
            else:
                output.append(np.zeros((self._data.shape[1], patch_size, patch_size)))
        return np.array(output)

    def dump_img(self, mean, std, t=0, downscale_factor=3, epoch=0):
        if not os.path.exists(self._dump_img_dir):
            os.makedirs(self._dump_img_dir)

        fname = f'T{t}_Dfac{downscale_factor}_Epoch{epoch}.png'
        img = self._data[t:(t + 1), :, ::downscale_factor, ::downscale_factor]
        img = (img * std + mean).astype(np.int32)[0]
        im = Image.fromarray(img[0])
        dir0 = os.path.join(self._dump_img_dir, 'Ch0')
        if not os.path.exists(dir0):
            os.makedirs(dir0)
        im.save(os.path.join(dir0, fname))

        im = Image.fromarray(img[1])
        dir1 = os.path.join(self._dump_img_dir, 'Ch1')
        if not os.path.exists(dir1):
            os.makedirs(dir1)
        im.save(os.path.join(dir1, fname))


class SolutionRAManager(LocationBasedSolutionRAManager):

    def __init__(self,
                 datasplit_type: DataSplitType,
                 skip_boundary_pixelcount,
                 patch_size,
                 dump_img_dir=None,
                 dropout=0.0,
                 enable_after_nepoch=-1) -> None:
        if datasplit_type == DataSplitType.Train:
            self._index_manager = GridIndexManager(get_train_instance=True)
        elif datasplit_type == DataSplitType.Val:
            self._index_manager = GridIndexManager(get_val_instance=True)
        elif datasplit_type == DataSplitType.Test:
            self._index_manager = GridIndexManager(get_test_instance=True)
        else:
            raise NotImplementedError()

        super().__init__(self._index_manager.get_data_shape(),
                         skip_boundary_pixelcount,
                         dump_img_dir=dump_img_dir,
                         dropout=dropout,
                         enable_after_nepoch=enable_after_nepoch)
        self._patch_size = patch_size
        print(
            f'[{self.__class__.__name__}] {DataSplitType.name(datasplit_type)} P{self._patch_size} Sk{self._skipN} D{self._dropout}'
        )

    def get_locations(self, indices, grid_sizes):
        assert isinstance(indices, torch.Tensor) and len(indices.shape) == 1

        locations = [
            self._index_manager.topleft_hwt_from_idx(indices[i].item(), grid_size=grid_sizes[i].item())
            for i in range(len(indices))
        ]
        locations = [Location(*location) for location in locations]
        return locations

    def get_top(self, indices, grid_sizes, cur_epoch=None, skipdata=False):
        locations = self.get_locations(indices, grid_sizes)
        for location in locations:
            location.shift_up(self._patch_size)

        return self.get_from_locations(locations, self._patch_size, cur_epoch, skipdata=skipdata)

    def get_bottom(self, indices, grid_sizes, cur_epoch=None, skipdata=False):
        locations = self.get_locations(indices, grid_sizes)
        for location in locations:
            location.shift_down(self._patch_size)

        return self.get_from_locations(locations, self._patch_size, cur_epoch, skipdata=skipdata)

    def get_left(self, indices, grid_sizes, cur_epoch=None, skipdata=False):
        locations = self.get_locations(indices, grid_sizes)
        for location in locations:
            location.shift_left(self._patch_size)

        return self.get_from_locations(locations, self._patch_size, cur_epoch, skipdata=skipdata)

    def get_right(self, indices, grid_sizes, cur_epoch=None, skipdata=False):
        locations = self.get_locations(indices, grid_sizes)
        for location in locations:
            location.shift_right(self._patch_size)

        return self.get_from_locations(locations, self._patch_size, cur_epoch, skipdata=skipdata)

    def get_nbrs(self, indices, grid_sizes, skipdata=None, cur_epoch=None):
        nbr_preds = []
        tmp_skipdata = skipdata == True or np.random.rand() < self._dropout
        nbr_preds.append(self.get_top(indices, grid_sizes, cur_epoch=cur_epoch, skipdata=tmp_skipdata))
        tmp_skipdata = skipdata == True or np.random.rand() < self._dropout
        nbr_preds.append(self.get_bottom(indices, grid_sizes, cur_epoch=cur_epoch, skipdata=tmp_skipdata))
        tmp_skipdata = skipdata == True or np.random.rand() < self._dropout
        nbr_preds.append(self.get_left(indices, grid_sizes, cur_epoch=cur_epoch, skipdata=tmp_skipdata))
        tmp_skipdata = skipdata == True or np.random.rand() < self._dropout
        nbr_preds.append(self.get_right(indices, grid_sizes, cur_epoch=cur_epoch, skipdata=tmp_skipdata))
        return nbr_preds

    def update(self, batch_predictions, indices, grid_sizes):
        locations = self.get_locations(indices, grid_sizes)
        self.update_at_locations(batch_predictions, locations)


if __name__ == '__main__':
    grid_size = 64
    patch_size = 64
    grid_alignment = GridAlignement.LeftTop
    data_shape = (10, 2720, 2720, 2)
    idx_manager = GridIndexManager(data_shape, grid_size, patch_size, grid_alignment, set_train_instance=True)
    N = idx_manager.grid_count()
    indices = torch.Tensor(np.random.randint(0, N, 8)).type(torch.int32)
    sol = SolutionRAManager(DataSplitType.Train, skip_boundary_pixelcount=5, patch_size=patch_size, dump_img_dir='.')
    sol.update(torch.Tensor(np.ones((8, data_shape[-1], patch_size, patch_size))), indices,
               torch.Tensor([grid_size] * 8).type(torch.int32))
    print(sol.get_top(torch.Tensor([0, 1]).type(torch.int32), torch.Tensor([10, 10]).type(torch.int32)))
    sol.dump_img(0, 1)

    ##
    # from disentangle.data_loader.patch_index_manager import GridAlignement, GridIndexManager
    # from skimage.io import imread, imsave
    # data = imread('/home/ubuntu/ashesh/data/microscopy/OptiMEM100x014.tif', plugin='tifffile')
    # data = data[:3].copy()
    # grid_size = 1
    # patch_size = 256
    # grid_alignment = GridAlignement.LeftTop
    # idx_manager= GridIndexManager(data.shape, grid_size, patch_size, grid_alignment, set_train_instance=True)

    # sol = SolutionRAManager(DataSplitType.Train, skip_boundary_pixelcount=0, patch_size=patch_size)
    # sol._data = np.swapaxes(data[:,None],1,4)[...,0]
    # N = idx_manager.grid_count()
    # indices = torch.Tensor(np.array([N//3])).type(torch.int32)
    # grid_sizes = torch.Tensor([grid_size]*len(indices)).type(torch.int32)
    # imgs_top = sol.get_top(indices, grid_sizes)[0,0]
    # imgs_bottom = sol.get_bottom(indices, grid_sizes)[0,0]
    # imgs_left = sol.get_left(indices, grid_sizes)[0,0]
    # imgs_right = sol.get_right(indices, grid_sizes)[0,0]

    # h,w,t= idx_manager.hwt_from_idx(indices[0], grid_size=1)
    # img_center = sol._data[t,0,h:h+patch_size,w:w+patch_size]

    # vmax = np.max([np.max(imgs_top), np.max(imgs_bottom), np.max(imgs_left), np.max(imgs_right), np.max(img_center)])
    # vmin = np.min([np.min(imgs_top), np.min(imgs_bottom), np.min(imgs_left), np.min(imgs_right), np.min(img_center)])
    # print(h,w,t)

    # # batch_predictions = np.zeros((1,patch_size,patch_size,data.shape[-1]))
    # # sol.update(batch_predictions, indices, grid_sizes)
    # # plt.imshow(sol._data[7,0])
    # _,ax = plt.subplots(nrows=3,ncols=3, figsize=(15,15))
    # ax[1,1].imshow(img_center,  vmin=vmin, vmax=vmax)
    # ax[0,1].imshow(imgs_top,  vmin=vmin, vmax=vmax)
    # ax[2,1].imshow(imgs_bottom,  vmin=vmin, vmax=vmax)
    # ax[1,0].imshow(imgs_left,  vmin=vmin, vmax=vmax)
    # ax[1,2].imshow(imgs_right,  vmin=vmin, vmax=vmax)
    # plt.subplots_adjust(wspace=0, hspace=0)
