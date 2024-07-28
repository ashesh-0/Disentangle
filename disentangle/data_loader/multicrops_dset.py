"""
Here, we have multiple folders, each containing images of a single channel. 
"""
from collections import defaultdict
from functools import cache
from typing import List

import albumentations as A
import numpy as np

from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.patch_index_manager import GridAlignement
from disentangle.data_loader.train_val_data import get_train_val_data
from disentangle.data_loader.vanilla_dloader import MultiChDloader


class MultiCropDset:
    def __init__(self,
                 data_config,
                 fpath: str,
                 datasplit_type: DataSplitType = None,
                 val_fraction=None,
                 test_fraction=None,
                 enable_rotation_aug: bool = False):
        
        assert data_config.input_is_sum == True, "This dataset is designed for sum of images"

        self._img_sz = data_config.image_size
        self._enable_rotation = enable_rotation_aug
        self._background_values = data_config.get('background_values', None)
        self._data_arr = get_train_val_data(data_config,fpath, datasplit_type, val_fraction, test_fraction)
        
        # remove background values
        if self._background_values is not None:
            final_data_arr = []
            for ch_idx, data in enumerate(self._data_arr):
                data_float = [x.astype(np.float32) for x in data]
                final_data_arr.append([x - self._background_values[ch_idx] for x in data_float])
            self._data_arr = final_data_arr

        self._rotation_transform = None
        if self._enable_rotation:
            self._rotation_transform = A.Compose([A.Flip(), A.RandomRotate90()])


    def compute_mean_std(self):
        mean_dict = defaultdict(list)
        std_dict = defaultdict(list)

        for ch_idx in range(len(self._data_arr)):
            mean_val = np.mean([x.mean() for x in self._data_arr[ch_idx]])
            max_val = np.max([x.max() for x in self._data_arr[ch_idx]])
            mean_dict['target'].append(mean_val)
            std_dict['target'].append(max_val)

        mean_dict['input'] = np.sum(mean_dict['target'])
        std_dict['input'] = np.sum(std_dict['target'])
        return mean_dict, std_dict

    def set_mean_std(self, mean_dict, std_dict):
        self._data_mean = mean_dict
        self._data_std = std_dict
    
    @cache
    def crop_probablities(self, ch_idx):
        sizes = np.array([np.prod(x.shape) for x in self._data_arr[ch_idx]])
        return sizes/sizes.sum()
    
    def sample_crop(self, ch_idx):
        idx = None
        count = 0
        while idx is None:
            count += 1
            idx = np.random.choice(len(self._data_arr[ch_idx]), p=self.crop_probablities(ch_idx))
            data = self._data_arr[ch_idx][idx]
            if data.shape[0] >= self._img_sz and data.shape[1] >= self._img_sz:
                h = np.random.randint(0, data.shape[0] - self._img_sz)
                w = np.random.randint(0, data.shape[1] - self._img_sz)
                return data[h:h+self._img_sz, w:w+self._img_sz]
            elif count > 100:
                raise ValueError("Cannot find a valid crop")
            else:
                idx = None
        
        return None

    
    def len_per_channel(self, ch_idx):
        return np.sum([np.prod(x.shape) for x in self._data_arr[ch_idx]])/(self._img_sz*self._img_sz)
    
    def imgs_for_patch(self):
        return [self.sample_crop(ch_idx) for ch_idx in range(len(self._data_arr))]

    def __len__(self):
        len_per_channel = [self.len_per_channel(ch_idx) for ch_idx in range(len(self._data_arr))]
        return int(np.max(len_per_channel))

    def _rotate(self, img_tuples):
        return self._rotate2D(img_tuples)

    def _rotate2D(self, img_tuples):
        img_kwargs = {}
        for i,img in enumerate(img_tuples):
            for k in range(len(img)):
                img_kwargs[f'img{i}_{k}'] = img[k]
        
        
        keys = list(img_kwargs.keys())
        self._rotation_transform.add_targets({k: 'image' for k in keys})
        rot_dic = self._rotation_transform(image=img_tuples[0][0], **img_kwargs)

        rotated_img_tuples = []
        for i,img in enumerate(img_tuples):
            if len(img) == 1:
                rotated_img_tuples.append(rot_dic[f'img{i}_0'][None])
            else:
                rotated_img_tuples.append(np.concatenate([rot_dic[f'img{i}_{k}'][None] for k in range(len(img))], axis=0))

        
        return rotated_img_tuples

    def _compute_input(self, imgs):
        inp = np.sum([x[None] for x in imgs],axis=0)
        inp = (inp - self._data_mean['input'])/(self._data_std['input'])
        return inp

    def _compute_target(self, imgs):
        return np.stack(imgs)

    def __getitem__(self, idx):
        imgs = self.imgs_for_patch()
        print(imgs)
        if self._enable_rotation:
            imgs = self._rotate(imgs)
        

        inp = self._compute_input(imgs)
        target = self._compute_target(imgs)
        return inp, target


if __name__ =='__main__':
    import ml_collections as ml
    datadir = '/group/jug/ashesh/data/Elisa/patchdataset/'
    data_config = ml.ConfigDict()
    data_config.channel_list = ['puncta','foreground']
    data_config.image_size = 128
    data_config.input_is_sum = True
    data_config.background_values = [100,100]
    data = MultiCropDset(data_config,datadir, DataSplitType.Train, val_fraction=0.1, test_fraction=0.1)
    mean, std = data.compute_mean_std()
    data.set_mean_std(mean, std)
    inp, tar = data[0]
    print(data.compute_mean_std())
    print(len(data))