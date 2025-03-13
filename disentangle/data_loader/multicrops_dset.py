"""
Here, we have multiple folders, each containing images of a single channel. 
"""
from collections import defaultdict
from functools import cache

import albumentations as A
import numpy as np

from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.train_val_data import get_train_val_data


def l2(x):
    return np.sqrt(np.mean(np.array(x)**2))


class MultiCropDset:
    def __init__(self,
                 data_config,
                 fpath: str,
                 datasplit_type: DataSplitType = None,
                 val_fraction=None,
                 test_fraction=None,
                 enable_rotation_aug: bool = False,
                 **ignore_kwargs):
        
        assert data_config.input_is_sum == True, "This dataset is designed for sum of images"

        self._img_sz = data_config.image_size
        self._enable_rotation = enable_rotation_aug
        self._background_values = data_config.get('background_values', None)
        self._data_arr = get_train_val_data(data_config,fpath, datasplit_type, val_fraction, test_fraction)
        self._is_train = datasplit_type == DataSplitType.Train
        # remove upper quantiles, crucial for removing puncta
        self.max_val = data_config.get('max_val', None)
        if self.max_val is not None:
            for ch_idx, data in enumerate(self._data_arr):
                if self.max_val[ch_idx] is not None:
                    for idx in range(len(data)):
                        data[idx][data[idx] > self.max_val[ch_idx]] = self.max_val[ch_idx]

        self._alpha_dirac_delta_weight = 0.0
        self._ch1_min_alpha = self._ch1_max_alpha = None
        if self._is_train:
            self._ch1_min_alpha = data_config.get('ch1_min_alpha', None)
            self._ch1_max_alpha = data_config.get('ch1_max_alpha', None)
            self._alpha_dirac_delta_weight = data_config.get('alpha_dirac_delta_weight', 0.0)
            self._alpha_dirac_delta_value = data_config.get('alpha_dirac_delta_value', 0.5)
            assert self._alpha_dirac_delta_weight >= 0.0 and self._alpha_dirac_delta_weight <= 1.0, 'Invalid alpha_dirac_delta_weight'
            if self._ch1_min_alpha is not None:
                assert self._alpha_dirac_delta_value >= self._ch1_min_alpha, 'Invalid alpha_dirac_delta_value'
                assert self._alpha_dirac_delta_value <= self._ch1_max_alpha, 'Invalid alpha_dirac_delta_value'
        
        alpha_str  = f'Alpha: {self._ch1_min_alpha}-{self._ch1_max_alpha}' if self._ch1_min_alpha is not None else 'Alpha: None'
        alpha_str += f' Dirac: {self._alpha_dirac_delta_weight}-{self._alpha_dirac_delta_value}' if self._alpha_dirac_delta_weight > 0.0 else ''

        # remove background values
        if self._background_values is not None:
            final_data_arr = []
            for ch_idx, data in enumerate(self._data_arr):
                data_float = [x.astype(np.float32) for x in data]
                final_data_arr.append([x - self._background_values[ch_idx] for x in data_float])
            self._data_arr = final_data_arr

        self._rotation_transform = None
        if self._enable_rotation:
            self._rotation_transform = A.Compose([A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90()])
        
        print(f'{self.__class__.__name__} N:{len(self)} Rot:{self._enable_rotation} Ch:{len(self._data_arr)} MaxVal:{self.max_val} Bg:{self._background_values}', alpha_str)


    def compute_mean_std(self):
        mean_tar_dict = defaultdict(list)
        std_tar_dict = defaultdict(list)
        mean_inp = []
        std_inp = []
        for _ in range(30000):
            crops = []
            for ch_idx in range(len(self._data_arr)):
                crop = self.sample_crop(ch_idx)
                mean_tar_dict[ch_idx].append(np.mean(crop))
                std_tar_dict[ch_idx].append(np.std(crop))
                crops.append(crop)

            inp = 0
            for img in crops:
                inp += img

            mean_inp.append(np.mean(inp))
            std_inp.append(np.std(inp))

        output_mean = defaultdict(list)
        output_std = defaultdict(list)
        NC = len(self._data_arr)
        for ch_idx in range(NC):
            output_mean['target'].append(np.mean(mean_tar_dict[ch_idx]))
            output_std['target'].append(l2(std_tar_dict[ch_idx]))
        
        output_mean['target'] = np.array(output_mean['target']).reshape(-1,NC,1,1)
        output_std['target'] = np.array(output_std['target']).reshape(-1,NC,1,1)

        output_mean['input'] = np.array([np.mean(mean_inp)]).reshape(-1,1,1,1)
        output_std['input'] = np.array([l2(std_inp)]).reshape(-1,1,1,1)
        return dict(output_mean), dict(output_std)

    def set_mean_std(self, mean_dict, std_dict):
        self._data_mean = mean_dict
        self._data_std = std_dict

    def get_mean_std(self):
        return self._data_mean, self._data_std

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
            img_kwargs[f'img{i}'] = img
        
        
        keys = list(img_kwargs.keys())
        self._rotation_transform.add_targets({k: 'image' for k in keys})
        rot_dic = self._rotation_transform(image=img_tuples[0], **img_kwargs)

        rotated_img_tuples = []
        for i,img in enumerate(img_tuples):
            rotated_img_tuples.append(rot_dic[f'img{i}'])

        
        return rotated_img_tuples

    def _compute_input(self, imgs, alpha_arr):
        # print('Alpha', alpha_arr)
        inp = 0        
        for img,alpha in zip(imgs, alpha_arr):
            inp += img*alpha
        
        inp = (inp - self._data_mean['input'].squeeze())/(self._data_std['input'].squeeze())
        return inp[None]

    def _compute_target(self, imgs):
        return np.stack(imgs)

    def _sample_alpha(self):
        if np.random.rand() < self._alpha_dirac_delta_weight:
            assert len(self._data_arr) == 2, 'Dirac delta sampling only supported for 2 channels'
            return [self._alpha_dirac_delta_value, 2.0 - self._alpha_dirac_delta_value]
        elif self._ch1_max_alpha is not None:
            assert len(self._data_arr) == 2, 'min-max alpha sampling only supported for 2 channels'
            alpha_width = self._ch1_max_alpha - self._ch1_min_alpha
            alpha = np.random.rand() * (alpha_width) + self._ch1_min_alpha
            return [alpha, 2.0 - alpha]
        else:
            return [1.0]*len(self._data_arr)

    def __getitem__(self, idx):
        imgs = self.imgs_for_patch()
        
        if self._enable_rotation:
            imgs = self._rotate(imgs)
        

        alpha = self._sample_alpha()
        inp = self._compute_input(imgs, alpha)
        target = self._compute_target(imgs)
        return inp, target


if __name__ =='__main__':
    import ml_collections as ml
    from disentangle.core.data_type import DataType
    datadir = '/group/jug/ashesh/data/Elisa/patchdataset/'
    data_config = ml.ConfigDict()
    data_config.channel_list = ['puncta','foreground']
    data_config.image_size = 64
    data_config.input_is_sum = True
    data_config.background_values = [100,100]
    data_config.max_val = [None, 137]
    data_config.ch1_min_alpha = 0.5
    data_config.ch1_max_alpha = 1.5
    data_config.data_type = DataType.MultiCropDset
    data = MultiCropDset(data_config,datadir, DataSplitType.Train, val_fraction=0.1, test_fraction=0.1)
    mean, std = data.compute_mean_std()
    data.set_mean_std(mean, std)
    inp, tar = data[0]
    print(data.compute_mean_std())
    print(inp.shape, tar.shape)