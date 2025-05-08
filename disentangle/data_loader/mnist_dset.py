import numpy as np
import torch
from sklearn.model_selection import train_test_split

from deepinv.transform.projective import Homography
from disentangle.core.data_split_type import DataSplitType
from finetunesplit.asymmetric_transforms import (DeepinvTransform, HFlip, Identity, Rotate, TransformAllChannels,
                                                 TransformEnum, VFlip)
from finetunesplit.aug_patch_shuffle import GridShuffle
from mnist import MNIST


def get_one_channel_transforms(transform_list, device='cpu'):
    transforms = []
    for transform_dict in transform_list:
        if transform_dict['name'] == TransformEnum.Identity:
            transforms.append(Identity())
        elif transform_dict['name'] == TransformEnum.Rotate:
            transforms.append(Rotate())
        elif transform_dict['name'] == TransformEnum.HFlip:
            transforms.append(HFlip())
        elif transform_dict['name'] == TransformEnum.VFlip:
            transforms.append(VFlip())
        elif transform_dict['name'] == TransformEnum.DeepInV:
            trans_homo = Homography(n_trans = 1, zoom_factor_min=1.0, theta_max=transform_dict['aug_theta_max'], 
                                    theta_z_max=transform_dict['aug_theta_z_max'], 
                                    skew_max=0, 
                                    shift_max=transform_dict['aug_shift_max'],
                                    x_stretch_factor_min = 1,
                                    y_stretch_factor_min = 1,
                                    padding = transform_dict['padding'],
                                    device=device)

            transforms.append(DeepinvTransform(trans_homo))
        elif transform_dict['name'] == TransformEnum.PatchShuffle:
            transforms.append(GridShuffle(transform_dict['patch_size'], transform_dict['grid_size']))
        else:
            raise ValueError(f"Unknown transform")
    return transforms

def get_transform_obj(ch1_transforms_params, ch2_transforms_params, device='cpu'):
    ch1_transforms = get_one_channel_transforms(ch1_transforms_params, device=device)
    ch2_transforms =get_one_channel_transforms(ch2_transforms_params, device=device)
    obj = TransformAllChannels({0: ch1_transforms, 1:ch2_transforms})
    return obj

def filter_images(images, labels, labels_list):
    if labels_list is None:
        return images, labels
    else:
        filtered_images = []
        filtered_labels = []
        for image, label in zip(images, labels):
            if label in labels_list:
                filtered_images.append(image)
                filtered_labels.append(label)
        return filtered_images, filtered_labels
    
def load_mnist_data(data_dir='/group/jug/ashesh/data/MNIST/', datasplit_type=DataSplitType.Train, val_fraction=0.1, labels_list=None):
    mndata = MNIST(data_dir)
    if datasplit_type in [DataSplitType.Train, DataSplitType.Val]:
        images, labels = mndata.load_training()
        images, labels = filter_images(images, labels, labels_list)
        images, val_images, labels, val_labels = train_test_split(images, labels, test_size=val_fraction, stratify=labels, random_state=955)
        # stratify split 
        if datasplit_type == DataSplitType.Val:
            # stratify split
            return val_images, val_labels
        else:
            return images, labels
            
    elif datasplit_type == DataSplitType.Test:
        images, labels = mndata.load_testing()
        images, labels = filter_images(images, labels, labels_list)
        return images, labels
    else:
        raise ValueError(f"Unknown data split type: {datasplit_type}")

class MnistDset:
    def __init__(self, data_config, data_dir, datasplit_type=None, val_fraction=0.1, random_indices=False):
        self.data_dir = data_dir
        self._datasplit_type = datasplit_type
        # Channel 1
        self._ch0_labels_list = data_config.ch0_labels_list
        images, labels = load_mnist_data(data_dir, datasplit_type, val_fraction=val_fraction, labels_list=self._ch0_labels_list)
        self._ch0_images = np.concatenate([np.array(x).reshape(1, 28,28) for x in images], axis=0)
        self._ch0_labels = labels
        # Channel 2
        self._ch1_labels_list = data_config.ch1_labels_list
        images, labels = load_mnist_data(data_dir, datasplit_type, val_fraction=val_fraction, labels_list=self._ch1_labels_list)
        self._ch1_images = np.concatenate([np.array(x).reshape(1, 28,28) for x in images], axis=0)
        self._ch1_labels = labels
        # self._data = np.stack([self._ch0_images, self._ch1_images], axis=-1)
        self._random_indices = random_indices
        # augmentations.
        self.aug = get_transform_obj(data_config.ch1_transforms_params, data_config.ch2_transforms_params)
        self._aug_orig = self.aug
    
    def get_mean_std(self):
        """
        Dummy function to make it work with the rest of the code. this is called to fetch mean,std and passed to the model.
        """
        return {'target': np.array([0.0]), 'input':np.array([0.0])}, {'target': np.array([1.0]), 'input':np.array([1.0])}
    
    def __len__(self):
        return min(len(self._ch0_images), len(self._ch1_images))

    def eval_mode(self):
        self._random_indices = False
        def identity_aug(x, *args, **kwargs):
            return x, None
        self.aug = identity_aug
    
    def train_mode(self):
        self._random_indices = True
        self.aug = self._aug_orig

    def __getitem__(self, idx):
        if self._random_indices is True:
            idx = np.random.randint(0, len(self._ch0_images))
            ch0_image = self._ch0_images[idx]
            idx = np.random.randint(0, len(self._ch1_images))
            ch1_image = self._ch1_images[idx]
        else:
            ch0_image = self._ch0_images[idx]
            ch1_image = self._ch1_images[idx]
        
        ch0_image = torch.Tensor(ch0_image/255.0)
        ch1_image = torch.Tensor(ch1_image/255.0)
        ch0_image,_ = self.aug(ch0_image[None,None], ch_idx =0)
        ch1_image,_ = self.aug(ch1_image[None,None], ch_idx =1)
        
        ch0_image = ch0_image[0]
        ch1_image = ch1_image[0]
        inp = (ch0_image + ch1_image)/2.0
        target = torch.cat((ch0_image, ch1_image), dim=0)
        return inp, target
    