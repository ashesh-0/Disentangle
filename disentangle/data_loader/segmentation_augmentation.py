import itertools

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import kornia
from deepinv.transform.projective import Homography


def find_objects(mask, img):
    id_list = np.unique(mask)
    # remove 0 
    id_list = id_list[id_list != 0]
    objects = []
    for id in id_list:
        obj_mask = mask == id
        # find a bounding box around the object
        y, x = np.where(obj_mask)
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        # extract the object
        obj = img[min_y:max_y, min_x:max_x]
        obj_mask = obj_mask[min_y:max_y, min_x:max_x]
        # concaatenate the object and the mask
        obj = np.concatenate((obj[None], obj_mask[None]), axis=0)
        objects.append(obj)
    return objects

def rotate(tensor, angle):
    assert isinstance(tensor, torch.Tensor)
    return kornia.geometry.transform.rotate(tensor, angle, 
                                            center=None, 
                                            mode='bilinear', 
                                            padding_mode='zeros', 
                                            align_corners=True)

def random_flip(tensor, p = 0.5):
    assert isinstance(tensor, torch.Tensor)
    tensor = kornia.augmentation.RandomHorizontalFlip(p=p)(tensor)
    tensor = kornia.augmentation.RandomVerticalFlip(p=p)(tensor)
    return tensor

def transform_object(tensor, max_angle=180):
    assert tensor.ndim == 3, "Tensor should be of shape (C, H, W)"
    # rotation.
    angle = np.random.uniform(0, max_angle)*1.0
    h,w = tensor.shape[-2:]
    sz = int(np.ceil(np.sqrt(h**2 + w**2)))
    tensor =torch.Tensor(tensor*1.0)
    tensor = F.pad(tensor, ((sz-w)//2,(sz-w)//2,(sz-h)//2,(sz-h)//2))
    object_rotated = rotate(tensor, torch.Tensor([angle]))
    # torch.where(object_rotated != 0)
    # crop it. 
    idx = object_rotated.nonzero()
    # print(idx.shape)
    x_min = idx[:, -2].min()
    x_max = idx[:, -2].max()
    y_min = idx[:, -1].min()
    y_max = idx[:, -1].max()
    # print(x_min, x_max, y_min, y_max, object_rotated.shape)
    object_rotated = object_rotated[...,x_min:x_max,y_min:y_max]
    # randomly flip, it adds one more dimension.
    object_rotated = random_flip(object_rotated)[0]
    return object_rotated

def get_object_ordering(objects, perm, square_size):
    new_row_loc = []
    next_pos = None
    for i, idx in enumerate(perm):
        if next_pos is None:
            next_pos = objects[idx].shape[1]
        else:
            if next_pos > square_size:
                new_row_loc.append(i)
                next_pos = None
    return new_row_loc        

def get_combined_frame_dims(objects, perm, ordering):
    combined_h = 0
    combined_w = 0
    row_h = 0
    row_w = 0
    i_start = 0
    ordering_full = [x for x in ordering] + [len(objects)]
    for i_end in ordering_full:
        for i in range(i_start, i_end):
            row_h = max(row_h, objects[perm[i]].shape[0])
            row_w += objects[perm[i]].shape[1]
        combined_h += row_h
        combined_w = max(combined_w, row_w)
        row_h = 0
        row_w = 0
        i_start = i_end
    return combined_h, combined_w

def get_background(size, background_patches):
    # create a white background
    idx_list = np.random.permutation(len(background_patches))
    for idx in idx_list:
        patch = background_patches[idx]
        h, w = patch.shape
        if h >= size[0] and w >= size[1]:
            # 
            patch = random_flip(torch.Tensor(patch*1.0))[0,0].numpy()
            print('after random_flip', patch.shape)
            # crop it.
            x_min = np.random.randint(0, h - size[0])
            x_max = x_min + size[0]
            y_min = np.random.randint(0, w - size[1])
            y_max = y_min + size[1]
            return patch[x_min:x_max, y_min:y_max] * 1.0
        elif h >= size[1] and w >= size[0]:
            # rotate by 90 degrees
            patch = rotate(patch, 90) if np.random.rand() > 0.5 else rotate(patch, -90)
            patch = random_flip(torch.Tensor(patch* 1.0))[0,0].numpy()
            print('after random_flip', patch.shape)
            h, w = patch.shape
            # crop it.
            x_min = np.random.randint(0, h - size[1])
            x_max = x_min + size[1]
            y_min = np.random.randint(0, w - size[0])
            y_max = y_min + size[0]
            return patch[x_min:x_max, y_min:y_max] * 1.0
        
    raise ValueError(f"No background patch found that fits the size")

def get_rectrangle_ratio(objects, perm, ordering):
    combined_h, combined_w = get_combined_frame_dims(objects, perm, ordering)
    return max(combined_h / combined_w, combined_w / combined_h)

def render_objects(objects, perm, ordering, background_patches):
    combined_h, combined_w = get_combined_frame_dims(objects, perm, ordering)
    final_frame = get_background((combined_h, combined_w), background_patches)
    ordering_full = [x for x in ordering] + [len(objects)]
    # find the avg value on the boundary
    mean_boundary_values = []
    for obj in objects:
        mask = get_reduced_mask((obj.numpy() > 0)*1.0)
        # print('mask', mask.shape, type(mask))
        b_mask = torch.Tensor(mask_to_boundary(mask)) > 0
        # print('b_mask', b_mask.shape, b_mask.sum(), obj.shape, obj[b_mask].shape)
        mean_boundary_values.append(obj[b_mask].mean())
    mean_boundary_value = np.mean(mean_boundary_values)

    # find the avg value of the background
    factor = mean_boundary_value/final_frame.mean()
    print('Factor', factor)
    final_frame = final_frame * factor
    

    combined_h = 0
    # combined_w = 0
    row_h = 0
    row_w = 0
    i_start = 0
    for i_end in ordering_full:
        for i in range(i_start, i_end):
            h,w = objects[perm[i]].shape
            mask = (objects[perm[i]] > 0) * 1.0
            mask = torch.Tensor(get_reduced_mask(mask.numpy()))
            final_frame[combined_h:combined_h+h, row_w:row_w+w] = (1-mask)*final_frame[combined_h:combined_h+h, row_w:row_w+w] + mask*objects[perm[i]]
            # final_frame[combined_h:combined_h+h, row_w:row_w+w] = mask*objects[perm[i]]
            row_h = max(row_h, objects[perm[i]].shape[0])
            row_w += objects[perm[i]].shape[1]
        combined_h += row_h
        # combined_w = max(combined_w, row_w)
        row_h = 0
        row_w = 0
        i_start = i_end
    return final_frame

def mask_to_boundary(mask, dilation_ratio=0.02):
    """Convert binary mask to boundary mask"""
    h, w = mask.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = int(round(dilation_ratio * img_diag))
    
    # Pad mask to handle borders
    padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    
    # Erode the mask
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(padded_mask, kernel, iterations=dilation)
    
    # Subtract eroded mask from original to get boundaries
    boundary = mask - eroded[1:h+1, 1:w+1]
    return boundary

def get_reduced_mask(mask, dilation_ratio=0.03):
    b_mask = mask_to_boundary(mask, dilation_ratio=dilation_ratio)
    return mask - b_mask
# def combine_objects(objects, background_patches):
#     area = 0
#     for obj in objects:
#         h,w = obj.shape
#         area += h*w

#     square_size = int(np.ceil(np.sqrt(area)))
#     h_max = max([x.shape[0] for x in objects])
#     w_max = max([x.shape[1] for x in objects])
#     square_size = max(max(square_size, h_max), w_max)
#     # find a generator for all permutations from 0 to n-1

#     n = len(objects)  # Change as needed
#     best_perm = None
#     best_ratio = None
#     for perm in itertools.permutations(range(n)):
#         ordering = get_object_ordering(objects, perm, square_size)
#         ratio = get_rectrangle_ratio(objects, ordering)
#         if best_ratio is None or ratio < best_ratio:
#             best_perm = perm
#             best_ratio = ratio

#     combined_img = render_objects(objects, best_perm, square_size, background_patches)
#     return combined_img
