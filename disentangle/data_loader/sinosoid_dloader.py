import os.path
import pickle
from typing import Union

import numpy as np
import math
from tqdm import tqdm
import lzma


def angle_shift(w1, w2, point):
    """
    Find x such that:    cos(w2*(point +x) = cos(w1*point)
    """
    # there should be two points at which the gradient's value should be same.
    # if I select the correct point, then I don't need to shift
    # d/dx(sin(w2*point +d)) = d/dx(sin(w1*point))
    # w2*cos() = w1*cos()
    assert w2 >= w1, 'w2 must be larger than w1. otherwise angle is not always possible'
    theta = np.arccos(w1 * np.cos(w1 * point) / w2)
    return theta - w2 * point


def generate_one_curve(w1, w2, max_angle, granularity=0.1):
    r1 = np.arange(0, max_angle // 2, granularity)
    shift = angle_shift(w1, w2, r1[-1])
    first_val = r1[-1] + shift / w2
    r2 = np.arange(first_val, first_val + max_angle // 2, granularity)
    lefthalf = np.sin(w1 * r1)
    value_shift = np.sin(w1 * r1[-1]) - np.sin(w2 * r2[0])
    righthalf = np.sin(w2 * r2) + value_shift

    y = np.concatenate([lefthalf[:-1], righthalf])
    x = np.concatenate([r1[:-1], r2 - shift / w2])
    return y, x


def apply_rotation(xy, radians):
    """
        Adapted from https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
        Args:
            xy: (2,N)
        """
    c, s = np.cos(radians), np.sin(radians)
    j = np.array([[c, -s], [s, c]])
    m = np.dot(j, xy)
    return np.array(m)


def post_processing(x, curve, img_sz):
    x = x.astype(np.int)
    # x can be < 0 due to horizontal shift.
    x_filtr = np.logical_and(x < img_sz, x >= 0)
    x = x[x_filtr]
    curve = curve[x_filtr]
    curve = curve.astype(np.int)
    y_filtr = curve < img_sz

    curve = curve[y_filtr]
    x = x[y_filtr]
    return x, curve


def rotate_curve(x, curve, rotate_radian):
    shift = (max(x) - min(x)) / 2
    x = x - shift
    x = x.reshape(1, -1)
    curve = curve.reshape(1, -1)
    xy = np.concatenate([x, curve], axis=0)
    xy = apply_rotation(xy, rotate_radian)
    x = xy[0] + shift
    x = x - min(x)
    curve = xy[1]
    return x, curve


def get_img(w_list, img_sz, vertical_shifts: list, horizontal_shifts: list, rotate_radians: list,
            curve_amplitudes: list,
            random_w12_flips: list, curve_thickness):
    assert len(vertical_shifts) == len(rotate_radians)
    assert len(vertical_shifts) == len(curve_amplitudes)
    img = np.zeros((img_sz, img_sz))
    for i in range(len(w_list)):
        w1, w2 = w_list[i]
        add_to_img(img, w1, w2, vertical_shift=vertical_shifts[i], horizontal_shift=horizontal_shifts[i],
                   flip_about_vertical=random_w12_flips[i],
                   rotate_radian=rotate_radians[i], curve_amplitude=curve_amplitudes[i], thickness=curve_thickness)

    return img


def add_thickness(img, thickness, x, curve):
    thickness = (thickness - 1) // 2

    for row_shift in range(-thickness, thickness):
        for col_shift in range(-thickness, thickness):
            if row_shift == 0 and col_shift == 0:
                continue
            temp_curve = curve + col_shift
            temp_x = x + row_shift
            filtr_x = np.logical_and(temp_x > 0, temp_x < img.shape[-1])
            filtr_curve = np.logical_and(temp_curve > 0, temp_curve < img.shape[-1])
            filtr = np.logical_and(filtr_x, filtr_curve)
            img[temp_curve[filtr], temp_x[filtr]] += 1 / (np.sqrt(0.5 * (col_shift ** 2 + row_shift ** 2)))


def add_to_img(img, w1, w2, vertical_shift=None, horizontal_shift: int = 0.0, flip_about_vertical=False,
               rotate_radian=None, curve_amplitude=None,
               thickness=None):
    assert thickness % 2 == 1
    max_angle = img.shape[-1] + abs(int(horizontal_shift))
    granularity = 0.1
    curve, x = generate_one_curve(w1, w2, max_angle, granularity=granularity)
    curve *= curve_amplitude
    if flip_about_vertical:
        min_x = min(x)
        max_x = max(x)
        x = min_x + (max_x - min_x) - (x - min_x)
    # positive
    curve = curve - min(curve)
    # vertical shift
    curve += vertical_shift
    if rotate_radian != 0:
        x, curve = rotate_curve(x, curve, rotate_radian)

    if horizontal_shift:
        x += horizontal_shift
    x, curve = post_processing(x, curve, img.shape[-1])
    img[curve, x] += 1
    add_thickness(img, thickness, x, curve)


class Range:
    def __init__(self, min_val, max_val):
        assert min_val < max_val
        self.min = min_val
        self.max = max_val

    def inrange(self, val):
        return val >= self.min and val < self.max

    def sample(self):
        return np.random.rand() * (self.max - self.min) + self.min


def sample_for_channel1(w_rangelist):
    assert len(w_rangelist) == 4
    if np.random.rand() > 0.5:
        return w_rangelist[0].sample(), w_rangelist[2].sample()
    else:
        return w_rangelist[1].sample(), w_rangelist[3].sample()


def sample_for_channel2(w_rangelist):
    assert len(w_rangelist) == 4
    if np.random.rand() > 0.5:
        return w_rangelist[0].sample(), w_rangelist[3].sample()
    else:
        return w_rangelist[1].sample(), w_rangelist[2].sample()


def spaced_out_vertical_shifts(max_value, num_curves, min_spacing):
    """
    Sometimes the vertical shifts are too close.The idea is to generate them in such a way that they don't
    overlap on each other
    min_spacing: enforces the minimum distance between the start point of the curves
    """
    if num_curves == 1:
        return np.random.rand() * max_value

    bucket_size = 1 / num_curves
    # normalizing min_spacing
    min_spacing = min_spacing / max_value

    assert bucket_size > min_spacing, 'min_spacing is too small'

    # adding bucket_size/10 ensures that 1 also comes in this range.
    disjoint_ranges = np.arange(0, 1 + bucket_size / 10, bucket_size)
    output = []
    range_s = 0
    for range_e in disjoint_ranges[1:]:
        # generate a value between [start_s+min_spacing/2, end_s-min_spacing/2]
        norm_shift = np.random.rand() * (bucket_size - min_spacing) + range_s + min_spacing / 2
        output.append(norm_shift * max_value)
        range_s = range_e
    assert len(output) == num_curves
    return output


def generate_dataset(w_rangelist, size, img_sz, num_curves=3, curve_amplitude=64, max_rotation=math.pi / 8,
                     max_vertical_shift_factor=0.8,
                     max_horizontal_shift_factor=0.3,
                     flip_w12_randomly=False,
                     curve_thickness=31,
                     encourage_non_overlap_single_channel=False,
                     vertical_min_spacing=0
                     ):
    """

    Args:
        w_rangelist:
        size:
        img_sz:
        num_curves:
        curve_amplitude:
        max_rotation:
        max_vertical_shift_factor:
        max_horizontal_shift_factor:
        flip_w12_randomly:
        encourage_non_overlap_single_channel: If True, curves of a single channel are well spaced vertically to prevent
                                overlap. Note that there is overlap of curves between the two channels.
        curve_thickness:

    Returns:

    """
    ch1_dset = []
    ch2_dset = []

    def sample_angle():
        return 2 * np.random.rand() * max_rotation - max_rotation

    def get_random_w12_flips():
        if flip_w12_randomly:
            random_w12_flips = [np.random.rand() > 0.5 for _ in range(num_curves)]
        else:
            random_w12_flips = [False] * num_curves
        return random_w12_flips

    def get_shifts():
        if encourage_non_overlap_single_channel:
            rand_vertical_shifts = spaced_out_vertical_shifts(img_sz * max_vertical_shift_factor, num_curves,
                                                              vertical_min_spacing)
        else:
            rand_vertical_shifts = [np.random.rand() * img_sz * max_vertical_shift_factor for _ in range(num_curves)]
        rand_horizontal_shifts = [np.random.rand() * img_sz * max_horizontal_shift_factor for _ in range(num_curves)]
        rand_horizontal_shifts = [x * -1 if np.random.rand() > 0.5 else x for x in rand_horizontal_shifts]
        return rand_vertical_shifts, rand_horizontal_shifts

    for _ in tqdm(range(size)):
        w1_list = [sample_for_channel1(w_rangelist) for _ in range(num_curves)]
        rotate_radians = [sample_angle() for _ in range(num_curves)]
        vertical_shifts, horizontal_shifts = get_shifts()
        img1 = get_img(w1_list, img_sz, vertical_shifts, horizontal_shifts, rotate_radians,
                       [curve_amplitude] * num_curves,
                       get_random_w12_flips(), curve_thickness)

        w2_list = [sample_for_channel2(w_rangelist) for _ in range(num_curves)]
        vertical_shifts, horizontal_shifts = get_shifts()
        rotate_radians = [sample_angle() for _ in range(num_curves)]
        img2 = get_img(w2_list, img_sz, vertical_shifts, horizontal_shifts, rotate_radians,
                       [curve_amplitude] * num_curves,
                       get_random_w12_flips(), curve_thickness)

        ch1_dset.append(img1[None])
        ch2_dset.append(img2[None])
    return np.concatenate(ch1_dset, axis=0), np.concatenate(ch2_dset, axis=0)


class CustomDataManager:
    """
    A class to manage(load/save) the data.
    """

    def __init__(self, data_dir, data_config):
        self._dir = data_dir
        self._dconfig = data_config

    def fname(self):
        fname = 'sin'
        fname += f'_N-{self._dconfig.total_size}'
        fname += f'_Fsz-{self._dconfig.frame_size}'
        fname += f'_CA-{np.round(self._dconfig.curve_amplitude, 2)}'
        fname += f'_CT-{self._dconfig.curve_thickness}'
        fname += f'_CN-{self._dconfig.num_curves}'
        fname += f'_MR-{self._dconfig.max_rotation}'
        fname += f'_VF-{self._dconfig.max_vshift_factor}'
        fname += f'_HF-{self._dconfig.max_hshift_factor}'
        if self._dconfig.encourage_non_overlap_single_channel:
            fname += f'_NO-{self._dconfig.vertical_min_spacing}'

        fr = self._dconfig.frequency_range_list
        diff = [fr[i][1] - fr[i][0] for i in range(len(fr))]
        gap = [fr[i + 1][0] - fr[i][1] for i in range(len(fr) - 1)]

        diff = int(np.mean(diff) * 100)
        gap = int(np.mean(gap) * 100)
        fname += f'_FR-{diff}.{gap}'
        fname += '.xz'
        return fname

    def exists(self):
        return os.path.exists(os.path.join(self._dir, self.fname()))

    def load(self, fname: Union[str, None] = None):
        fpath = os.path.join(self._dir, self.fname())
        if not os.path.exists(fpath):
            print(f'File {fpath} does not exist.')
            return None

        with lzma.open(fpath, 'rb') as f:
            data_dict = pickle.load(f)
            print(f'Loaded from file {fpath}')

        # Note that simpler arguments are already included in the name itself.
        assert tuple(data_dict['frequency_range_list']) == tuple(self._dconfig.frequency_range_list)
        return data_dict

    def save(self, data_dict):
        data_dict['frequency_range_list'] = self._dconfig.frequency_range_list
        fpath = os.path.join(self._dir, self.fname())
        with lzma.open(fpath, 'wb') as f:
            pickle.dump(data_dict, f)
            print(f'File {fpath} saved.')

    def remove(self):
        fpath = os.path.join(self._dir, self.fname())
        if os.path.exists(fpath):
            os.remove(fpath)


def train_val_data(data_dir, data_config, is_train, val_fraction=None):
    datamanager = CustomDataManager(data_dir, data_config)
    total_size = data_config.total_size
    frequency_range_list = data_config.frequency_range_list
    frame_size = data_config.frame_size
    curve_amplitude = data_config.curve_amplitude
    num_curves = data_config.num_curves
    max_rotation = data_config.max_rotation
    curve_thickness = data_config.curve_thickness
    max_vertical_shift_factor = data_config.max_vshift_factor
    max_horizontal_shift_factor = data_config.max_hshift_factor
    encourage_non_overlap_single_channel = data_config.encourage_non_overlap_single_channel
    if encourage_non_overlap_single_channel:
        vertical_min_spacing = data_config.vertical_min_spacing
    else:
        vertical_min_spacing = 0
    # I think this needs to be True for the data to be only dependant on the pairing. And not who is on left/right.
    flip_w12_randomly = True
    data_dict = None
    if datamanager.exists():
        data_dict = datamanager.load()

    if data_dict is None:
        print('Data not found in the file. generating the data')
        w_rangelist = [Range(x[0], x[1]) for x in frequency_range_list]
        imgs1, imgs2 = generate_dataset(w_rangelist, total_size, frame_size, num_curves=num_curves,
                                        curve_amplitude=curve_amplitude,
                                        max_rotation=max_rotation,
                                        max_vertical_shift_factor=max_vertical_shift_factor,
                                        max_horizontal_shift_factor=max_horizontal_shift_factor,
                                        flip_w12_randomly=flip_w12_randomly,
                                        curve_thickness=curve_thickness,
                                        encourage_non_overlap_single_channel=encourage_non_overlap_single_channel,
                                        vertical_min_spacing=vertical_min_spacing)
        imgs1 = imgs1[..., None]
        imgs2 = imgs2[..., None]
        data = np.concatenate([imgs1, imgs2], axis=3)
        val_size = int(total_size * val_fraction)
        data_dict = {'train': data[val_size:], 'val': data[:val_size], 'frequency_range_list': frequency_range_list}
        datamanager.save(data_dict)

    if is_train:
        return data_dict['train']
    else:
        return data_dict['val']


if __name__ == '__main__':
    w1 = 0.05
    w2 = 0.15
    max_angle = 100
    # curve, x = generate_one_curve(w1, w2, max_angle)
    # x = 2 * x / max_angle - 1
    # # x = np.arange(len(curve))
    # xy = np.concatenate([x.reshape(1, -1), curve.reshape(1, -1)], axis=0)
    # rotated = apply_rotation(xy, math.pi / 200)
    # print(curve.shape)
    import matplotlib.pyplot as plt

    # img = np.zeros((512, 512))
    # vshift = np.random.rand() * img.shape[-1]
    # max_rotate = math.pi / 8
    # rotate = 2 * np.random.rand() * max_rotate - max_rotate
    # add_to_img(img, w1, w2, vertical_shift=vshift, rotate_radian=rotate)
    #
    # vshift = np.random.rand() * img.shape[-1]
    # rotate = 2 * np.random.rand() * max_rotate - max_rotate
    # add_to_img(img, w1, w2, vertical_shift=vshift, rotate_radian=rotate)
    # plt.imshow(img)
    # plt.plot(x, curve)
    # plt.plot(rotated[0], rotated[1], color='r')
    w_rangelist = [Range(0.05, 0.1), Range(0.15, 0.2), Range(0.25, 0.3), Range(0.35, 0.4)]
    size = 10
    img_sz = 512
    imgs1, imgs2 = generate_dataset(w_rangelist, size, img_sz, num_curves=3, curve_amplitude=64,
                                    max_rotation=math.pi / 8, flip_w12_randomly=True)
    plt.imshow(imgs1[0])
    plt.show()
