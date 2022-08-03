import numpy as np


def get_frequency_sets():
    w1 = [1, 2, 3]
    w2 = [10, 11, 12]
    return w1, w2


def angle_shift(w1, w2, point):
    """
    Find x such that:    cos(w2*(point +x) = cos(w1*point)
    """
    # there should be two points at which the gradient's value should be same.
    # if I select the correct point, then I don't need to shift
    theta = arccos(cos(w1 * point))
    point_plux_x = theta / w2
    return point_plux_x - point


def generate_one_curve(w1, w2, count):
    r1 = np.arange(count // 2 + 1)
    r2 = r1[:-1] + angle_shift(w1, w2, r1[-1])
    lefthalf = np.sin(w1 * r1)
    righthalf = np.sin(w2 * r2)
    return np.concatenate(lefthalf, righthalf)
