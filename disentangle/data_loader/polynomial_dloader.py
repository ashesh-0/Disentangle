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
    # d/dx(sin(w2*point +d)) = d/dx(sin(w1*point))
    # w2*cos() = w1*cos()
    theta = np.arccos(w1 * np.cos(w1 * point) / w2)
    return theta - w2 * point
    # theta = np.arccos(np.cos(w1 * point))
    # point_plux_x = theta / w2
    # return point_plux_x - point


def generate_one_curve(w1, w2, count, granularity=0.1):
    r1 = np.arange(0, count // 2, granularity)
    shift = angle_shift(w1, w2, r1[-1])
    first_val = r1[-1] + shift / w2
    r2 = np.arange(first_val, first_val + count // 2, granularity)
    lefthalf = np.sin(w1 * r1)
    value_shift = np.sin(w1 * r1[-1]) - np.sin(w2 * r2[0])
    righthalf = np.sin(w2 * r2) + value_shift

    return np.concatenate([lefthalf[:-1], righthalf])


if __name__ == '__main__':
    w1 = 0.05
    w2 = 0.15
    count = 100
    curve = generate_one_curve(w1, w2, count)
    import matplotlib.pyplot as plt

    plt.plot(curve)
    plt.show()
