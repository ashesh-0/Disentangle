import numpy as np


def gaussuian_filter(kernel_size, sigma=1, muu=0):

    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size

    x, y = np.meshgrid(np.linspace(-2, 2, kernel_size), np.linspace(-2, 2, kernel_size))
    dst = np.sqrt(x**2 + y**2)

    # lower normal part of gaussian
    normal = 1 / (np.sqrt(2.0 * np.pi) * sigma)

    # Calculating Gaussian filter
    gauss = np.exp(-((dst - muu)**2 / (2.0 * sigma**2))) * normal
    return gauss


def get_gaussian_focus(kernel_size=200, scale_factor=3, sigma=1, qt_background=0.5):
    gaussian = gaussuian_filter(kernel_size=kernel_size, sigma=sigma)
    # [0,1]
    gaussian = (gaussian - gaussian.min()) / gaussian.max()
    # [0, scale_factor]
    gaussian = gaussian * scale_factor
    return gaussian


def get_weight_mask(shape, foreground_mask, kernel_size=200, scale_factor=3, sigma=1, qt_background=0.5):
    factor = np.zeros(shape)
    factor = factor
    gaus_focus = get_gaussian_focus(kernel_size=kernel_size,
                                    scale_factor=scale_factor,
                                    sigma=sigma,
                                    qt_background=qt_background)
    for i in range(20):
        h = np.random.randint(0, shape[0] - kernel_size)
        w = np.random.randint(0, shape[1] - kernel_size)
        factor[h:h + kernel_size,
               w:w + kernel_size] += gaus_focus * foreground_mask[h:h + kernel_size, w:w + kernel_size]

    factor[factor < 1] = 1
    return factor
