"""
This module computes the gradients and stores them so that next access is fast.
This can be used to compute gradients of arbitrary order on images. 
Last two dimensions of the data are assumed to be x & y dimension.

grads = GradientFetcher(imgs)
To get d/dx2y3,
grad_x2_y3 = grads[2,3]
 
"""
import numpy as np


class GradientFetcher:
    def __init__(self, data) -> None:
        self._data = data

        self._grad_data = {0: {0: self._data}}

    @staticmethod
    def apply_x_grad(data):
        grad = np.empty(data.shape)
        grad[:] = np.nan
        grad[..., :, 1:] = data[..., :, 1:] - data[..., :, :-1]
        return grad

    @staticmethod
    def apply_y_grad(data):
        grad = np.empty(data.shape)
        grad[:] = np.nan
        grad[..., 1:, :] = data[..., 1:, :] - data[..., :-1, :]
        return grad

    def __getitem__(self, order):
        order_x, order_y = order
        if order_x in self._grad_data and order_y in self._grad_data[order_x]:
            return self._grad_data[order_x][order_y]

        self.compute(order_x, order_y)
        return self._grad_data[order_x][order_y]

    def compute(self, order_x, order_y):
        assert order_y >= 0 and order_x >= 0
        print(order_x, order_y)
        if order_x in self._grad_data:
            if order_y in self._grad_data[order_x]:
                return self._grad_data[order_x][order_y]
            if order_y - 1 not in self._grad_data[order_x]:
                self.compute(order_x, order_y - 1)

            self._grad_data[order_x][order_y] = self.apply_y_grad(self._grad_data[order_x][order_y - 1])
            return self._grad_data[order_x][order_y]

        self._grad_data[order_x] = {}
        self.compute(order_x - 1, order_y)
        self._grad_data[order_x][order_y] = self.apply_x_grad(self._grad_data[order_x - 1][order_y])
        return self._grad_data[order_x][order_y]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    imgs = np.arange(1024).reshape(1, 1, 32, 32)
    plt.imshow(imgs[0, 0])
    grads = GradientFetcher(imgs)
    gradx = grads[1, 0]
    print('next')
    grady = grads[0, 1]
    print('next')
    gradxy = grads[1, 1]
