import torch
import torch.nn as nn

from disentangle.core.data_utils import pad_img_tensor


class ReplaceLayer(nn.Module):
    """
    Replaces the central portion of large_tensor with small_tensor. 
    """

    def __init__(self, large_tensor_size, small_tensor_size, small_tensor_location, channel_count) -> None:
        super().__init__()
        self._large_sz = large_tensor_size
        self._small_sz = small_tensor_size
        self._loc = small_tensor_location
        self._c = channel_count
        self._mask = torch.ones((1, self._c, self._large_sz[0], self._large_sz[1]))
        self._mask[:, :, self._loc[0]:self._loc[0] + self._small_sz[0],
                   self._loc[1]:self._loc[1] + self._small_sz[1]] = 0

    def forward(self, large_tensor, small_tensor):
        assert large_tensor.shape[-3:] == (self._c, *self._large_sz)
        assert small_tensor.shape[-3:] == (self._c, *self._small_sz)
        small_tensor = pad_img_tensor(small_tensor, large_tensor.shape[-2:])
        return large_tensor * self._mask + small_tensor


if __name__ == '__main__':
    layer = ReplaceLayer((64,64), (32,32),(16,16),12)
    inp = torch.rand(5,12,64,64)
    small_inp = torch.ones(5,12,32,32)
    out = layer(inp, small_inp)