"""
We would like to have a common logic to map between an index and location on the image.
We assume the data to be of shape N * H * W * C (C: channels, H,W: spatial dimensions, N: time/number of frames)
We assume the square patches.
The extra content on the right side will not be used( as shown below). 
.-----------.-.
|           | |
|           | |
|           | |
|           | |
.-----------.-.

"""


class PatchIndexManager:
    def __init__(self, data_shape, patch_size) -> None:
        self._data_shape = data_shape
        self._patch_size = patch_size
        self.N = self._data_shape[0]

    def patch_count(self):
        repeat_factor = (self._data_shape[-2] // self._patch_size)**2
        return self.N * repeat_factor

    def hwt_from_idx(self, index, img_sz=None):
        _, H, W, _ = self._data_shape
        t = self.get_t(index)
        return (*self.get_deterministic_hw(index, H, W, img_sz=img_sz), t)

    def get_t(self, index):
        return index % self.N

    def get_top_nbr_idx(self, index):
        h, w = self._data_shape[1:3]
        nrows = h // self._patch_size
        index -= nrows * self.N
        if index < 0:
            return None

        return index

    def get_bottom_nbr_idx(self, index):
        h, w = self._data_shape[1:3]
        nrows = h // self._patch_size
        index += nrows * self.N
        if index > self.patch_count():
            return None

        return index

    def get_left_nbr_idx(self, index):
        if self.on_left_boundary(index):
            return None

        index -= self.N
        return index

    def get_right_nbr_idx(self, index):
        if self.on_right_boundary(index):
            return None
        index += self.N
        return index

    def on_left_boundary(self, index):
        factor = index // self.N
        nrows = self._data_shape[-2] // self._patch_size

        left_boundary = (factor // nrows) != (factor - 1) // nrows
        return left_boundary

    def on_right_boundary(self, index):
        factor = index // self.N
        nrows = self._data_shape[-2] // self._patch_size

        right_boundary = (factor // nrows) != (factor + 1) // nrows
        return right_boundary

    def on_top_boundary(self, index):
        h, w = self._data_shape[1:3]
        nrows = h // self._patch_size
        return index < self.N * nrows

    def on_bottom_boundary(self, index):
        h, w = self._data_shape[1:3]
        nrows = h // self._patch_size
        return index + self.N * nrows > self.patch_count()

    def on_boundary(self, idx):
        if self.on_left_boundary(idx):
            return True

        if self.on_right_boundary(idx):
            return True

        if self.on_top_boundary(idx):
            return True

        if self.on_bottom_boundary(idx):
            return True
        return False

    def get_deterministic_hw(self, index: int, img_sz=None):
        """
        Fixed starting position for the crop for the img with index `index`.
        """
        if img_sz is None:
            img_sz = self._patch_size

        _, h, w, _ = self._data_shape
        assert h == w
        factor = index // self.N
        nrows = h // img_sz

        ith_row = factor // nrows
        jth_col = factor % nrows
        h_start = ith_row * img_sz
        w_start = jth_col * img_sz
        return h_start, w_start
