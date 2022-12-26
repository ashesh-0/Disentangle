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


class GridIndexManager:
    def __init__(self, data_shape, grid_size) -> None:
        self._data_shape = data_shape
        self._default_grid_size = grid_size
        self.N = self._data_shape[0]

    def use_default_grid(self, grid_size):
        return grid_size is None or grid_size < 0

    def grid_count(self, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        repeat_factor = (self._data_shape[-2] // grid_size)**2
        return self.N * repeat_factor

    def hwt_from_idx(self, index, grid_size=None):
        t = self.get_t(index)
        return (*self.get_deterministic_hw(index, grid_size=grid_size), t)

    def get_t(self, index):
        return index % self.N

    def get_top_nbr_idx(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        h, w = self._data_shape[1:3]
        nrows = h // grid_size
        index -= nrows * self.N
        if index < 0:
            return None

        return index

    def get_bottom_nbr_idx(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        h, w = self._data_shape[1:3]
        nrows = h // grid_size
        index += nrows * self.N
        if index > self.grid_count(grid_size=grid_size):
            return None

        return index

    def get_left_nbr_idx(self, index, grid_size=None):
        if self.on_left_boundary(index, grid_size=grid_size):
            return None

        index -= self.N
        return index

    def get_right_nbr_idx(self, index, grid_size=None):
        if self.on_right_boundary(index, grid_size=grid_size):
            return None
        index += self.N
        return index

    def on_left_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        factor = index // self.N
        nrows = self._data_shape[-2] // grid_size

        left_boundary = (factor // nrows) != (factor - 1) // nrows
        return left_boundary

    def on_right_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        factor = index // self.N
        nrows = self._data_shape[-2] // grid_size

        right_boundary = (factor // nrows) != (factor + 1) // nrows
        return right_boundary

    def on_top_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        h, w = self._data_shape[1:3]
        nrows = h // grid_size
        return index < self.N * nrows

    def on_bottom_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        h, w = self._data_shape[1:3]
        nrows = h // grid_size
        return index + self.N * nrows > self.grid_count(grid_size=grid_size)

    def on_boundary(self, idx, grid_size=None):
        if self.on_left_boundary(idx, grid_size=grid_size):
            return True

        if self.on_right_boundary(idx, grid_size=grid_size):
            return True

        if self.on_top_boundary(idx, grid_size=grid_size):
            return True

        if self.on_bottom_boundary(idx, grid_size=grid_size):
            return True
        return False

    def get_deterministic_hw(self, index: int, grid_size=None):
        """
        Fixed starting position for the crop for the img with index `index`.
        """
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        _, h, w, _ = self._data_shape
        assert h == w
        factor = index // self.N
        nrows = h // grid_size

        ith_row = factor // nrows
        jth_col = factor % nrows
        h_start = ith_row * grid_size
        w_start = jth_col * grid_size
        return h_start, w_start
