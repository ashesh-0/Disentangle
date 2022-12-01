"""
SeamlessStitchBase class will ensure the basic functionality
"""


class SeamlessStitchBase:
    def __init__(self, grid_size, stitched_frame):
        self._data = stitched_frame
        self._sz = grid_size
        self._N = stitched_frame.shape[-1] // self._sz
        assert stitched_frame.shape[-1] % self._sz == 0
        # self.params =  Model(self._N)
        # self.opt = torch.optim.SGD(self.params.parameters(), lr=learning_rate)
        # self.loss_metric = nn.L1Loss(reduction='sum')

        # self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt,
        #                                                          'min',
        #                                                          patience=10,
        #                                                          factor=0.5,
        #                                                          threshold_mode='abs',
        #                                                          min_lr=1e-12,
        #                                                          verbose=True)

    def patch_location(self, row_idx, col_idx):
        """
        Top left location of the patch
        """
        return self._sz * row_idx, self._sz * col_idx

    def get_lboundary(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        return self._data[..., h:h + self._sz, w:w + 1]

    def get_rboundary(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        return self._data[..., h:h + self._sz, w + self._sz - 1:w + self._sz]

    def get_tboundary(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        return self._data[..., h:h + 1, w:w + self._sz]

    def get_bboundary(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        return self._data[..., h + self._sz - 1:h + self._sz, w:w + self._sz]

    def get_ch0_offset(self, row_idx, col_idx):
        pass

    def get_data(self):
        return self._data.cpu().numpy()

    def get_output(self):
        data = self.get_data()
        for row_idx in range(self._N):
            for col_idx in range(self._N):
                h, w = self.patch_location(row_idx, col_idx)
                data[0, h:h + self._sz, w:w + self._sz] += self.get_ch0_offset(row_idx, col_idx)
                data[1, h:h + self._sz, w:w + self._sz] -= self.get_ch0_offset(row_idx, col_idx)
        return data