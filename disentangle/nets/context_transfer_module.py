"""
Context Transfer module coded following https://www.researchgate.net/publication/331159375_Context-Aware_U-Net_for_Biomedical_Image_Segmentation
"""
import torch.nn as nn
import torch


class ContextTransferModule(nn.Module):
    def __init__(self, tensor_shape):
        super().__init__()
        self.C, self.H, self.W = tensor_shape
        # UP, DOWN, LEFT, RIGHT
        self.weights = torch.zeros((4, self.H, self.W), requires_grad=True)
        self.final_layer = nn.Sequential(nn.Conv2d(4 * self.C, self.C, 1, padding=0), nn.ReLU())

    def set_params_to_same_device_as(self, correct_device_tensor):
        if isinstance(self.weights, torch.Tensor):
            if self.weights.device != correct_device_tensor.device:
                self.weights = self.weights.to(correct_device_tensor.device)

    def get_up_W(self):
        return torch.sigmoid(self.weights[0])

    def get_down_W(self):
        return torch.sigmoid(self.weights[1])

    def get_left_W(self):
        return torch.sigmoid(self.weights[2])

    def get_right_W(self):
        return torch.sigmoid(self.weights[3])

    def up_context(self, inp):
        out = inp.clone()
        assert out.shape[1] == self.C
        assert out.shape[2] == self.H
        assert out.shape[3] == self.W
        w = self.get_up_W()
        for i in range(1, self.H):
            old_version = out[:, :, i]
            new_version = w[i - 1] * out[:, :, i - 1] + old_version
            new_version[new_version < 0] = 0
            out[:, :, i] = new_version
        return out

    def down_context(self, inp):
        out = inp.clone()
        assert out.shape[1] == self.C
        assert out.shape[2] == self.H
        assert out.shape[3] == self.W
        w = self.get_down_W()
        rel_idx = -1
        for i in range(self.H - 2, -1, -1):
            old_version = out[:, :, i]
            new_version = w[i - rel_idx] * out[:, :, i - rel_idx] + old_version
            new_version[new_version < 0] = 0
            out[:, :, i] = new_version
        return out

    def right_context(self, inp):
        out = inp.clone()
        assert out.shape[1] == self.C
        assert out.shape[2] == self.H
        assert out.shape[3] == self.W
        w = self.get_right_W()
        rel_idx = -1
        for i in range(self.W - 2, -1, -1):
            old_version = out[:, :, :, i]
            new_version = w[:, i - rel_idx] * out[:, :, :, i - rel_idx] + old_version
            new_version[new_version < 0] = 0
            out[:, :, :, i] = new_version
        return out

    def left_context(self, inp):
        out = inp.clone()
        assert out.shape[1] == self.C
        assert out.shape[2] == self.H
        assert out.shape[3] == self.W
        w = self.get_left_W()
        rel_idx = 1
        for i in range(1, self.W):
            old_version = out[:, :, :, i]
            new_version = w[:, i - rel_idx] * out[:, :, :, i - rel_idx] + old_version
            new_version[new_version < 0] = 0
            out[:, :, :, i] = new_version
        return out

    def forward(self, inp):
        lc = self.left_context(inp)
        rc = self.right_context(inp)
        uc = self.up_context(inp)
        dc = self.down_context(inp)
        context = torch.cat([lc, rc, uc, dc], dim=1)
        return self.final_layer(context)


if __name__ == '__main__':
    shape = (64, 16, 16)
    cxt = ContextTransferModule(shape)
    inp = torch.rand((32, 64, 16, 16)) * 0.5 - 1
    out = cxt.up_context(inp)
    out1 = cxt(inp)
    # import pdb
    # pdb.set_trace()
