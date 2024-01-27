"""
Absolute Positional encodings. Adapted from https://github.com/gazelle93/Transformer-Various-Positional-Encoding/blob/main/positional_encoders.py
"""

import torch
import torch.nn as nn


class AbsolutePositionalEncoder(nn.Module):

    def __init__(self, emb_dim, max_position=512):
        super(AbsolutePositionalEncoder, self).__init__()
        self.position = torch.arange(max_position).unsqueeze(1)

        self.positional_encoding = torch.zeros(1, max_position, emb_dim)

        _2i = torch.arange(0, emb_dim, step=2).float()

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 0::2] = torch.sin(self.position / (10000**(_2i / emb_dim)))

        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 1::2] = torch.cos(self.position / (10000**(_2i / emb_dim)))
        self.positional_encoding = self.positional_encoding.cuda()

    def forward(self, start_idx, end_idx):
        return self.positional_encoding[:, start_idx:end_idx, :]


class TwoDimPositionalEncoder(nn.Module):

    def __init__(self, emb_dim, max_x=512, max_y=512):
        super(TwoDimPositionalEncoder, self).__init__()
        assert emb_dim % 2 == 0, 'emb_dim must be even'
        self.max_x = max_x
        self.max_y = max_y
        self.x_encoder = AbsolutePositionalEncoder(emb_dim // 2, max_x)
        self.y_encoder = AbsolutePositionalEncoder(emb_dim // 2, max_y)
        self.encoding = self.get()
        # make the channel 2nd dimension
        self.encoding = self.encoding.permute(0, 3, 1, 2)
        self.encoding = self.encoding.cuda()

    def get(self):
        h, w = self.max_x, self.max_y
        x_enc = self.x_encoder(0, h)
        x_enc = x_enc[:, :, None, :]
        x_enc = x_enc.repeat(1, 1, w, 1)

        y_enc = self.y_encoder(0, w)
        y_enc = y_enc[:, None, :, :]
        y_enc = y_enc.repeat(1, h, 1, 1)
        enc = torch.cat([x_enc, y_enc], dim=-1)
        return enc

    def forward(self, x_start_idx, x_end_idx, y_start_idx, y_end_idx):
        assert x_start_idx < x_end_idx and y_start_idx < y_end_idx
        assert x_start_idx >= 0 and x_end_idx <= self.max_x
        assert y_start_idx >= 0 and y_end_idx <= self.max_y
        return self.encoding[:, :, x_start_idx:x_end_idx, y_start_idx:y_end_idx]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    emb_dim = 16
    enc = TwoDimPositionalEncoder(emb_dim=emb_dim, max_x=64, max_y=64)
    emb = enc(0, 16, 0, 16)
    print(emb.shape)
    _, ax = plt.subplots(figsize=(12, 8), ncols=3, nrows=2)
    ax[0, 0].imshow(emb[0, 0])
    ax[0, 1].imshow(emb[0, emb_dim // 8])
    ax[0, 2].imshow(emb[0, emb_dim // 2 - 1])

    ax[1, 0].imshow(emb[0, 0 + emb_dim // 2])
    ax[1, 1].imshow(emb[0, emb_dim // 8 + emb_dim // 2])
    ax[1, 2].imshow(emb[0, emb_dim // 2 - 1 + emb_dim // 2])
