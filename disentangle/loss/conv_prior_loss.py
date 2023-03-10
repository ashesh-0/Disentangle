import torch


def convolution_prior_loss(conv_weight, min_loss):
    """
    min_loss is needed to ensure that we don't make the middle element too big or other elements too small.
    """
    conv_weight = conv_weight**2
    h, w = conv_weight.shape[-2:]
    assert h % 2 == 1
    assert w % 2 == 1
    hmid = h // 2
    wmid = w // 2
    return torch.clip(torch.mean(conv_weight - conv_weight[:, :, hmid:hmid + 1, wmid:wmid + 1]), min=min_loss)
