import torch

from disentangle.core.custom_enum import Enum


class ConvolutionPriorLossType(Enum):
    # we just say that the middle weight should be larger. we minimize the squared difference.
    Absolute = 0
    # we say that surrounding weights should less than alpha times central in absolute terms.
    FactorBased = 1
    # deeper levels have more strict factor
    MultiStepFactorBased = 2


class ConvolutionPriorLoss:

    def __init__(self, cp_loss_type, rf_clip_val=None, rf_factor=None) -> None:
        self._cp_loss_type = cp_loss_type
        self._mask3 = torch.ones((3, 3), requires_grad=False)
        self._mask3[1, 1] = 0

        self._mask5 = torch.ones((5, 5), requires_grad=False)
        self._mask5[2, 2] = 0
        self.rf_clip_val = rf_clip_val
        self.rf_factor = rf_factor
        if self._cp_loss_type == ConvolutionPriorLossType.Absolute:
            assert self.rf_clip_val is not None
        elif self._cp_loss_type == ConvolutionPriorLossType.FactorBased:
            assert self.rf_factor is not None
        print(
            f'[{self.__class__.__name__}] {ConvolutionPriorLossType.name(self._cp_loss_type)} Clip:{self.rf_clip_val} Factor:{self.rf_factor}'
        )

    
    def get_factor_from_name(self, name):
        if self._cp_loss_type == ConvolutionPriorLossType.FactorBased:
            return self.rf_factor
        elif self._cp_loss_type == ConvolutionPriorLossType.MultiStepFactorBased:
            all_possible_first_tokens = ['bottom_up_layers','final_top_down','first_bottom_up','likelihood','top_down_layers']
            tokens = name.split('.')
            first_token = tokens[0]
            assert first_token in all_possible_first_tokens
            if first_token in ['bottom_up_layers','top_down_layers']:
                level = int(tokens[1])
                pow = 2**level
                return self.rf_factor**(pow)

            else:
                return self.rf_factor

    def get(self, conv_weight, **kwargs):
        if self._cp_loss_type == ConvolutionPriorLossType.Absolute:
            return convolution_prior_loss_absolute(conv_weight, self.rf_clip_val)
        elif self._cp_loss_type in [ConvolutionPriorLossType.FactorBased,ConvolutionPriorLossType.MultiStepFactorBased]:
            if 'factor' in kwargs:
                factor = kwargs['factor']
            else:
                factor = self.rf_factor
            loss_term1 = self.convolution_prior_loss_factor_based(conv_weight, factor)
            loss_term2 = weight_less_than1_loss(conv_weight)
            return loss_term1 + loss_term2

    def convolution_prior_loss_factor_based(self, conv_weight, factor):
        if self._mask3.device != conv_weight.device:
            self._mask3 = self._mask3.to(conv_weight.device)
            self._mask5 = self._mask5.to(conv_weight.device)

        conv_weight = conv_weight**2
        h, w = conv_weight.shape[-2:]
        assert h % 2 == 1
        assert w % 2 == 1
        hmid = h // 2
        wmid = w // 2
        value = factor * conv_weight[:, :, hmid:hmid + 1, wmid:wmid + 1]
        loss = conv_weight - value
        # remove the unwanted loss of mid element
        if h == 5:
            loss = loss * self._mask5
        elif h == 3:
            loss = loss * self._mask3
        else:
            raise ValueError(f"Invalid (h,w): should be either 3 or 5, but is {h,w}")

        loss = torch.clip(loss, min=0)
        return torch.mean(loss)


def weight_less_than1_loss(conv_weight):
    conv_weight = conv_weight**2
    loss = conv_weight - 1
    loss = torch.clip(loss, min=0)
    return torch.mean(loss)


def convolution_prior_loss_absolute(conv_weight, min_loss):
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


if __name__ == '__main__':
    loss = ConvolutionPriorLoss(ConvolutionPriorLossType.MultiStepFactorBased, rf_factor=0.5)
    print(loss.get_factor_from_name('top_down_layers.0.deterministic_block.0.pre_conv.weight'))
    print(loss.get_factor_from_name('top_down_layers.1.deterministic_block.0.pre_conv.weight'))
    print(loss.get_factor_from_name('top_down_layers.2.deterministic_block.0.pre_conv.weight'))
    print(loss.get_factor_from_name('top_down_layers.3.deterministic_block.0.pre_conv.weight'))
    print(loss.get_factor_from_name('likelihood.3.deterministic_block.0.pre_conv.weight'))
    inp = torch.Tensor([[0.1, 0.8, 0.9], [0.6, 0.9, 0.95], [0.98, 0.9, 0.2]])[None, None]
    print(loss.get(inp))
