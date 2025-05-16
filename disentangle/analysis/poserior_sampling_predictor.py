import numpy as np
import torch
import torch.nn as nn

from deepinv.transform.projective import Homography
from finetunesplit.asymmetric_transforms import (CorrelationPreservingTransforms, DeepinvTransform, HFlip, Identity,
                                                 Rotate, TransformAllChannels, TransformEnum, Translate, VFlip,
                                                 get_inverse_transforms)


def _get_homography(kwargs):
    return Homography(n_trans = 1, 
                        zoom_factor_min=1.0, 
                        theta_max=kwargs.get('theta_max', 0),
                        theta_z_max=kwargs.get('theta_z_max',0), 
                        skew_max=kwargs.get('skew_max',0), 
                        shift_max=kwargs.get('shift_max',0),
                        x_stretch_factor_min = 1,
                        y_stretch_factor_min = 1, 
                        device = kwargs['device'])


def get_one_transform(transform_enum, **kwargs):
    if transform_enum == TransformEnum.Identity:
        return Identity()
    elif transform_enum == TransformEnum.VFlip:
        return VFlip()
    elif transform_enum == TransformEnum.HFlip:
        return HFlip()
    elif transform_enum == TransformEnum.Rotate:
        return Rotate()
    elif transform_enum == TransformEnum.Translate:
        return Translate(**kwargs)
    elif transform_enum == TransformEnum.DeepInV:
        trans_homo = _get_homography(kwargs)
        return DeepinvTransform(trans_homo)
    else:
        raise ValueError(f"Unknown transform enum: {transform_enum}")

def get_transform_obj(ch1_transforms, ch2_transforms, correlation_preserving_transforms=False):    
    transform_types = {
                    0:[get_one_transform(x,**kwargs) for x, kwargs in ch1_transforms], 
                    1:[get_one_transform(x,**kwargs) for x, kwargs in ch2_transforms]
                    }

    # print 
    print('Transforms for Ch1:', ch1_transforms)
    print('Transforms for Ch2:', ch2_transforms)
    if correlation_preserving_transforms:
        print('Using correlation preserving transforms')
        transform_all = CorrelationPreservingTransforms(transform_types)
    else:
        print('Using asymmetric transforms')
        transform_all = TransformAllChannels(transform_types)
    return transform_all

class PosteriorSamplingPredictor(nn.Module):
    def __init__(self, model, transform, forward_operator_params, k_predictions=1, k_prediction_mode='entire'):
        """
        Args:
        k_prediction_mode: 'entire' - use the entire prediction scheme with first and the augmentation based predictions
        'only_transformed' - use only the transformed prediction: skips the stochasticity of the first prediction. Stochasticity comes only from transforms.
        'only_first' - use only the first prediction: stochasticity comes only from the first prediction. augmentation is not used.
        """
        super().__init__()
        self.model = model
        self.transform = transform
        
        self.mixing_t_min = forward_operator_params['mixing_t_min']
        self.mixing_t_max = forward_operator_params['mixing_t_max']
        self.mu = forward_operator_params['mu']
        self.sigma = forward_operator_params['sigma']

        self.k = k_predictions
        self.k_prediction_mode = k_prediction_mode
        print(f'[{self.__class__.__name__}] k_prediction_mode: {self.k_prediction_mode}')

        assert self.k_prediction_mode in ['entire', 'only_transformed', 'only_first']

    def forward(self, x):
        outputs = []
        pred1,_ = self.model(x)
        # assert pred1.shape[1] == 2, f"Expected pred1 to have 2 channels, but got {pred1.shape[1]}"
        pred1_mmse = 0
        for _ in range(self.k):
            pred1_mmse += pred1[:,:2]/self.k
            if self.k_prediction_mode == 'only_first':
                outputs.append(pred1[:,:2])
            elif self.k_prediction_mode in ['only_transformed', 'entire']:
                pred1_transformed, applied_transforms = self.transform(pred1[:,:2])
                # print('max difference pred<->pred_transformed', torch.abs(pred1_transformed - pred1[:,:2]).max())
                inv_transform, invertible = get_inverse_transforms(applied_transforms)
                assert invertible is True, "Transform is not invertible"
                mixing_t = np.random.uniform(self.mixing_t_min, self.mixing_t_max)
                new_inp = pred1_transformed[:,:1]*mixing_t + pred1_transformed[:,1:2]* (1-mixing_t)
                new_inp = new_inp * self.sigma + self.mu
                pred2,_ = self.model(new_inp)
                # apply inverse transform on pred2
                inv_transformed_pred, _  = self.transform(pred2[:,:2], params_dict = inv_transform, inverse=True)
                # print('max difference pred<->inverse of pred_transformed', torch.abs(inv_transformed_pred - pred1[:,:2]).max())
                outputs.append(inv_transformed_pred)
            if self.k_prediction_mode in ['entire', 'only_first']:
                pred1, _ = self.model(x)
                print('sampling  again')
        
        return outputs, pred1_mmse