import torch

from deepinv.physics.forward import LinearPhysics


class Superposition(LinearPhysics):
    def __init__(self):
        super().__init__(A = lambda x, **kwargs: torch.mean(x, dim=1, keepdim=True),
                         A_adjoint = lambda x, *kwargs: x.repeat(1,2,1,1))