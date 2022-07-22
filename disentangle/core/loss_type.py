from disentangle.core.custom_enum import Enum


class LossType:
    Elbo = 0
    ElboWithCritic = 1
    ElboMixedReconstruction = 2
