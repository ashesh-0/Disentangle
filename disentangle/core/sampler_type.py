from disentangle.core.custom_enum import Enum


class SamplerType(Enum):
    DefaultSampler = 0
    RandomSampler = 1
    SingleImgSampler = 2
