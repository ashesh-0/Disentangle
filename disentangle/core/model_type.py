from disentangle.core.custom_enum import Enum
from disentangle.nets.lvae_with_stitch import LadderVAEwithStitching


class ModelType(Enum):
    LadderVae = 3
    LadderVaeTwinDecoder = 4
    LadderVAECritic = 5
    # Separate vampprior: two optimizers
    LadderVaeSepVampprior = 6
    # one encoder for mixed input, two for separate inputs.
    LadderVaeSepEncoder = 7
    LadderVAEMultiTarget = 8
    LadderVaeSepEncoderSingleOptim = 9
    UNet = 10
    BraveNet = 11
    LadderVaeStitch = 12
