from distutils.command.config import LANG_EXT
from disentangle.nets.lvae import LadderVAE


class LadderVAESemiSupervised(LadderVAE):
    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=..., target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at, target_ch)