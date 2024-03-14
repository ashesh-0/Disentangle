import numpy as np

from disentangle.core.data_split_type import DataSplitType
from disentangle.data_loader.xyzinstitute2_dloader import xyzinstitute2DataSetChannels, xyzinstitute2V1Dloader


class xyzinstitute2ThreeChannelDloader(xyzinstitute2V1Dloader):

    def __init__(self,
                 data_config,
                 fpath: str,
                 datasplit_type: DataSplitType = None,
                 val_fraction=None,
                 test_fraction=None,
                 normalized_input=None,
                 enable_rotation_aug: bool = False,
                 enable_random_cropping: bool = False,
                 use_one_mu_std=None,
                 allow_generation=False,
                 max_val=None) -> None:

        # which are the indices for bleedthrough nucleus, clean nucleus, tubulin
        self._bt_nuc_idx = data_config.channel_idx_list.index(xyzinstitute2DataSetChannels.NucMTORQ)
        self._cl_nuc_idx = data_config.channel_idx_list.index(xyzinstitute2DataSetChannels.NucRFP670)
        self._tubuln_idx = data_config.channel_idx_list.index(xyzinstitute2DataSetChannels.TUBULIN)

        # self._relv_channel_idx = [xyzinstitute2DataSetChannels.NucRFP670, xyzinstitute2DataSetChannels.NucMTORQ, xyzinstitute2DataSetChannels.TUBULIN]
        super().__init__(data_config, fpath, datasplit_type, val_fraction, test_fraction, normalized_input,
                         enable_rotation_aug, enable_random_cropping, use_one_mu_std, allow_generation, max_val)

    def get_max_val(self):
        return self._dloader_clean.get_max_val()

    def process_data(self):
        """
        We are ignoring the actin channel.
        We know that MTORQ(uise) has sigficant bleedthrough from TUBULIN channels. So, when MTORQ has no content, then 
        we sum it with TUBULIN so that tubulin has whole of its content. 
        When MTORQ has content, then we sum RFP670 with tubulin. This makes sure that tubulin channel has the same data distribution. 
        During validation/testing, we always feed sum of these three channels as the input.
        """
        pass


if __name__ == '__main__':
    from disentangle.configs.xyzinstitute2_config import get_config
    config = get_config()
    fpath = '/group/ubuntu/ubuntu/data/xyzinstitute2/'
    dloader = xyzinstitute2ThreeChannelDloader(config.data,
                                               fpath,
                                               datasplit_type=DataSplitType.Train,
                                               val_fraction=0.1,
                                               test_fraction=0.1,
                                               normalized_input=True,
                                               use_one_mu_std=False,
                                               enable_random_cropping=True)
    mean_val, std_val = dloader.compute_mean_std()
    dloader.set_mean_std(mean_val, std_val)
    inp, tar, source = dloader[0]
    print('This is working')
