from disentangle.core.custom_enum import Enum


class xyzinstitute2DataSetType(Enum):
    JustCYAN = '0b001'
    JustMAGENTA = '0b010'
    MIXED = '0b100'


class xyzinstitute2DataSetChannels(Enum):
    NucRFP670 = 0
    NucMTORQ = 1
    ACTIN = 2
    TUBULIN = 3


class xyzinstitute2DataSetVersion(Enum):
    DD = 'DenoisedDeconvolved'
    RAW = 'Raw data'


class xyzinstitute2BleedthroughType(Enum):
    Clean = 0
    Bleedthrough = 1
    Mixed = 2
