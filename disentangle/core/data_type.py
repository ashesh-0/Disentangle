from disentangle.core.custom_enum import Enum


class DataType(Enum):
    MNIST = 0
    Places365 = 1
    NotMNIST = 2
    OptiMEM100_014 = 3
    CustomSinosoid = 4
    Prevedel_EMBL = 5
    AllenCellMito = 6
    SeparateTiffData = 7
    CustomSinosoidThreeCurve = 8
    SemiSupBloodVesselsEMBL = 9
    Pavia2 = 10
    Pavia2VanillaSplitting = 11
    ExpMicroscopyV1 = 12
    ShroffMitoEr = 13
    HTIba1Ki67 = 14
    BSD68 = 15
    BioSR_MRC = 16
    TavernaSox2Golgi = 17
    Dao3Channel = 18
    ExpMicroscopyV2 = 19
    Dao3ChannelWithInput = 20
    TavernaSox2GolgiV2 = 21
    TwoDset = 22
    PredictedTiffData = 23
    Pavia3SeqData = 24
    # Here, we have 16 splitting tasks.
    NicolaData = 25
    SilvioLabCSHLData = 26
    ExpMicroscopyV3 = 27
    MultiCropDset = 28
    Elisa3DData = 29
    Care3D = 30
    SimilarityExperiment = 31
    MultiTiffSameSizeDset = 32
    HHMI25V2 = 33
    HHMI25V3 = 34