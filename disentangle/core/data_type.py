from disentangle.core.custom_enum import Enum


class DataType(Enum):
    MNIST = 0
    Places365 = 1
    NotMNIST = 2
    OptiMEM100_014 = 3
    CustomSinosoid = 4
    Prevedel_pqrsinstitute = 5
    AllenCellMito = 6
    SeparateTiffData = 7
    CustomSinosoidThreeCurve = 8
    SemiSupBloodVesselspqrsinstitute = 9
    xyzinstitute2 = 10
    xyzinstitute2VanillaSplitting = 11
    ExpansionMicroscopyMitoTub = 12
    ShroffMitoEr = 13
    HTIba1Ki67 = 14
    BSD68 = 15
    BioSR_MRC = 16
    TavernaSox2Golgi = 17
    Dao3Channel = 18
    ExpMicroscopyV2 = 19
    Dao3ChannelV2 = 20
    TavernaSox2GolgiV2 = 21
    TwoDset = 22
    PredictedTiffData = 23
    Derain100H = 30
    Dehaze4K = 31


if __name__ == '__main__':
    from disentangle.core.tiff_reader import load_tiff
    data = load_tiff('/group/jug/ashesh/data/Rain1000HNew/combined/data_1800.tif')
    print(data.shape)
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(6, 3), ncols=2)
    ax[0].imshow(data[:3].transpose(1, 2, 0))
    ax[1].imshow(data[3:].transpose(1, 2, 0))
