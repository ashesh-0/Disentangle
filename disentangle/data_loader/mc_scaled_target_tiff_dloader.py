from disentangle.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader


class MCScaledTargetDloader(MultiChDeterministicTiffDloader):
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        inp, target = super().__getitem__(index)

