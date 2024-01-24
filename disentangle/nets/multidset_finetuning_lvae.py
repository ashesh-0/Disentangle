import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class MultiDsetFineTuningLvae(nn.Module):
    """
    If model_index=-1 is fed, then it predicts on all.  
    """

    def __init__(self, *models):
        self._models = nn.ModuleList(models)

    def forward(self, x, model_index):
        assert isinstance(model_index, int)
        output = {}
        if model_index == -1:
            # all
            for idx, model in enumerate(self._models):
                output[idx] = model(x)
        return {model_index: self._models[model_index](x)}


if __name__ == '__main__':
    from disentangle.configs.multidset_finetuning_config import get_config
    from disentangle.nets.model_utils import create_model
    from disentangle.training import create_dataset

    # from disentangle.data_loader.two_dset_dloader import TwoDsetDloader
    config = get_config()
    train_dset, val_dset = create_dataset()
    model1 = create_model(config1, data_mean, data_std)
    model2 = create_model(config2, data_mean, data_std)
    model = MultiDsetFineTuningLvae(model1, model2)

    batch_size = config.training.batch_size
    train_dloader = DataLoader(train_dset,
                               pin_memory=False,
                               num_workers=config.training.num_workers,
                               shuffle=True,
                               batch_size=batch_size)
    val_dloader = DataLoader(val_dset,
                             pin_memory=False,
                             num_workers=config.training.num_workers,
                             shuffle=False,
                             batch_size=batch_size)

    optimizer = optim.Adamax(model.parameters(), lr=config.training.lr, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min',
                                                     patience=config.training.lr_scheduler_patience,
                                                     factor=0.5,
                                                     min_lr=1e-12,
                                                     verbose=True)
