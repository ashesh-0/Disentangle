import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from disentangle.loss.restricted_reconstruction_loss import RestrictedReconstruction


class MultiDsetFineTuningLvae(nn.Module):
    """
    If model_index=-1 is fed, then it predicts on all.  
    """

    def __init__(self, config, finetuning_input_mean, finetuning_input_std, *models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.finetuning_dset_idx = config.model.finetuning_dset_idx
        # dset_idx -> channel_idx.
        # Note that model_idx and dset_idx must be the same.
        self.relevant_channels_dict = config.model.relevant_channels_dict
        self.grad_setter = RestrictedReconstruction(1, config.loss.mixed_rec_weight)
        # dset_idx -> position_idx in the final prediction.
        self.relevant_channels_position_dict = config.model.relevant_channels_position_dict
        self.finetune_data_mean = {'input': torch.Tensor(finetuning_input_mean), 'target': None}
        self.finetune_data_std = {'input': torch.Tensor(finetuning_input_std), 'target': None}
        assert len(self.models) == 2

        self.interchannel_weights = None
        self.compute_finetune_normalization_stats()

        if config.model.get('enable_learnableinterchannel_weights', False):
            target_ch = 2
            # self.interchannel_weights = nn.Parameter(torch.ones((1, target_ch, 1, 1)), requires_grad=True)
            self.interchannel_weights = nn.Conv2d(target_ch, target_ch, 1, bias=True, groups=target_ch)
            self.interchannel_weights.weight.data.fill_(1.0 * 0.01)
            self.interchannel_weights.bias.data.fill_(0.0)

    def set_params_to_same_device_as(self, x):
        for model in self.models:
            model.set_params_to_same_device_as(x)

        for key in self.finetune_data_mean:
            if self.finetune_data_mean[key].device != x.device:
                self.finetune_data_mean[key] = self.finetune_data_mean[key].to(x.device)
                self.finetune_data_std[key] = self.finetune_data_std[key].to(x.device)

    def compute_finetune_normalization_stats(self):
        mean = [None, None]
        std = [None, None]
        mean1 = self.models[0].data_mean['target'][:, self.relevant_channels_dict[0]]
        std1 = self.models[0].data_std['target'][:, self.relevant_channels_dict[0]]
        mean2 = self.models[1].data_mean['target'][:, self.relevant_channels_dict[1]]
        std2 = self.models[1].data_std['target'][:, self.relevant_channels_dict[1]]

        mean[self.relevant_channels_position_dict[0]] = mean1[:, None]
        std[self.relevant_channels_position_dict[0]] = std1[:, None]
        mean[self.relevant_channels_position_dict[1]] = mean2[:, None]
        std[self.relevant_channels_position_dict[1]] = std2[:, None]
        mean = torch.cat(mean, dim=1)
        std = torch.cat(std, dim=1)
        self.finetune_data_mean['target'] = mean
        self.finetune_data_std['target'] = std

    def forward(self, x, model_index):
        assert isinstance(model_index, int)
        output = {}
        if model_index == -1:
            # all
            for idx, model in enumerate(self.models):
                output[idx] = model(x)
            return output
        else:
            return {model_index: self.models[model_index](x)}

    def normalize_target(self, target, model_index):
        assert isinstance(model_index, int)
        return self.models[model_index].normalize_target(target)


def train_one_batch(batch, model, optimizer, current_epoch):
    x, target, dset_idx, loss_idx = batch
    x = x.cuda()
    model.set_params_to_same_device_as(x)
    target = target.cuda()
    dset_idx = dset_idx.cuda()
    loss_idx = loss_idx.cuda()

    optimizer.zero_grad()
    loss = 0.0
    other_targets = []
    other_predictions = []
    for didx in torch.unique(dset_idx):
        didx = didx.item()
        mask = dset_idx == didx
        if didx != model.finetuning_dset_idx:
            target_normalized = model.normalize_target(target[mask], didx)
            out_dict = model(x[mask], model_index=didx)
            out, _ = out_dict[didx]
            other_targets.append(target_normalized)
            other_predictions.append(out)
            recons_loss_dict = model.models[didx].get_reconstruction_loss(out,
                                                                          target_normalized,
                                                                          x[mask],
                                                                          return_predicted_img=False)
            loss += recons_loss_dict['loss']
        else:
            continue
            # this will be handled at the end.

    other_targets = torch.cat(other_targets, dim=0)
    other_predictions = torch.cat(other_predictions, dim=0)

    loss.backward(retain_graph=True)

    mask = dset_idx == model.finetuning_dset_idx
    x_finetune = x[mask]
    out_dict = model(x_finetune, model_index=-1)
    ftune_out = [None, None]

    for dset_idx in out_dict:
        out, _ = out_dict[dset_idx]
        ch_idx = model.relevant_channels_dict[dset_idx]
        # get the relevant prediction.
        out = out[:, ch_idx:ch_idx + 1, :, :]
        ftune_out[model.relevant_channels_position_dict[dset_idx]] = out

    out = torch.cat(ftune_out, dim=1)
    # accounting for the scaling factor.
    if model.interchannel_weights is not None:
        out = model.interchannel_weights(out)

    pred_x, _ = model.models[0].get_mixed_prediction(out,
                                                     None,
                                                     model.finetune_data_mean,
                                                     model.finetune_data_std,
                                                     channel_weights=None)
    model.grad_setter.update_gradients(model.named_parameters(), x_finetune, other_targets, other_predictions, pred_x,
                                       current_epoch)
    optimizer.step()
    return loss.item()


if __name__ == '__main__':
    from copy import deepcopy

    from tqdm import tqdm

    from disentangle.configs.multidset_finetuning_config import get_config
    from disentangle.nets.model_utils import create_model, get_mean_std_dict_for_model
    from disentangle.training import create_dataset

    # from disentangle.data_loader.two_dset_dloader import TwoDsetDloader
    config = get_config()
    datadir = '/group/jug/ashesh/data/microscopy/'
    train_dset, val_dset = create_dataset(config, datadir)
    mean_dict, std_dict = get_mean_std_dict_for_model(config, train_dset)

    config1 = deepcopy(config)
    for key in config1.model.model1:
        config1.model[key] = config.model.model1[key]

    config2 = deepcopy(config)
    for key in config2.model.model2:
        config2.model[key] = config.model.model2[key]

    model1 = create_model(config1, mean_dict['subdset_0'], std_dict['subdset_0'])
    model2 = create_model(config2, mean_dict['subdset_1'], std_dict['subdset_1'])
    model = MultiDsetFineTuningLvae(config, mean_dict['subdset_2']['input'], std_dict['subdset_2']['input'], model1,
                                    model2)
    model = model.cuda()

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
    # train now
    for epoch in tqdm(range(config.training.max_epochs)):
        model.train()
        for batch_idx, batch in enumerate(train_dloader):
            train_one_batch(batch, model, optimizer, epoch)

        # validate now
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_dloader):
                data = data.cuda()
                output = model(data, model_index=-1)
                loss = 0
                for idx, out in output.items():
                    loss += out['loss']
                val_loss += loss.item()
        val_loss /= len(val_dloader)
        scheduler.step(val_loss)
        print('Epoch: {} \tValidation Loss: {:.6f}'.format(epoch, val_loss))
