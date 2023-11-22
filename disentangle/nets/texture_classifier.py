import torch
import torch.nn as nn

from disentangle.nets.lvae_layers import (BottomUpDeterministicResBlock, BottomUpLayer, TopDownDeterministicResBlock,
                                          TopDownLayer)


class SingleBottomUpLayer(BottomUpLayer):

    def forward(self, x):
        x, _ = super().forward(x)
        return x


class TextureEncoder(nn.Module):

    def __init__(self, with_sigmoid=True):
        super().__init__()

        self.nonlin = nn.LeakyReLU
        self.num_blocks_per_layer = 2
        self.num_hierarchy_levels = 2
        self.color_ch = 1
        self.encoder_n_filters = 32
        self.encoder_res_block_kernel = 3
        self.encoder_res_block_skip_padding = True
        self.encoder_dropout = 0.0
        self.res_block_type = 'bacdbacd'
        self.batchnorm = True
        self.gated = False
        modules = [
            nn.Conv2d(self.color_ch,
                      self.encoder_n_filters,
                      self.encoder_res_block_kernel,
                      padding=0 if self.encoder_res_block_skip_padding else self.encoder_res_block_kernel // 2,
                      stride=1),
            self.nonlin()
        ]
        for _ in range(self.num_hierarchy_levels):
            modules.append(
                SingleBottomUpLayer(n_res_blocks=self.num_blocks_per_layer,
                                    n_filters=self.encoder_n_filters,
                                    downsampling_steps=1,
                                    nonlin=self.nonlin,
                                    batchnorm=self.batchnorm,
                                    dropout=self.encoder_dropout,
                                    res_block_type=self.res_block_type,
                                    res_block_kernel=self.encoder_res_block_kernel,
                                    res_block_skip_padding=self.encoder_res_block_skip_padding,
                                    gated=self.gated))
        self.encoder = nn.Sequential(*modules)
        self.with_sigmoid = with_sigmoid
        if self.with_sigmoid:
            self.classifier = nn.Sequential(nn.Conv2d(self.encoder_n_filters, 1, 1), nn.Sigmoid())
        else:
            self.classifier = nn.Sequential(nn.Conv2d(self.encoder_n_filters, 1, 1))

    def forward(self, x):
        latent = self.encoder(x)
        return self.classifier(latent)


if __name__ == '__main__':

    import torch.optim as optim

    from disentangle.configs.biosr_sparsely_supervised_config import get_config
    from disentangle.data_loader.multi_channel_determ_tiff_dloader import (DataSplitType, GridAlignement,
                                                                           MultiChDeterministicTiffDloader)

    lr = 1e-3
    batch_size = 32
    num_epochs = 10

    config = get_config()

    # model related stuff
    model = TextureClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min',
                                                     patience=config.training.lr_scheduler_patience,
                                                     factor=0.5,
                                                     min_lr=1e-12,
                                                     verbose=True)

    # data related stuff
    dset = MultiChDeterministicTiffDloader(
        config.data,
        #    '/group/jug/ashesh/data/microscopy/OptiMEM100x014.tif',
        '/mnt/ashesh/BioSR/',
        DataSplitType.Train,
        val_fraction=config.training.val_fraction,
        test_fraction=config.training.test_fraction,
        normalized_input=config.data.normalized_input,
        enable_rotation_aug=config.data.normalized_input,
        enable_random_cropping=config.data.deterministic_grid is False,
        use_one_mu_std=config.data.use_one_mu_std,
        allow_generation=False,
        max_val=None,
        grid_alignment=GridAlignement.LeftTop,
        overlapping_padding_kwargs=None)

    mean, std = dset.compute_mean_std()
    dset.set_mean_std(mean, std)
    trainloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=2)

    #### Training #####

    loss_arr = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            _, tar = data
            inputs = torch.cat([tar[:, :1], tar[:, 1:]], dim=0)
            inputs = (inputs - mean.mean()) / std.mean()
            labels = torch.ones(inputs.shape[0], dtype=torch.float32)
            labels[:len(labels) // 2] = 0
            labels = labels.reshape(-1, 1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            avg_outputs = torch.mean(outputs, dim=(1, 2, 3)).reshape(-1, 1)
            loss = criterion(avg_outputs, labels)
            loss.backward()
            optimizer.step()
            loss_arr.append(loss.item())
            scheduler.step(loss.item())
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    patch = '/mnt/ashesh/texture_classifier.pth'
    torch.save(model.state_dict(), patch)
