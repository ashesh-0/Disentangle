from disentangle.nets.coatnet_encoder import TransformerEncoder
from disentangle.nets.lvae import LadderVAE


class LadderVAEWithTransformer(LadderVAE):
    def __init__(self, data_mean, data_std, config):
        super().__init__(data_mean, data_std, config)

        # remove bottom-up layers
        del self.first_bottom_up 
        del self.lowres_first_bottom_ups 
        del self.bottom_up_layers

        encoder_params = config.model.transformer_encoder_params
        sz = config.data.image_size
        # add transformer encoder
        
        self.encoder = TransformerEncoder((sz, sz), config.data.get('color_ch', 1), encoder_params.num_blocks, encoder_params.channels, encoder_params.block_types,
                                          final_channel_size=config.model.decoder.n_filters)
    
    def bottomup_pass(self, x_pad):
        return self.encoder(x_pad)
    
        
if __name__ == '__main__':
    import numpy as np
    import torch

    import ml_collections
    # from disentangle.configs.microscopy_multi_channel_lvae_config import get_config
    from disentangle.configs.htt24_denoisplit import get_config
    config = get_config()

    config.model.mode_pred=True
    config.model.transformer_encoder_params = ml_collections.ConfigDict()
    config.model.transformer_encoder_params.num_blocks = [2, 2, 3, 5, 2]            # L
    config.model.transformer_encoder_params.channels = [64, 96, 192, 384, 768]  
    config.model.z_dims = [128, 128, 128, 128, 128]
    config.model.transformer_encoder_params.block_types=['C', 'C', 'T', 'T']

    
    data_mean = torch.Tensor([0]).reshape(1, 1,1, 1)
    # copy twice along 2nd dimensiion
    data_std = torch.Tensor([1]).reshape(1, 1,1, 1)
    model = LadderVAEWithTransformer({
        'input': data_mean,
        'target': data_mean.repeat(1, 2, 1,1)
    }, {
        'input': data_std,
        'target': data_std.repeat(1, 2, 1,1)
    }, config)
    mc = 1 if config.data.multiscale_lowres_count is None else config.data.multiscale_lowres_count
    # 3D example
    inp = torch.rand((2, mc, config.data.image_size, config.data.image_size))
    out, td_data = model(inp)
    batch = (
        torch.rand((16, mc, config.data.image_size, config.data.image_size)),
        torch.rand((16, 2, config.data.image_size, config.data.image_size)),
    )
    model.mode_pred=False
    model.training_step(batch, 0)
    model.validation_step(batch, 0)