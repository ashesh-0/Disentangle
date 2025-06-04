import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from disentangle.loss.discriminator_loss import (DiscriminatorLossWithExistingData, RealData,
                                                 update_gradients_with_generator_loss)
from disentangle.loss.ssl_finetuning import get_stats_loss_func
from disentangle.nets.discriminator import Discriminator
from disentangle.nets.vae_compressor import get_vae
from finetunesplit.loss import SSL_loss_n2v


def finetune_with_D_two_forward_passes_n2v(model, 
                                           finetune_dset, 
                                           finetune_val_dset, 
                                           transform_obj, 
                                            max_step_count=10000, 
                                            batch_size=16,                    
                                            scalar_params_dict=None,
                                            validation_step_freq=1000,
                                            optimization_params_dict=None, 
                                            num_workers=4,
                                            k_augmentations=1,
                                            sample_mixing_ratio=False, 
                                            tmp_dir='/group/jug/ashesh/tmp'
                                ):
    """
    external_real_data: For a discriminator based setup, we need to have a notion of how the real data should look like. So, this would either be the predictions on the training data, or would be real channel data. Note the it must be of the same shape as the predictions. Alos, it must be normalized. Because otherwise, one can simply differentiate by the scaling.
    """
    tmp_path = f'{tmp_dir}/finetune_with_discriminator_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(tmp_path, exist_ok=True)
    # enable dropout.
    # model.train()
    print(f'Finetuning with discriminator, {k_augmentations} augmentations, batch size {batch_size}, max step count {max_step_count}, sample mixing ratio {sample_mixing_ratio}')
    def pred_func(inp):
        return model(inp)[0][:,:2]

    factor1 = scalar_params_dict.get('factor1', None)
    offset1 = scalar_params_dict.get('offset1', None)
    mixing_ratio = scalar_params_dict.get('mixing_ratio', None)
                                        
    # define a learnable scalar and an offset 
    assert factor1 is not None
    assert offset1 is not None
    assert optimization_params_dict is not None, "Please provide optimization parameters"
    gen_params  = list(optimization_params_dict['parameters'])
    opt_gen = torch.optim.Adam(gen_params, lr=optimization_params_dict['lr'], weight_decay=0)

    # SSL losses. 
    loss_arr = []
    best_val_loss = 1e6
    best_factors = best_offsets = best_step = None

    cnt = 0
    while True:
        dloader = DataLoader(finetune_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        bar = tqdm(total =len(dloader), desc='Finetuning')
        for i, (inp, tar) in enumerate(dloader):
            inp = inp.cuda()
            opt_gen.zero_grad()
            keys = ['loss_inp']
            agg_loss_dict = {}
            loss_dict = SSL_loss_n2v(pred_func, inp, transform_obj, mixing_ratio=mixing_ratio, factor1=factor1, offset1=offset1)
            for key in keys:
                agg_loss_dict[key] = loss_dict[key]
            
            loss = agg_loss_dict['loss_inp']
            loss.backward()
            opt_gen.step()
            loss_arr.append(loss.item())
            cnt += len(inp)
            update_str = f'[{cnt}] Loss:{np.mean(loss_arr[-10:]):.2f}'
            bar.set_description(update_str)
            bar.update(1)
            
            if validation_step_freq is not None and cnt //validation_step_freq > (cnt - len(inp))//validation_step_freq:
                # validate the model
                print('Validating the model', end=' ')
                val_loss = 0
                val_steps = 0
                val_steps_max = 1000
                val_dloader = DataLoader(finetune_val_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                for j, (inp, tar) in tqdm(enumerate(val_dloader)):
                    inp = inp.cuda()
                    with torch.no_grad():
                        loss_dict = SSL_loss_n2v(pred_func, inp, transform_obj, mixing_ratio=mixing_ratio, factor1=factor1, offset1=offset1)
                        val_loss += loss_dict['loss_inp'].item() #+ loss_dict['loss_pred'].item()
                    val_steps += len(inp)
                    if val_steps >= val_steps_max:
                        break
                val_loss /= j
                
                print(f'Validation Loss: {val_loss:.4f}')

                if val_loss < best_val_loss:
                    print('Saving best model')
                    best_factors = [factor1.item()]
                    best_offsets = [offset1.item()]
                    best_val_loss = val_loss
                    best_step = len(loss_arr)
                    # save the model
                    torch.save(model.state_dict(), f'{tmp_path}/best_model.pth')

            if cnt >= max_step_count:
                break
        if cnt >= max_step_count:
            break
    
        bar.close()
    model.eval()
    if best_factors is None:
        best_factors = [factor1.item()]
        best_offsets = [offset1.item()]
    
    # load the best model
    print(f'Loading best model with loss {best_val_loss:.2f}')
    model.load_state_dict(torch.load(f'{tmp_path}/best_model.pth'))

    return {'loss': loss_arr, 
            'best_loss': best_val_loss, 
            'best_factors': best_factors, 
            'best_offsets': best_offsets, 
            'best_step': best_step,
            }
