import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from disentangle.loss.discriminator_loss import (DiscriminatorLossWithExistingData, RealData,
                                                 update_gradients_with_generator_loss)
from disentangle.loss.ssl_finetuning import get_stats_loss_func
from disentangle.nets.discriminator import Discriminator
from finetunesplit.loss import SSL_loss


def finetune_with_D_two_forward_passes(model, finetune_dset, finetune_val_dset, transform_obj, 
                                       external_real_data=None,
                                       external_real_data_probability=0.0, 
                                max_step_count=10000, 
                                batch_size=16,                    
                                skip_pixels=0,
                                scalar_params_dict=None,
                                validation_step_freq=1000,
                                k_Dsteps_perG=1,
                                optimization_params_dict=None, stats_enforcing_loss_fn=None, 
                                D_mode='wgan',
                                D_realimg_key='gt',
                                D_fakeimg_key='pred_FP1',
                                D_gp_lambda=0.1,
                                D_loss_scalar=1.0,
                                num_workers=4,
                                k_augmentations=1,sample_mixing_ratio=False, tmp_dir='/group/jug/ashesh/tmp'):
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
    factor2 = scalar_params_dict.get('factor2', None)
    offset2 = scalar_params_dict.get('offset2', None)
    mixing_ratio = scalar_params_dict.get('mixing_ratio', None)
                                        
    # define a learnable scalar and an offset 
    assert factor1 is not None
    assert offset1 is not None
    assert factor2 is not None
    assert offset2 is not None
    assert optimization_params_dict is not None, "Please provide optimization parameters"

    # define the discriminator
    discriminator = Discriminator(channels=2, first_out_channel=128)
    _ = discriminator.cuda()
    AdvLoss = DiscriminatorLossWithExistingData(discriminator, external_real_data, use_external_data_probability=external_real_data_probability, 
                                                      gradient_penalty_lambda=D_gp_lambda, loss_mode=D_mode, 
                                                      realimg_key=D_realimg_key, fakeimg_key=D_fakeimg_key,
                                                      loss_scalar=D_loss_scalar)
    real_data_gen = RealData(external_real_data)
    # define an optimizer
    # opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_gen = torch.optim.Adam(optimization_params_dict['parameters'], lr=optimization_params_dict['lr'], weight_decay=0)
    opt_dis = torch.optim.Adam(discriminator.parameters(), lr=optimization_params_dict['lr'], weight_decay=0)

    # SSL losses. 
    loss_arr = []
    loss_inp_arr = []
    loss_pred_arr = []
    loss_inp2_arr = []
    stats_loss_arr = []

    # discriminator based losses
    gen_fake_arr = []
    discrim_real_arr = []
    discrim_fake_arr = []
    grad_penalty_arr = []

    # factors and offsets
    factor1_arr = []
    offset1_arr = []
    factor2_arr = []
    offset2_arr = []
    mixing_ratio_arr= []
    
    best_val_loss = 1e6
    best_factors = best_offsets = None

    cnt = 0
    step_idx = 0
    while True:
        dloader = DataLoader(finetune_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        bar = tqdm(total =len(dloader), desc='Finetuning')
        for i, (inp, tar) in enumerate(dloader):
            inp = inp.cuda()
            # reset the gradients
            opt_gen.zero_grad()
            opt_dis.zero_grad()

            keys = ['loss_inp', 'loss_pred', 'loss_inp2', 'stats_loss']
            agg_loss_dict = {key: 0 for key in keys}
            for _ in range(k_augmentations):
                # apply the augmentations
                loss_dict = SSL_loss(pred_func, inp, transform_obj, mixing_ratio=mixing_ratio, factor1=factor1, offset1=offset1, factor2=factor2, 
                                    offset2=offset2, skip_pixels=skip_pixels,
                                    stats_enforcing_loss_fn=stats_enforcing_loss_fn, 
                                    sample_mixing_ratio=sample_mixing_ratio, return_predictions=True)
                for key in keys:
                    agg_loss_dict[key] =agg_loss_dict[key] + loss_dict[key]/ k_augmentations
                
            
            loss = agg_loss_dict['loss_inp'] + agg_loss_dict['loss_pred'] + agg_loss_dict['loss_inp2'] + agg_loss_dict['stats_loss']

            adv_data_dict = {'gt':None, # one can use tar here, but it is not needed
                             'pred_FP1':loss_dict['pred_FP1'], 
                             'pred_FP2':loss_dict['pred_FP2']
                             }
            # print('inspecting grad', adv_data_dict['pred_FP1'].mean(), adv_data_dict['pred_FP2'].mean())      
            
            
            # discriminator based loss 
            for p in discriminator.parameters():
                p.requires_grad = False
            
            if step_idx % k_Dsteps_perG == 0:
                loss.backward(retain_graph=True)
                gen_loss_dict= AdvLoss.G_loss(adv_data_dict)
                # gen_loss_dict = update_gradients_with_generator_loss(discriminator, pred[:,:2], mode=D_mode)
                opt_gen.step()


            # Now, we update the discriminator
            for p in discriminator.parameters():
                p.requires_grad = True
            
            
            # real_data = real_data_gen.get(len(inp)).to(inp.device)
            # disc_loss_dict = update_gradients_with_discriminator_loss(discriminator, real_data, pred[:,:2].detach(), lambda_term=grain_penalty_lambda, enable_gradient_penalty=enable_gradient_penalty, mode=D_mode)
            # print('inspecting grad', adv_data_dict['pred_FP1'].mean(), adv_data_dict['pred_FP2'].mean())      
            disc_loss_dict = AdvLoss.D_loss(adv_data_dict)
            opt_dis.step()
            
            
            
            
            
            
            loss_arr.append(loss.item())
            loss_inp_arr.append(agg_loss_dict['loss_inp'].item())
            loss_pred_arr.append(agg_loss_dict['loss_pred'].item())
            loss_inp2_arr.append(agg_loss_dict['loss_inp2'].item() if torch.is_tensor(agg_loss_dict['loss_inp2']) else agg_loss_dict['loss_inp2'])
            stats_loss_arr.append(agg_loss_dict['stats_loss'].item() if torch.is_tensor(agg_loss_dict['stats_loss']) else agg_loss_dict['stats_loss'])
            factor1_arr.append(factor1.item())
            offset1_arr.append(offset1.item())
            factor2_arr.append(factor2.item())
            offset2_arr.append(offset2.item())
            gen_fake_arr.append(gen_loss_dict['g_loss'])
            discrim_real_arr.append(disc_loss_dict['d_pred_real'])
            discrim_fake_arr.append(disc_loss_dict['d_pred_fake'])
            grad_penalty_arr.append(disc_loss_dict['d_loss_gradient_penalty'])

            # updating the counters
            cnt += len(inp)
            step_idx += 1

            mixing_ratio_arr.append(mixing_ratio.item())
            d_real = np.mean(discrim_real_arr[-100:])
            d_fake = np.mean(discrim_fake_arr[-100:])
            bar.set_description(f'[{cnt}] Loss:{np.mean(loss_arr[-10:]):.2f} D_Real:{d_real:.2f} D_Fake:{d_fake:.2f}')
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
                        loss_dict = SSL_loss(pred_func, inp, transform_obj, mixing_ratio=mixing_ratio, factor1=factor1, offset1=offset1, factor2=factor2, 
                                            offset2=offset2, skip_pixels=skip_pixels,
                                            stats_enforcing_loss_fn=stats_enforcing_loss_fn)
                        val_loss += loss_dict['loss_inp'].item() #+ loss_dict['loss_pred'].item()
                    val_steps += len(inp)
                    if val_steps >= val_steps_max:
                        break
                val_loss /= j
                
                print(f'Validation Loss: {val_loss:.4f}')

                if val_loss < best_val_loss:
                    print('Saving best model')
                    best_factors = [factor1.item(), factor2.item()]
                    best_offsets = [offset1.item(), offset2.item()]
                    best_val_loss = val_loss
                    # save the model
                    torch.save(model.state_dict(), f'{tmp_path}/best_model.pth')
                    torch.save(discriminator.state_dict(), f'{tmp_path}/best_discriminator.pth')

            if cnt >= max_step_count:
                break
        if cnt >= max_step_count:
            break
    
        bar.close()
    model.eval()
    if best_factors is None:
        best_factors = [factor1.item(), factor2.item()]
        best_offsets = [offset1.item(), offset2.item()]
    
    # load the best model
    print(f'Loading best model with loss {best_val_loss:.2f}')
    model.load_state_dict(torch.load(f'{tmp_path}/best_model.pth'))
    discriminator.load_state_dict(torch.load(f'{tmp_path}/best_discriminator.pth'))

    return {'loss': loss_arr, 'best_loss': best_val_loss, 'best_factors': best_factors, 
            'best_offsets': best_offsets, 'factor1': factor1_arr, 'offset1': offset1_arr, 
            'factor2': factor2_arr, 'offset2': offset2_arr, 'mixing_ratio': mixing_ratio_arr,
            'loss_inp': loss_inp_arr, 'loss_pred': loss_pred_arr,
            'loss_inp2': loss_inp2_arr,
            'stats_loss': stats_loss_arr,
            'gen_fake': gen_fake_arr, 'discrim_real': discrim_real_arr,
            'discrim_fake': discrim_fake_arr, 'grad_penalty': grad_penalty_arr,
            'discriminator':discriminator}
