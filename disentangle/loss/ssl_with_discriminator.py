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
from finetunesplit.loss import SSL_loss


def total_variation_loss(img, weight):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

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
                                dense_discriminator=False,
                                D_realimg_key='gt',
                                D_fakeimg_key='pred_FP1',
                                D_gp_lambda=0.1,
                                D_loss_scalar=1.0,
                                D_only_one_channel_idx =None,
                                D_train_G_on_both_real_and_fake=False,
                                use_embedding_network=False,
                                pretrained_embedding_network_fpath=None,
                                embedding_zdim = 8,
                                tv_weight =0.0,
                                enable_supervised_loss=False,
                                start_adv_loss_step=10000,
                                num_workers=4,
                                k_augmentations=1,sample_mixing_ratio=False, tmp_dir='/group/jug/ashesh/tmp'):
    """
    external_real_data: For a discriminator based setup, we need to have a notion of how the real data should look like. So, this would either be the predictions on the training data, or would be real channel data. Note the it must be of the same shape as the predictions. Alos, it must be normalized. Because otherwise, one can simply differentiate by the scaling.
    """
    if enable_supervised_loss:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print('')
        print('WARNING: !! Enabling supervised loss')
        print('')
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
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

    embedding_network = None
    update_embedding_network = use_embedding_network and pretrained_embedding_network_fpath is None
    if use_embedding_network:
        # print('Using embedding network for discriminator', embedding_zdim)
        embedding_network,num_input_channels = get_vae(input_channels=1 if D_realimg_key == 'inp' else 2, z_dim=embedding_zdim)
        if pretrained_embedding_network_fpath is not None:
            print(f'Loading pretrained embedding network from {pretrained_embedding_network_fpath}', embedding_zdim)
            embedding_network.load_state_dict(torch.load(pretrained_embedding_network_fpath))

        embedding_network.set_params_to_same_device_as(torch.ones(1).cuda())
    # define the discriminator
    # discriminator = Discriminator(channels=2, first_out_channel=128)
    # _ = discriminator.cuda()
    if embedding_network is None:
        if D_only_one_channel_idx is not None:
            num_input_channels = 1
        elif D_realimg_key =='inp':
            num_input_channels = 1
        else:
            num_input_channels = 2
    if dense_discriminator:
        # update 2x2 with whatever is the spatial size of the input.
        if embedding_network is None:
            num_input_channels *= 28*28
        else:
            num_input_channels *= 2*2
    AdvLoss = DiscriminatorLossWithExistingData(external_real_data, 
                                                num_channels=num_input_channels, 
                                                dense_discriminator=dense_discriminator,
                                                use_external_data_probability=external_real_data_probability, 
                                                      gradient_penalty_lambda=D_gp_lambda, loss_mode=D_mode, 
                                                      realimg_key=D_realimg_key, fakeimg_key=D_fakeimg_key,
                                                      loss_scalar=D_loss_scalar,
                                                      only_one_channel_idx=D_only_one_channel_idx,
                                                      train_G_on_both_real_and_fake=D_train_G_on_both_real_and_fake,
                                                      embedding_network=embedding_network)
    _ = AdvLoss.cuda()

    # real_data_gen = RealData(external_real_data)
    # define an optimizer
    # opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen_params  = list(optimization_params_dict['parameters'])
    if update_embedding_network:
        gen_params += list(embedding_network.parameters())
    
    opt_gen = torch.optim.Adam(gen_params, lr=optimization_params_dict['lr'], weight_decay=0)
    opt_dis = torch.optim.Adam(AdvLoss.discriminator_network.parameters(), lr=optimization_params_dict['lr'], weight_decay=0)

    # SSL losses. 
    loss_arr = []
    loss_inp_arr = []
    loss_pred_arr = []
    loss_inp2_arr = []
    loss_tv_arr = []
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
    loss_sup_arr = []
    
    best_val_loss = 1e6
    best_factors = best_offsets = best_step = None

    cnt = 0
    step_idx = 0
    switch_made = False
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
                loss_sup = 0
                if enable_supervised_loss:
                    loss_sup = nn.MSELoss()(loss_dict['pred_FP1'], tar.cuda())
                
                for key in keys:
                    agg_loss_dict[key] =agg_loss_dict[key] + loss_dict[key]/ k_augmentations
                
            
            loss = agg_loss_dict['loss_inp'] + agg_loss_dict['loss_pred'] + agg_loss_dict['loss_inp2'] + agg_loss_dict['stats_loss'] + loss_sup
            loss_tv = 0
            if tv_weight > 0:
                loss_tv = total_variation_loss(loss_dict['pred_FP1'], tv_weight)
            
            loss += loss_tv

            adv_data_dict = {'gt':None, # one can use tar here, but it is not needed
                             'pred_FP1':loss_dict['pred_FP1'], 
                             'pred_FP2':loss_dict['pred_FP2'],
                             'pred_FP1_aug':loss_dict['pred_FP1_aug'],
                             'inv_inp2':loss_dict['inv_inp2'],
                            #   'inp':loss_dict['inp'],
                              'predInp1':loss_dict['predInp1'],
                             }
            # print('inspecting grad', adv_data_dict['pred_FP1'].mean(), adv_data_dict['pred_FP2'].mean())      
            
            
            # discriminator based loss 
            for p in AdvLoss.discriminator_network.parameters():
                p.requires_grad = False
            
            loss_embedding_network = 0
            # gen_loss_dict = update_gradients_with_generator_loss(discriminator, pred[:,:2], mode=D_mode)
            if update_embedding_network:
                # We need to train on real image as well. 
                pred_detached = loss_dict[D_realimg_key].detach() if D_realimg_key != 'inp' else AdvLoss.get_external_data(len(inp)).to(inp.device)
                # print(pred_detached.shape, 'pred_detached')

                loss_embedding_network = embedding_network.training_step((pred_detached, pred_detached), i)['loss']
                loss += loss_embedding_network
            
            loss.backward(retain_graph=True)
            enable_adv = cnt >= start_adv_loss_step

            update_with_g_loss = enable_adv and ((k_Dsteps_perG > 1 and step_idx % k_Dsteps_perG == 0) or (k_Dsteps_perG <= 1))
            gen_loss_dict= AdvLoss.G_loss(adv_data_dict, return_loss_without_update=not update_with_g_loss)

            opt_gen.step()


            # Now, we update the discriminator
            for p in AdvLoss.parameters():
                p.requires_grad = True
            
            
            # real_data = real_data_gen.get(len(inp)).to(inp.device)
            # disc_loss_dict = update_gradients_with_discriminator_loss(discriminator, real_data, pred[:,:2].detach(), lambda_term=grain_penalty_lambda, enable_gradient_penalty=enable_gradient_penalty, mode=D_mode)
            # print('inspecting grad', adv_data_dict['pred_FP1'].mean(), adv_data_dict['pred_FP2'].mean())      
            update_with_D_loss =enable_adv and (k_Dsteps_perG > 1 or (step_idx % (int(1/k_Dsteps_perG)) == 0))
            disc_loss_dict = AdvLoss.D_loss(adv_data_dict, return_loss_without_update=not update_with_D_loss)
            if update_with_D_loss:
                opt_dis.step()
                # print('not skipping discriminator')
            # else:
                # print('skipping discriminator')            
            # disc_loss_dict = AdvLoss.D_loss(adv_data_dict, return_loss_without_update=cnt > 40_000)
            # if cnt <= 40_000:
            #     opt_dis.step()
            # else:
            #     if not switch_made:
            #         print('disabling the discriminator')
            #         switch_made = True    
            
            
            
            
            
            loss_arr.append(loss.item())
            loss_inp_arr.append(agg_loss_dict['loss_inp'].item())
            loss_pred_arr.append(agg_loss_dict['loss_pred'].item())
            loss_inp2_arr.append(agg_loss_dict['loss_inp2'].item() if torch.is_tensor(agg_loss_dict['loss_inp2']) else agg_loss_dict['loss_inp2'])
            loss_tv_arr.append(loss_tv.item() if torch.is_tensor(loss_tv) else loss_tv)
            loss_sup_arr.append(loss_sup.item() if torch.is_tensor(loss_sup) else None)
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
            update_str = f'[{cnt}] Loss:{np.mean(loss_arr[-10:]):.2f} D_Real:{d_real:.2f} D_Fake:{d_fake:.2f}'
            if tv_weight > 0:
                update_str += f' TV:{np.mean(loss_tv_arr[-10:]):.2f}'
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
                    best_step = len(loss_arr)
                    # save the model
                    torch.save(model.state_dict(), f'{tmp_path}/best_model.pth')
                    torch.save(AdvLoss.state_dict(), f'{tmp_path}/best_discriminator.pth')

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
    AdvLoss.load_state_dict(torch.load(f'{tmp_path}/best_discriminator.pth'))

    return {'loss': loss_arr, 'best_loss': best_val_loss, 'best_factors': best_factors, 
            'best_offsets': best_offsets, 'factor1': factor1_arr, 'offset1': offset1_arr, 
            'best_step': best_step,
            'factor2': factor2_arr, 'offset2': offset2_arr, 'mixing_ratio': mixing_ratio_arr,
            'loss_inp': loss_inp_arr, 'loss_pred': loss_pred_arr,
            'loss_inp2': loss_inp2_arr,
            'loss_tv': loss_tv_arr,
            'loss_sup': loss_sup_arr,
            'stats_loss': stats_loss_arr,
            'gen_fake': gen_fake_arr, 'discrim_real': discrim_real_arr,
            'discrim_fake': discrim_fake_arr, 'grad_penalty': grad_penalty_arr,
            'discriminator':AdvLoss}
