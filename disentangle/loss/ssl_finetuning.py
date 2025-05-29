
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from disentangle.analysis.stitch_prediction import stitch_predictions
from disentangle.core.psnr import RangeInvariantPsnr
from finetunesplit.loss import SSL_loss


def k_moment(data, k):
    # data: N x C x H x W
    if k ==0:
        return torch.Tensor([0.0], device=data.device)
    
    elif k == 1:
        return torch.mean(data, dim=(0, 2,3))
    
    dif = data - torch.mean(data, dim=(2,3))[...,None, None]
    moment = torch.mean(dif**k, dim=(2,3))
    neg_mask = (moment < 0).type(torch.long)
    moment = torch.pow(torch.abs(moment), 1/k)
    moment = moment * (1-neg_mask) -moment * neg_mask
    return moment.mean(dim=0)

def k_moment_loss(actual_moment, estimated_moment):
    err =  actual_moment - estimated_moment
    err = torch.clip(err, min=0)
    return torch.mean(err)


def get_stats_loss_func(pred_tiled:np.ndarray, k:int):
    # pred_tiled: N x C x H x W
    print(f'Creating stats loss function with k={k}')
    moments = [k_moment(torch.Tensor(pred_tiled), i) for i in range(1, k+1)]
    def stats_loss_func(two_channel_prediction):
        loss = 0
        for i in range(k):
            est_moment = k_moment(two_channel_prediction, i+1)
            # loss += k_moment_loss(moments[i].to(est_moment.device), est_moment)/k
            loss += k_moment_loss(moments[i].to(est_moment.device), est_moment)
        return loss
    return stats_loss_func

    # mean_channels = torch.Tensor(np.mean(pred_tiled, axis=(2,3)).mean(axis=0))
    # std_channels = torch.Tensor(np.std(pred_tiled, axis=(2,3)).mean(axis=0))
    # def stats_loss_func(two_channel_prediction):
    #     mean_pred = torch.mean(two_channel_prediction, dim=(2,3)).mean(dim=0)
    #     std_pred = torch.std(two_channel_prediction, dim=(2,3)).mean(dim=0)
    #     device = std_pred.device
    #     mean_err =  mean_channels.to(device) - mean_pred
    #     mean_err = torch.clip(mean_err, min=0)
    #     std_err = std_channels.to(device) - std_pred
    #     std_err = torch.clip(std_err, min=0)
    #     mean_loss = torch.mean(mean_err)
    #     std_loss = torch.mean(std_err)
    #     return  mean_loss + std_loss
    # return stats_loss_func


def finetune_two_forward_passes(model, finetune_dset, finetune_val_dset, transform_obj, max_step_count=10000, batch_size=16, skip_pixels=0,
                                scalar_params_dict=None,
                                validation_step_freq=1000,
                                optimization_params_dict=None, stats_enforcing_loss_fn=None, 
                                num_workers=4,
                                # lookback=10, 
                                k_augmentations=1,sample_mixing_ratio=False, psnr_evaluation=False, tmp_dir='/group/jug/ashesh/tmp'):
    
    import os
    from datetime import datetime
    assert 2*skip_pixels <= finetune_dset[0][0].shape[-1], "skip_pixels should be less than half of the image size"
    assert 2*skip_pixels <= finetune_val_dset[0][0].shape[-2], "skip_pixels should be less than half of the image size"
    finetune_dset.train_mode()
    finetune_val_dset.eval_mode()

    tmp_path = f'{tmp_dir}/finetune_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(tmp_path, exist_ok=True)
    # enable dropout.
    # model.train()
    print(f'Finetuning with {k_augmentations} augmentations, batch size {batch_size}, max step count {max_step_count}, sample mixing ratio {sample_mixing_ratio}')
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
    # if factor1 is None:
    #     factor1 = torch.nn.Parameter(torch.tensor(1.0).cuda())

    # if offset1 is None:
    #     offset1 = torch.nn.Parameter(torch.tensor(0.0).cuda())

    # if factor2 is None:
    #     factor2 = torch.nn.Parameter(torch.tensor(1.0).cuda())
    # if offset2 is None:
    #     offset2 = torch.nn.Parameter(torch.tensor(0.0).cuda())
    val_dict = perform_validation(finetune_val_dset, batch_size=batch_size, num_workers=num_workers, 
                                              mixing_ratio=mixing_ratio, factor1=factor1, offset1=offset1, factor2=factor2, 
                                              offset2=offset2, skip_pixels=skip_pixels,pred_func=pred_func, 
                                              transform_obj=transform_obj, stats_enforcing_loss_fn=stats_enforcing_loss_fn, tmp_path=tmp_path,
                                              psnr_evaluation=psnr_evaluation)
    print(val_dict) 
    
    # define an optimizer
    # opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt = torch.optim.Adam(optimization_params_dict['parameters'], lr=optimization_params_dict['lr'], weight_decay=0)

    loss_arr = []
    loss_inp_arr = []
    loss_pred_arr = []
    loss_inp2_arr = []
    stats_loss_arr = []

    best_val_loss = 1e6

    factor1_arr = []
    offset1_arr = []

    factor2_arr = []
    offset2_arr = []
    mixing_ratio_arr= []
    psnr_arr = []
    best_factors = best_offsets = None

    cnt = 0
    while True:
        dloader = DataLoader(finetune_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        bar = tqdm(total =len(dloader), desc='Finetuning')
        for i, (inp, tar) in enumerate(dloader):
            inp = inp.cuda()
            # reset the gradients
            opt.zero_grad()

            keys = ['loss_inp', 'loss_pred', 'loss_inp2', 'stats_loss']
            agg_loss_dict = {key: 0 for key in keys}
            for _ in range(k_augmentations):
                # apply the augmentations
                loss_dict = SSL_loss(pred_func, inp, transform_obj, mixing_ratio=mixing_ratio, factor1=factor1, offset1=offset1, factor2=factor2, 
                                    offset2=offset2, skip_pixels=skip_pixels,
                                    stats_enforcing_loss_fn=stats_enforcing_loss_fn, sample_mixing_ratio=sample_mixing_ratio)
                for key in keys:
                    agg_loss_dict[key] += loss_dict[key]/ k_augmentations
                
            loss = agg_loss_dict['loss_inp'] + agg_loss_dict['loss_pred'] + agg_loss_dict['loss_inp2'] + agg_loss_dict['stats_loss']
            # print('Loss upper level:',agg_loss_dict['loss_inp'].item(), agg_loss_dict['loss_pred'].item(), agg_loss_dict['loss_inp2'], agg_loss_dict['stats_loss'].item())
            loss.backward()
            loss_arr.append(loss.item())
            loss_inp_arr.append(agg_loss_dict['loss_inp'].item())
            loss_pred_arr.append(agg_loss_dict['loss_pred'].item())
            loss_inp2_arr.append(agg_loss_dict['loss_inp2'].item() if torch.is_tensor(agg_loss_dict['loss_inp2']) else agg_loss_dict['loss_inp2'])
            stats_loss_arr.append(agg_loss_dict['stats_loss'].item() if torch.is_tensor(agg_loss_dict['stats_loss']) else agg_loss_dict['stats_loss'])
            
            factor1_arr.append(factor1.item())
            offset1_arr.append(offset1.item())
            factor2_arr.append(factor2.item())
            offset2_arr.append(offset2.item())
            cnt += len(inp)
            mixing_ratio_arr.append(mixing_ratio.item())
            opt.step()
            
            bar.set_description(f'[{cnt}] Loss:{np.mean(loss_arr[-10:]):.2f}')
            bar.update(1)
            
            if validation_step_freq is not None and cnt //validation_step_freq > (cnt - len(inp))//validation_step_freq:
                val_dict = perform_validation(finetune_val_dset, batch_size=batch_size, num_workers=num_workers, 
                                              mixing_ratio=mixing_ratio, factor1=factor1, offset1=offset1, factor2=factor2, 
                                              offset2=offset2, skip_pixels=skip_pixels,pred_func=pred_func, 
                                              transform_obj=transform_obj, stats_enforcing_loss_fn=stats_enforcing_loss_fn, tmp_path=tmp_path,
                                              psnr_evaluation=psnr_evaluation)
                val_loss = val_dict['loss']
                psnr_arr.append(val_dict['psnr'])
                # validate the model
                if val_loss < best_val_loss:
                    print('Saving best model')
                    best_factors = [factor1.item(), factor2.item()]
                    best_offsets = [offset1.item(), offset2.item()]
                    best_val_loss = val_loss
                    # save the model
                    torch.save(model.state_dict(), f'{tmp_path}/best_model.pth')

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

    return {'loss': loss_arr, 'best_loss': best_val_loss, 'best_factors': best_factors, 
            'best_offsets': best_offsets, 'factor1': factor1_arr, 'offset1': offset1_arr, 
            'factor2': factor2_arr, 'offset2': offset2_arr, 'mixing_ratio': mixing_ratio_arr,
            'loss_inp': loss_inp_arr, 'loss_pred': loss_pred_arr,
            'loss_inp2': loss_inp2_arr,
            'stats_loss': stats_loss_arr,
            'psnr': psnr_arr,}

def perform_validation(finetune_val_dset, batch_size=None, num_workers=None, mixing_ratio=None, factor1=None, offset1=None, factor2=None, offset2=None, skip_pixels=None,pred_func=None, transform_obj=None, stats_enforcing_loss_fn=None, tmp_path=None, psnr_evaluation=False):
    print('Validating the model', end=' ')  
    finetune_val_dset.eval_mode()

    val_loss = 0
    val_steps = 0
    val_steps_max = None if psnr_evaluation else 1000
    val_dloader = DataLoader(finetune_val_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pred_arr = []
    tar_arr = []
    for j, (inp, tar) in tqdm(enumerate(val_dloader)):
        inp = inp.cuda()
        with torch.no_grad():
            loss_dict = SSL_loss(pred_func, inp, transform_obj, mixing_ratio=mixing_ratio, factor1=factor1, offset1=offset1, factor2=factor2, 
                                offset2=offset2, skip_pixels=skip_pixels,
                                stats_enforcing_loss_fn=stats_enforcing_loss_fn, return_predictions=psnr_evaluation)
            val_loss += loss_dict['loss_inp'].item() #+ loss_dict['loss_pred'].item()
            val_steps += len(inp)
            if val_steps_max is not None and val_steps >= val_steps_max:
                break

            if psnr_evaluation:
                pred_arr.append(loss_dict['first_prediction'].cpu().numpy())
                tar_arr.append(tar.cpu().numpy())

    val_loss /= j
    psnr_arr = None
    if psnr_evaluation:
        pred_arr = np.concatenate(pred_arr, axis=0)
        tar_arr = np.concatenate(tar_arr, axis=0)
        pred_stitched = stitch_predictions(pred_arr, finetune_val_dset)
        tar_stitched = stitch_predictions(tar_arr, finetune_val_dset)
        nC = pred_stitched[0].shape[-1]
        psnr_arr = []
        for ch_idx in range(nC):
            avg_psnr = np.mean([RangeInvariantPsnr(tar_stitched[i][...,ch_idx], pred_stitched[i][...,ch_idx]) for i in range(len(tar_stitched))])
            psnr_arr.append(avg_psnr)
        psnr_arr_str = [f'{x:.2f}' for x in psnr_arr]
        print(f'Validation PSNR: {" ".join(psnr_arr_str)}' )
    else:
        print(f'Validation Loss: {val_loss:.4f}')
    
    finetune_val_dset.train_mode()
    return {'loss': val_loss, 'psnr': psnr_arr}

def get_best_mixing_t(pred, inp_unnorm, enable_tqdm=False):
    mean_psnr_arr = []
    std_psnr_arr = []
    t_values = np.arange(0.0,1.0, 0.05) 
    if enable_tqdm:
        t_values_iter = tqdm(t_values)
    else:
        t_values_iter = t_values
    for t in t_values_iter:
        inp_tiled = [(t *pred[i][...,0] + (1-t)*pred[i][...,1]) for i in range(len(pred))]
        psnr_values = [RangeInvariantPsnr(inp_unnorm[i]*1.0, inp_tiled[i]).item() for i in range(len(inp_unnorm))]
        mean_psnr_arr.append(np.mean(psnr_values))
        std_psnr_arr.append(np.std(psnr_values))

    best_idx = np.argmax(mean_psnr_arr)
    best_t_estimate = t_values[best_idx]
    if enable_tqdm:
        print(f'Best t value: {best_t_estimate}', mean_psnr_arr[best_idx])
    return best_t_estimate, mean_psnr_arr[best_idx]