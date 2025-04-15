
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def finetune_two_forward_passes(model, val_dset, transform_obj, max_step_count=10000, batch_size=16, skip_pixels=0,
                                scalar_params_dict=None,
                                optimization_params_dict=None, stats_enforcing_loss_fn=None, lookback=10, k_augmentations=1,sample_mixing_ratio=False):
    
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
    
    # define an optimizer
    # opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt = torch.optim.Adam(optimization_params_dict['parameters'], lr=optimization_params_dict['lr'], weight_decay=0)

    loss_arr = []
    loss_inp_arr = []
    loss_pred_arr = []
    loss_inp2_arr = []
    stats_loss_arr = []

    best_loss = 1e6

    factor1_arr = []
    offset1_arr = []

    factor2_arr = []
    offset2_arr = []
    mixing_ratio_arr= []
    best_factors = best_offsets = None

    cnt = 0
    while True:
        dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=4)
        for i, (inp, tar) in tqdm(enumerate(dloader)):
            if i >= max_step_count:
                break
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
                
            # return {'loss_pred':loss_pred, 'loss_inp2':loss_inp2, 'loss_inp':loss_inp}
            loss = agg_loss_dict['loss_inp'] + agg_loss_dict['loss_pred'] + agg_loss_dict['loss_inp2'] + agg_loss_dict['stats_loss']
            
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
            rolling_loss = np.mean(loss_inp_arr[-lookback:])
            if rolling_loss < best_loss and len(loss_inp_arr) > 10:
                best_loss = rolling_loss.item()
                print(f'Loss Inp Rolling(10): {rolling_loss:.2f}')
                best_factors = [factor1.item(), factor2.item()]
                best_offsets = [offset1.item(), offset2.item()]
            # print(f'Loss: {loss.item():.2f}')
            if cnt >= max_step_count:
                break
        if cnt >= max_step_count:
            break
    
    model.eval()
    if best_factors is None:
        best_factors = [factor1.item(), factor2.item()]
        best_offsets = [offset1.item(), offset2.item()]
    
    return {'loss': loss_arr, 'best_loss': best_loss, 'best_factors': best_factors, 
            'best_offsets': best_offsets, 'factor1': factor1_arr, 'offset1': offset1_arr, 
            'factor2': factor2_arr, 'offset2': offset2_arr, 'mixing_ratio': mixing_ratio_arr,
            'loss_inp': loss_inp_arr, 'loss_pred': loss_pred_arr,
            'loss_inp2': loss_inp2_arr,
            'stats_loss': stats_loss_arr}