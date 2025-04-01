
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from finetunesplit.loss import SSL_loss


def finetune_two_forward_passes(model, val_dset, transform_obj, max_step_count=10000, batch_size=16, skip_pixels=0,
                                scalar_params_dict=None,
                                optimization_params_dict=None, lookback=10):
    
    # enable dropout.
    # model.train()

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

            loss_dict = SSL_loss(pred_func, inp, transform_obj, mixing_ratio=mixing_ratio, factor1=factor1, offset1=offset1, factor2=factor2, offset2=offset2, skip_pixels=skip_pixels)
            # return {'loss_pred':loss_pred, 'loss_inp2':loss_inp2, 'loss_inp':loss_inp}
            loss = loss_dict['loss_inp'] + loss_dict['loss_pred'] + loss_dict['loss_inp2']
            loss.backward()
            loss_arr.append(loss.item())
            loss_inp_arr.append(loss_dict['loss_inp'].item())
            loss_pred_arr.append(loss_dict['loss_pred'].item())
            loss_inp2_arr.append(loss_dict['loss_inp2'].item() if torch.is_tensor(loss_dict['loss_inp2']) else loss_dict['loss_inp2'])
            
            factor1_arr.append(factor1.item())
            offset1_arr.append(offset1.item())
            factor2_arr.append(factor2.item())
            offset2_arr.append(offset2.item())
            cnt += len(inp)
            mixing_ratio_arr.append(mixing_ratio.item())
            opt.step()
            rolling_loss = np.mean(loss_arr[-lookback:])
            if rolling_loss < best_loss and len(loss_arr) > 10:
                best_loss = loss.item()
                print(f'Loss: {rolling_loss:.2f}')
                best_factors = [factor1.item(), factor2.item()]
                best_offsets = [offset1.item(), offset2.item()]
            # print(f'Loss: {loss.item():.2f}')
            if cnt >= max_step_count:
                break
        if cnt >= max_step_count:
            break
    
    model.eval()
    return {'loss': loss_arr, 'best_loss': best_loss, 'best_factors': best_factors, 
            'best_offsets': best_offsets, 'factor1': factor1_arr, 'offset1': offset1_arr, 
            'factor2': factor2_arr, 'offset2': offset2_arr, 'mixing_ratio': mixing_ratio_arr,
            'loss_inp': loss_inp_arr, 'loss_pred': loss_pred_arr,
            'loss_inp2': loss_inp2_arr}