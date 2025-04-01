
def get_input_normalized(channel_data_normalized, t_estimate, channel_pos='last'):
    assert channel_pos in ['last', 'second']
    if channel_pos == 'last':
        return channel_data_normalized[...,0] * t_estimate + channel_data_normalized[...,1] * (1-t_estimate)
    elif channel_pos == 'second':
        return channel_data_normalized[:,0] * t_estimate + channel_data_normalized[:,1] * (1-t_estimate)

def get_input_unnormalized(pred_normalized, t_estimate, dset, channel_pos='last'):
    mean_val, std_val = dset.dsets[0].get_mean_std_for_input()
    input_norm = get_input_normalized(pred_normalized, t_estimate, channel_pos=channel_pos)
    return input_norm * std_val + mean_val