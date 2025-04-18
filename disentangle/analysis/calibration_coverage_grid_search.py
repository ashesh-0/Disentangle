import numpy as np
import scipy


def get_percentiles(error_array, var_array, factor):
    """
    error_array: N 
    var_array:   N x MC
    factor:      scalar
    around_center: bool
    """
    assert var_array.ndim == 2
    assert error_array.ndim == 1
    percentiles = np.array([scipy.stats.percentileofscore(var_array[elem_idx] * factor ,error_array[elem_idx]) for elem_idx in range(var_array.shape[0])])
    # if arround_center:
    #     percentiles = 2*np.abs(percentiles - 50)
    # print('min', 'max', percentiles.min(), percentiles.max())
    return percentiles

def get_percentage_occurance(percentile_list, error_array, var_array, factor):
    """
    error_array: N 
    var_array:   N x MC
    factor:      scalar
    around_center: bool
    """
    percentiles = get_percentiles(error_array, var_array, factor)
    output = []
    for kth_percentile in percentile_list:
        output.append(100 * np.mean(percentiles <= kth_percentile))
    return output

def one_step_optimization(percentile_to_optimize, error_array, var_array, factor, delta):
    achieved_percentile = get_percentage_occurance([percentile_to_optimize], error_array, var_array, factor)[0]
    if achieved_percentile < percentile_to_optimize:
        factor += delta
    elif achieved_percentile > percentile_to_optimize:
        if factor < delta:
            delta = delta/2
        factor -= delta
    return factor, delta, achieved_percentile

def monotone(arr):
    """
    Check if the array is monotone increasing or decreasing
    """
    return np.all(np.diff(arr) > 0) or np.all(np.diff(arr) < 0)

def grid_search(error_array, var_array, init_factor=1.0, init_delta=5, percentile_to_optimize=50, direction_change_limit=6, upscaling_patience=3):
    # ch_idx = 0
    factor = init_factor
    used_factors = [factor]
    delta = init_delta
    direction_change_count = 0
    last_upscaling = 0

    while True:
        next_factor, delta, achieved_percentile = one_step_optimization(percentile_to_optimize, error_array, var_array, factor, delta)
        if (next_factor > factor  and factor < used_factors[-1]) or (next_factor < factor and factor > used_factors[-1]):
            delta = delta/2
            direction_change_count += 1
        # achieved_percentile = get_percentage_occurance([percentile_to_optimize], err, var, factor)[0]
        print(f'{achieved_percentile:.2f} {factor} D{direction_change_count}')
        if direction_change_count > direction_change_limit:
            break
        used_factors.append(factor)
        if direction_change_count==0 and last_upscaling >= upscaling_patience and len(used_factors)>=3 and monotone(used_factors[-3:]):
            # do it only if there has not been a direction change ever.
            print(f'Upscaling {delta} -> {delta*2}')
            delta = delta*2
            last_upscaling = 0
        else:
            last_upscaling += 1
        
        factor = next_factor
    
    return factor, achieved_percentile