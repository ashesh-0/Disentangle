import numpy as np
import scipy


def get_percentiles(error_array, var_array, factor, offset):
    """
    error_array: N 
    var_array:   N x MC
    factor:      scalar
    around_center: bool
    """
    assert var_array.ndim == 2
    assert error_array.ndim == 1
    percentiles = np.array([scipy.stats.percentileofscore(var_array[elem_idx] * factor + offset ,error_array[elem_idx]) for elem_idx in range(var_array.shape[0])])
    # if arround_center:
    #     percentiles = 2*np.abs(percentiles - 50)
    # print('min', 'max', percentiles.min(), percentiles.max())
    return percentiles

def get_percentage_occurance(percentile_list, error_array, var_array, factor, offset=0, around_center=False):
    """
    error_array: N 
    var_array:   N x MC
    factor:      scalar
    around_center: bool
    Returns 3 values:
        [percentage in lesser interval] [percentage in desired interval] [percentage in larger interval]
    """
    percentiles = get_percentiles(error_array, var_array, factor, offset)
    if around_center:
        output = []
        for kth_percentile in percentile_list:
            left_per = 50 - kth_percentile/2
            right_per = 50 + kth_percentile/2
            left_mass = 100*np.mean(percentiles < left_per)
            right_mass = 100*np.mean(percentiles > right_per)
            # print(left_mass, right_mass)
            output.append((left_mass, 100 - left_mass - right_mass, right_mass))
        return output

    else:
        output = []
        for kth_percentile in percentile_list:
            percent_desired = 100 * np.mean(percentiles <= kth_percentile)
            output.append((0,percent_desired, 100 - percent_desired))
        return output

def one_step_optimization(percentile_to_optimize, error_array, var_array, factor, delta, around_center=False):
    """
    We take the percentile from the left.
    """
    achieved_percentile = get_percentage_occurance([percentile_to_optimize], error_array, var_array, factor, around_center=around_center)[0]
    # in this case, we just care about the mass present in the desired interval
    achieved_percentile = achieved_percentile[1]

    if achieved_percentile < percentile_to_optimize:
        factor += delta
    elif achieved_percentile > percentile_to_optimize:
        while factor <= delta:
            delta = delta/2
        factor -= delta
    return factor, delta, achieved_percentile


def monotone(arr):
    """
    Check if the array is monotone increasing or decreasing
    """
    return np.all(np.diff(arr) > 0) or np.all(np.diff(arr) < 0)

def grid_search(error_array, var_array, init_factor=1.0, init_delta=5, percentile_to_optimize=50, around_center=False, direction_change_limit=6, upscaling_patience=3):
    # ch_idx = 0
    factor = init_factor
    used_factors = [factor]
    delta = init_delta
    direction_change_count = 0
    last_upscaling = 0
    mean_var = mean_error = 0
    
    if around_center:
        mean_var = np.median(var_array)
        mean_error = np.median(error_array)
        var_array = var_array - mean_var
        error_array = error_array - mean_error
    


    while True:
        next_factor, delta, achieved_percentile = one_step_optimization(percentile_to_optimize, error_array, var_array, factor, delta, around_center=around_center)
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
    
    offset = mean_error - mean_var * factor
    return factor, offset, achieved_percentile


def plot_coverage_plot(var, err, factors, offsets, around_center=False):
    import matplotlib.pyplot as plt
    ncols = var.shape[1]
    _,ax = plt.subplots(figsize=(5*ncols,5),ncols=ncols)
    data = {}
    linestyles = [
        'solid',
        'dotted',
        'dashed',
        'dashdot',]
    for col_idx in range(ncols):    
        # percentiles = x
        
        data[col_idx] = {'scaled': None, 'unscaled': None}
        x = np.linspace(0,100,100)
        # scaled. 
        less_y, y, more_y = zip(*get_percentage_occurance(x, err[:,col_idx], var[:,col_idx], factor=factors[col_idx], offset=offsets[col_idx], around_center=around_center))
        data[col_idx]['scaled'] = [less_y, y, more_y]
        ax[col_idx].plot(x, y, label=f'Scaled', linestyle= linestyles[0])
        less_y, y, more_y = zip(*get_percentage_occurance(x, err[:,col_idx], var[:,col_idx], factor=1, offset=0, around_center=around_center))
        data[col_idx]['unscaled'] = [less_y, y, more_y]
        ax[col_idx].plot(x, y, label=f'Unscaled', linestyle= linestyles[0])

        ax[col_idx].grid()
        # facecolor to gray
        ax[col_idx].set_facecolor('lightgray')
        ax[col_idx].set_xlabel('Percentile (Confidence level)')
        ax[col_idx].set_ylabel('Percentage of data (Empirical coverage)')
    # plot y=x line
        ax[col_idx].plot([0,100],[0,100], 'k--')

    ax[-1].legend()
    return data
