import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch as t 
import preprocess

ALL_MODEL_NAMES = ['bus_breakdown', 'chimpanzees', 'movielens', 'occupancy', 'radon']

DEFAULT_ALPHA_FUNC = lambda i, num_lrs: 1 if i == 0 else 1 - 0.5*i/(num_lrs-1)

def load_results(model_name):
    with open(f'{model_name}_results/best.pkl', 'rb') as f:
        return pickle.load(f)
    
def smooth(x, window):
    # result = np.convolve(x, np.ones(window)/window, mode='valid')
    
    result = np.zeros_like(x)

    result[0] = x[0]

    for i in range(1,len(x)):
        if x[i] != np.nan:
            result[i] = x[max(i-window, 0):i].mean()
        # result[i,:] = np.nanmean(x[max(i-window, 0):i,:], 1)

    return result

def plot_method_K_lines(ax, 
                        model_results, 
                        method,
                        K, 
                        metric_name, 
                        num_lrs, 
                        colour,
                        x_axis_iters = True,
                        smoothing_window = 1,
                        x_lim_iters = None,
                        error_bars = False,
                        alpha_func = DEFAULT_ALPHA_FUNC):
    
    metric     = model_results[method][K][metric_name]
    stderrs    = model_results[method][K][f'{metric_name[:-1]}_stderrs']
    iter_times = model_results[method][K]['iter_times'].cumsum(axis=1)

    lrs = model_results[method][K]['lrs']

    for i in range(num_lrs):
        if i < metric.shape[0]:
            smoothed_metric = smooth(metric[i, :], smoothing_window)

            alpha_val = alpha_func(i, num_lrs)

            xs = np.arange(smoothed_metric.shape[0]) if x_axis_iters else iter_times[i]

            if x_lim_iters is None:
                x_lim_iters = len(xs)

            ax.plot(xs[:x_lim_iters], smoothed_metric[:x_lim_iters], label=f'{method.upper()}: lr={lrs[i]}', color=colour, alpha=alpha_val)

            if error_bars:
                ax.fill_between(xs, smoothed_metric - stderrs[i, :], smoothed_metric + stderrs[i, :], color=colour, alpha=0.2*alpha_val)

    
def plot_all_2col(model_names  = ALL_MODEL_NAMES,
                  Ks_to_plot   = 'largest',
                  num_lrs      = 1,
                  x_axis_iters = True,
                  x_lim        = None,
                  smoothing_window = 3,
                  error_bars   = False,
                  alpha_func   = DEFAULT_ALPHA_FUNC,
                  save_pdf     = False,
                  filename_end = ""):
    
    results = {model_name: load_results(model_name) for model_name in model_names}

    if Ks_to_plot == 'largest':
        Ks_to_plot = {model_name: [max(results[model_name]['qem'].keys())] for model_name in model_names}
    elif Ks_to_plot == 'smallest':
        Ks_to_plot = {model_name: [min(results[model_name]['qem'].keys())] for model_name in model_names}
    elif Ks_to_plot == 'all':
        Ks_to_plot = {model_name: list(results[model_name]['qem'].keys()) for model_name in model_names}

    num_rows = sum([len(Ks_to_plot[model_name]) for model_name in model_names])

    fig, axs = plt.subplots(num_rows, 2, figsize=(7, num_rows*2), sharex=x_axis_iters)

    row_counter = 0
    for i, model_name in enumerate(model_names):
        for k, K in enumerate(Ks_to_plot[model_name]):
            for j, method_name in enumerate(results[model_name].keys()):
                colour = f'C{j}'

                axs[row_counter,0].set_title(f'{model_name.upper()} (K={K})')

                plot_method_K_lines(ax = axs[row_counter,0], 
                                    model_results = results[model_name],
                                    method = method_name,
                                    K = K, 
                                    metric_name = 'elbos',
                                    num_lrs = num_lrs,
                                    colour = colour,
                                    x_axis_iters = x_axis_iters,
                                    smoothing_window = smoothing_window,
                                    error_bars = error_bars,
                                    alpha_func = alpha_func)

                axs[row_counter,0].set_ylabel('ELBO')

                if x_axis_iters:
                    axs[row_counter,0].set_xlim(0, x_lim)
                else:
                    axs[row_counter,0].set_xscale('log')

                axs[row_counter,0].set_ylim(None, None)
                
                plot_method_K_lines(ax = axs[row_counter,1], 
                                    model_results = results[model_name],
                                    method = method_name,
                                    K = K, 
                                    metric_name = 'p_lls',
                                    num_lrs = num_lrs,
                                    colour = colour,
                                    x_axis_iters = x_axis_iters,
                                    smoothing_window = smoothing_window,
                                    error_bars = error_bars,
                                    alpha_func = alpha_func)
                
                axs[row_counter,1].set_ylabel('Predictive log-likelihood')

                if x_axis_iters:
                    axs[row_counter,1].set_xlim(0, x_lim)
                else:
                    axs[row_counter,1].set_xscale('log')

                axs[row_counter,1].set_ylim(None, None)

                axs[row_counter,1].legend()

            row_counter += 1

        axs[-1,0].set_xlabel('Iterations' if x_axis_iters else 'Time (s)')
        axs[-1,1].set_xlabel('Iterations' if x_axis_iters else 'Time (s)')

    fig.tight_layout()

    plt.savefig(f'plots/all_2col{filename_end}.png')
    if save_pdf:
        plt.savefig(f'plots/all_2col{filename_end}.pdf')

def plot_all_2row(model_names  = ALL_MODEL_NAMES,
                  Ks_to_plot   = 'largest',
                  num_lrs      = 1,
                  x_axis_iters = True,
                  x_lim        = None,
                  smoothing_window = 1,
                  error_bars   = False,
                  alpha_func   = DEFAULT_ALPHA_FUNC,
                  save_pdf     = False,
                  filename_end = ""):
    
    results = {model_name: load_results(model_name) for model_name in model_names}

    if Ks_to_plot == 'largest':
        Ks_to_plot = {model_name: [max(results[model_name]['qem'].keys())] for model_name in model_names}
    elif Ks_to_plot == 'smallest':
        Ks_to_plot = {model_name: [min(results[model_name]['qem'].keys())] for model_name in model_names}
    elif Ks_to_plot == 'all':
        Ks_to_plot = {model_name: list(results[model_name]['qem'].keys()) for model_name in model_names}

    num_cols = sum([len(Ks_to_plot[model_name]) for model_name in model_names])

    fig, axs = plt.subplots(2, num_cols, figsize=(num_cols*3, 7), sharex=x_axis_iters)

    col_counter = 0
    for i, model_name in enumerate(model_names):
        for k, K in enumerate(Ks_to_plot[model_name]):
            for j, method_name in enumerate(results[model_name].keys()):
                colour = f'C{j}'

                axs[0,col_counter].set_title(f'{model_name.upper()}\nK={K}')

                plot_method_K_lines(ax = axs[0,col_counter], 
                                    model_results = results[model_name],
                                    method = method_name,
                                    K = K, 
                                    metric_name = 'elbos',
                                    num_lrs = num_lrs,
                                    colour = colour,
                                    x_axis_iters = x_axis_iters,
                                    smoothing_window = smoothing_window,
                                    x_lim_iters=x_lim,
                                    error_bars = error_bars,
                                    alpha_func = alpha_func)

                if x_axis_iters:
                    axs[0,col_counter].set_xlim(0, x_lim)
                # else:
                #     axs[0,col_counter].set_xscale('log')

                axs[0,col_counter].set_ylim(None, None)
                
                plot_method_K_lines(ax = axs[1,col_counter], 
                                    model_results = results[model_name],
                                    method = method_name,
                                    K = K, 
                                    metric_name = 'p_lls',
                                    num_lrs = num_lrs,
                                    colour = colour,
                                    x_axis_iters = x_axis_iters,
                                    smoothing_window = smoothing_window,
                                    x_lim_iters=x_lim,
                                    error_bars = error_bars,
                                    alpha_func = alpha_func)
                

                if x_axis_iters:
                    axs[1,col_counter].set_xlim(0, x_lim)
                # else:
                #     axs[1,col_counter].set_xscale('log')

                axs[1,col_counter].set_ylim(None, None)

                axs[1,col_counter].set_xlabel('Iterations' if x_axis_iters else 'Time (s)')


                axs[1,col_counter].legend()

            col_counter += 1

        axs[0,0].set_ylabel('ELBO')
        axs[1,0].set_ylabel('Predictive log-likelihood')

    fig.tight_layout()

    plt.savefig(f'plots/all_2row{filename_end}.png')
    if save_pdf:
        plt.savefig(f'plots/all_2row{filename_end}.pdf')

def plot_avg_iter_time_per_K(model_names  = ALL_MODEL_NAMES,
                             save_pdf     = False,
                             filename_end = ""):
    
    results = {model_name: load_results(model_name) for model_name in model_names}

    fig, axs = plt.subplots(1, len(model_names), figsize=(len(model_names)*3, 3))

    for i, model_name in enumerate(model_names):
        # times = {K: {method_name: results[model_name][method_name][K]['iter_times'] for method_name in results[model_name].keys()} for K in results[model_name]['qem'].keys()}
        Ks = results[model_name]['qem'].keys()

        x = np.arange(len(Ks))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        
        for j, method_name in enumerate(results[model_name].keys()):
            avg_iter_times_per_K = [np.nanmean(results[model_name][method_name][K]['iter_times']) for K in Ks]
            
            offset = width * multiplier
            rects = axs[i].bar(x + offset, avg_iter_times_per_K, width, label=method_name.upper())
            # axs[i].bar_label(rects, padding=3)
            multiplier += 1

        axs[i].set_xlabel('K')
        axs[i].set_title(f'{model_name.upper()}')
        axs[i].set_xticks(x + width, results[model_name][method_name].keys())

    axs[0].legend()
    axs[0].set_ylabel('Average iteration time (s)')

    fig.tight_layout()

    plt.savefig(f'plots/avg_iter_time_per_K{filename_end}.png')

if __name__ == "__main__":
    validation_iter_number = 25
    iteration_x_lim = 2*validation_iter_number

    for model_name in ALL_MODEL_NAMES:
        if model_name == 'occupancy':
            preprocess.get_best_results(model_name, validation_iter_number=validation_iter_number, method_names=['qem', 'rws'])
        else:
            preprocess.get_best_results(model_name, validation_iter_number=validation_iter_number)
    
    # plot_all_2col(Ks_to_plot='all', num_lrs=10, x_lim = iteration_x_lim, filename_end="_EVERYTHING")

    # plot_all_2col(Ks_to_plot='all', num_lrs=1, x_lim = iteration_x_lim, filename_end="_BEST")

    # plot_all_2row(Ks_to_plot='all', num_lrs=10, x_lim = iteration_x_lim, filename_end="_EVERYTHING")

    # plot_all_2row(Ks_to_plot='all', num_lrs=1, x_lim = iteration_x_lim, filename_end="_BEST")

    # plot_all_2row(Ks_to_plot='all', num_lrs=1, x_lim = iteration_x_lim, filename_end="_BEST_errs", error_bars=True)

    # plot_all_2row(Ks_to_plot='all', num_lrs=2, x_lim = iteration_x_lim, filename_end="_BEST_2lrs")

    # plot_all_2row(Ks_to_plot='all', num_lrs=3, x_lim = iteration_x_lim, filename_end="_BEST_3lrs")


    best_Ks = {'bus_breakdown': [30], 'chimpanzees': [5], 'movielens': [30], 'occupancy': [3], 'radon': [3]}

    # plot_all_2col(Ks_to_plot=best_Ks,
    #                num_lrs=1, x_lim = iteration_x_lim, filename_end="_BEST_SPECIFIC_Ks")
    
    # plot_all_2col(Ks_to_plot=best_Ks,
    #                num_lrs=1, filename_end="_BEST_SPECIFIC_Ks_TIME", x_axis_iters=False)
    
    # plot_all_2row(Ks_to_plot=best_Ks,
    #                num_lrs=1, x_lim = iteration_x_lim, filename_end="_BEST_SPECIFIC_Ks")
    
    plot_all_2row(Ks_to_plot=best_Ks,
                   num_lrs=1, filename_end="_BEST_SPECIFIC_Ks_TIME", x_axis_iters=False)
    
    # # with std_err bars
    # plot_all_2row(Ks_to_plot=best_Ks,
    #                num_lrs=1, x_lim = iteration_x_lim, filename_end="_BEST_SPECIFIC_Ks_err",
    #                error_bars=True)
    
    # plot_all_2row(Ks_to_plot=best_Ks,
    #                num_lrs=1, filename_end="_BEST_SPECIFIC_Ks_TIME_err", x_axis_iters=False,
    #                error_bars=True)
            
    # # now with 2 lrs
    # plot_all_2row(Ks_to_plot=best_Ks,
    #                num_lrs=2, x_lim = iteration_x_lim, filename_end="_BEST_SPECIFIC_Ks_2lrs")
    
    # plot_all_2row(Ks_to_plot=best_Ks,
    #                num_lrs=2, filename_end="_BEST_SPECIFIC_Ks_TIME_2lrs", x_axis_iters=False)
            

    # # now with 3 lrs
    # plot_all_2row(Ks_to_plot=best_Ks,
    #                num_lrs=3, x_lim = iteration_x_lim, filename_end="_BEST_SPECIFIC_Ks_3lrs")
    
    # plot_all_2row(Ks_to_plot=best_Ks,
    #                num_lrs=3, filename_end="_BEST_SPECIFIC_Ks_TIME_3lrs", x_axis_iters=False)
            
    # # with 2 lrs AND std_err bars
    # plot_all_2row(Ks_to_plot=best_Ks,
    #                num_lrs=2, x_lim = iteration_x_lim, filename_end="_BEST_SPECIFIC_Ks_2lrs_err",
    #                error_bars=True)
    
    # plot_all_2row(Ks_to_plot=best_Ks,
    #                num_lrs=2, filename_end="_BEST_SPECIFIC_Ks_TIME_2lrs_err", x_axis_iters=False,
    #                error_bars=True)

    
    # plot_avg_iter_time_per_K()
        
