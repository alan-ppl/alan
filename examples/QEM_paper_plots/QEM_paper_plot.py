import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import torch as t 
import preprocess

# ALL_MODEL_NAMES = ['bus_breakdown', 'chimpanzees', 'movielens', 'occupancy', 'radon']
ALL_MODEL_NAMES = ['bus_breakdown', 'bus_breakdown_reparam', 'chimpanzees', 'movielens', 'movielens_reparam', 'occupancy', 'radon']

DEFAULT_ALPHA_FUNC = lambda i, num_lrs: 1 if i == 0 else 1 - 0.5*i/(num_lrs-1)

# using default matplotlib colours
DEFAULT_COLOURS = {'qem': "C1", 'rws': "C2", 'vi': "C3", 
                   'qem_nonmp': "C4", 'global_rws': "C2", 'global_vi': "C3"}

# using colorbrewer2.org
DEFAULT_COLOURS = {'qem': '#e7298a', 'qem_nonmp' : '#7570b3',
                   'rws': '#1b9e77', 'global_rws': '#1b9e77', 
                   'vi' : '#d95f02', 'global_vi' : '#d95f02'}

def load_results(model_name):
    with open(f'{model_name}_results/best.pkl', 'rb') as f:
        return pickle.load(f)
    
def smooth(x, window):
    # result = np.convolve(x, np.ones(window)/window, mode='valid')

    if window == 1:
        return x
    
    result = np.zeros_like(x)

    result[0] = x[0]

    for i in range(1,len(x)):
        if x[i] != np.nan:
            result[i] = x[max(i-window, 0):i].mean()
        # result[i,:] = np.nanmean(x[max(i-window, 0):i,:], 1)

    return result

class Zoomed_Inset:
    def __init__(self, model_name, methods, K, metric_name, xlims, ylims):
        self.model_name = model_name
        self.methods = methods
        self.K = K
        self.metric_name = metric_name
        self.xlims = xlims
        self.ylims = ylims

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
                        alpha_func = DEFAULT_ALPHA_FUNC,
                        force_lrs = None,
                        short_labels = True):
    
    short_label_dict = {'qem': 'QEM', 'rws': 'MP RWS', 'vi': 'MP VI', 'qem_nonmp': 'Global QEM', 'global_rws': 'Global RWS', 'global_vi': 'IWAE'}
    
    metric     = model_results[method][K][metric_name]
    stderrs    = model_results[method][K][f'{metric_name[:-1]}_stderrs']
    iter_times = model_results[method][K]['iter_times'].cumsum(axis=1)

    lrs = model_results[method][K]['lrs']

    min_x_val = np.inf

    if force_lrs is not None:
        # order the 0th (lr) dimension of metric, stderrs and iter_times to match the order of force_lrs
        lr_order = np.array([lrs.tolist().index(lr) for lr in force_lrs])
        metric = metric[lr_order, :]
        stderrs = stderrs[lr_order, :]
        iter_times = iter_times[lr_order, :]
        lrs = force_lrs
        num_lrs = len(lrs)

    for i in range(num_lrs):
        if i < metric.shape[0]:
            smoothed_metric = smooth(metric[i, :], smoothing_window)

            alpha_val = alpha_func(i, num_lrs)

            xs = np.arange(smoothed_metric.shape[0]) if x_axis_iters else iter_times[i]

            if x_lim_iters is None:
                x_lim_iters = len(xs)

            if short_labels:
                label = short_label_dict[method]
            else:
                label=f'{method.upper()}: lr={lrs[i]}'

            ax.plot(xs[:x_lim_iters], smoothed_metric[:x_lim_iters], label=label, color=colour, alpha=alpha_val)

            if error_bars:
                ax.fill_between(xs[:x_lim_iters], (smoothed_metric - stderrs[i, :])[:x_lim_iters], (smoothed_metric + stderrs[i, :])[:x_lim_iters], color=colour, alpha=0.2*alpha_val)

            if xs[x_lim_iters-1] < min_x_val:
                min_x_val = xs[x_lim_iters-1]

    return min_x_val
    
def plot_all_2col(model_names  = ALL_MODEL_NAMES,
                  Ks_to_plot   = 'largest',
                  num_lrs      = 1,
                  x_axis_iters = True,
                  x_lim        = None,
                  smoothing_window = 1,
                  error_bars   = False,
                  alpha_func   = DEFAULT_ALPHA_FUNC,
                  colours_dict = DEFAULT_COLOURS,
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
                colour = colours_dict[method_name]

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
        plt.savefig(f'plots/pdfs/all_2col{filename_end}.pdf')

def plot_all_2row(model_names   = ALL_MODEL_NAMES,
                  Ks_to_plot    = 'largest',
                  num_lrs       = 1,
                  x_axis_iters  = True,
                  x_lim         = 50,
                  smoothing_window = 1,
                  error_bars    = False,
                  alpha_func    = DEFAULT_ALPHA_FUNC,
                  colours_dict  = DEFAULT_COLOURS,
                  ylims         = {'elbo': {}, 'p_ll': {}},
                  zoomed_insets = [],
                  save_pdf      = False,
                  filename_end  = ""):
    
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

            axs[0,col_counter].set_title(f'{model_name.upper()}\nK={K}')

            for j, method_name in enumerate(results[model_name].keys()):
                colour = colours_dict[method_name]

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
                axs[0,col_counter].set_xlim(0, x_lim)
                axs[1,col_counter].set_xlim(0, x_lim)
            # else:
            #     axs[0,col_counter].set_xscale('log')
            #     axs[1,col_counter].set_xscale('log')
  
            ylim_for_model = ylims['elbo'].get(model_name, (None, None))
            axs[0,col_counter].set_ylim(*ylim_for_model)

            ylim_for_model = ylims['p_ll'].get(model_name, (None, None))
            axs[1,col_counter].set_ylim(*ylim_for_model)


            for inset in zoomed_insets:
                if inset.model_name == model_name and inset.K == K:
                    row = 0 if inset.metric_name == 'elbos' else 1
                    
                    axins = zoomed_inset_axes(axs[row,col_counter], 6.5, loc='center', bbox_to_anchor=(0.58,0.7), bbox_transform=axs[row,col_counter].transAxes, axes_kwargs={'aspect': 2.5})
                    
                    for method_name in inset.methods:
                        colour = colours_dict[method_name]

                        plot_method_K_lines(ax = axins, 
                                            model_results = results[model_name],
                                            method = method_name,
                                            K = K, 
                                            metric_name = inset.metric_name,
                                            num_lrs = num_lrs,
                                            colour = colour,
                                            x_axis_iters = x_axis_iters,
                                            smoothing_window = smoothing_window,
                                            x_lim_iters=x_lim,
                                            error_bars = error_bars,
                                            alpha_func = alpha_func)
                        
                    # axins.set_xlim(*inset.xlims)
                    # axins.set_ylim(*inset.ylims)

                    # remove y ticks
                    plt.yticks(visible=False)

                    mark_inset(axs[row,col_counter], axins, loc1=1, loc2=3, facecolor="none", edgecolor="none") 


            axs[1,col_counter].set_xlabel('Iterations' if x_axis_iters else 'Time (s)')

            axs[1,col_counter].legend()

            col_counter += 1

        axs[0,0].set_ylabel('ELBO')
        axs[1,0].set_ylabel('Predictive log-likelihood')

    fig.tight_layout()

    plt.savefig(f'plots/all_2row{filename_end}.png')
    if save_pdf:
        plt.savefig(f'plots/pdfs/all_2row{filename_end}.pdf')

def plot_all_2row_plus_global(model_names  = ALL_MODEL_NAMES,
                              Ks_to_plot   = 'largest',
                              num_lrs      = 1,
                              x_axis_iters = True,
                              x_lim        = 250,
                              x_lim_time   = None,
                              smoothing_window = 1,
                              error_bars   = False,
                              alpha_func   = DEFAULT_ALPHA_FUNC,
                              colours_dict = DEFAULT_COLOURS,
                              ylims        = {'elbo': {}, 'p_ll': {}},
                              force_lrs_per_model_method = None,
                              short_labels = True,
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
            auto_x_lim = [np.inf, np.inf]

            for j, method_name in enumerate(results[model_name].keys()):
                colour = colours_dict[method_name]

                if method_name in ['qem', 'rws', 'vi']:
                    alpha_func = lambda i, num_lrs : 1
                else:
                    alpha_func = lambda i, num_lrs: 0.5

                axs[0,col_counter].set_title(f'{model_name.upper()}\nK={K}')

                if force_lrs_per_model_method is not None:
                    force_lrs = force_lrs_per_model_method.get(model_name, {}).get(method_name, None)
                else:
                    force_lrs = None

                auto_x_lim0  =  plot_method_K_lines(ax = axs[0,col_counter], 
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
                                                    alpha_func = alpha_func,
                                                    force_lrs = force_lrs,
                                                    short_labels = short_labels)

                auto_x_lim1  =  plot_method_K_lines(ax = axs[1,col_counter], 
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
                                                    alpha_func = alpha_func,
                                                    force_lrs = force_lrs,
                                                    short_labels = short_labels)
                
                if auto_x_lim0 < auto_x_lim[0]:
                    auto_x_lim[0] = auto_x_lim0

                if auto_x_lim1 < auto_x_lim[1]:
                    auto_x_lim[1] = auto_x_lim1
                
            if x_axis_iters:
                    axs[0,col_counter].set_xlim(0, x_lim)
                    axs[1,col_counter].set_xlim(0, x_lim)
            elif x_lim_time is not None:
                if x_lim_time == 'auto':
                    axs[0, col_counter].set_xlim(0, auto_x_lim[0])
                    axs[1, col_counter].set_xlim(0, auto_x_lim[1])
                else:
                    axs[0, col_counter].set_xlim(0, x_lim_time[model_name])
                    axs[1, col_counter].set_xlim(0, x_lim_time[model_name])
                    
            # else:
            #     axs[0,col_counter].set_xscale('log')
                
            ylim_for_model = ylims['elbo'].get(model_name, (None, None))
            axs[0,col_counter].set_ylim(*ylim_for_model)

            ylim_for_model = ylims['p_ll'].get(model_name, (None, None))
            axs[1,col_counter].set_ylim(*ylim_for_model)

            axs[1,col_counter].set_xlabel('Iterations' if x_axis_iters else 'Time (s)')

            axs[1,0].legend()

            col_counter += 1

        axs[0,0].set_ylabel('ELBO')
        axs[1,0].set_ylabel('Predictive log-likelihood')

    fig.tight_layout()

    plt.savefig(f'plots/all_2row{filename_end}.png')
    if save_pdf:
        plt.savefig(f'plots/pdfs/all_2row{filename_end}.pdf')

def plot_avg_iter_time_per_K(model_names  = ALL_MODEL_NAMES,
                             colours_dict = DEFAULT_COLOURS,
                             save_pdf     = False,
                             filename_end = ""):
    
    results = {model_name: load_results(model_name) for model_name in model_names}

    fig, axs = plt.subplots(1, len(model_names), figsize=(len(model_names)*3, 3))

    for i, model_name in enumerate(model_names):
        # times = {K: {method_name: results[model_name][method_name][K]['iter_times'] for method_name in results[model_name].keys()} for K in results[model_name]['qem'].keys()}
        Ks = results[model_name]['qem'].keys()

        x = np.arange(len(Ks))  # the label locations
        width = 1/(1+len(results[model_name].keys()))  # the width of the bars
        multiplier = 0
        
        for j, method_name in enumerate(results[model_name].keys()):
            colour = colours_dict[method_name]

            avg_iter_times_per_K = [np.nanmean(results[model_name][method_name][K]['iter_times']) for K in Ks]
            
            offset = width * multiplier
            rects = axs[i].bar(x + offset, avg_iter_times_per_K, width, label=method_name.upper(), color=colour)
            # axs[i].bar_label(rects, padding=3)
            multiplier += 1

        axs[i].set_xlabel('K')
        axs[i].set_title(f'{model_name.upper()}')
        axs[i].set_xticks(x + width, results[model_name][method_name].keys())

    axs[0].legend()
    axs[0].set_ylabel('Average iteration time (s)')

    fig.tight_layout()

    plt.savefig(f'plots/avg_iter_time_per_K{filename_end}.png')
    if save_pdf:
        plt.savefig(f'plots/pdfs/avg_iter_time_per_K{filename_end}.pdf')

        

if __name__ == "__main__":
    # whether to ignore NaNs in the results or not
    # i.e. if True, then the results are averaged over all runs, even if some runs have NaNs (runs which failed after some number of iterations)
    #      if False, then the results are averaged over all runs until the first NaN is encountered (only reports results up to the first failure of any run)
    #                (this helps avoid weird leaps in the plots when some runs fail early and others don't)
    ignore_nans = False

    validation_iter_number = 125
    iteration_x_lim = 2*validation_iter_number

    for model_name in ALL_MODEL_NAMES:
        if model_name == 'occupancy':
            pass
            preprocess.get_best_results(model_name, validation_iter_number=validation_iter_number, method_names=['qem', 'rws', 'qem_nonmp'], ignore_nans=ignore_nans)
        else:
            preprocess.get_best_results(model_name, validation_iter_number=validation_iter_number, method_names=['qem', 'rws', 'vi', 'qem_nonmp'], ignore_nans=ignore_nans)

    best_Ks = {'bus_breakdown': [30], 'bus_breakdown_reparam': [30], 'chimpanzees': [30], 'movielens': [30], 'movielens_reparam': [30], 'occupancy': [30], 'radon': [30]}

    ylims = {'elbo': {'bus_breakdown': (-2500,  -1240),
                      'chimpanzees':   (-270,   -243),
                      'movielens':     (-3000,  -900),
                      'occupancy':     (-52000, -49300),
                      'radon':         (-16000, 0), #(-494,   -484)},
                      'bus_breakdown_reparam': (-2500,  -1240),
                      'movielens_reparam':     (-10000,  -900),},
             'p_ll': {'bus_breakdown': (-3000,  -1450),
                      'chimpanzees':   (-45,    -39),
                      'movielens':     (-1200,  -940),
                      'occupancy':     (-25800, -24700),
                      'radon':         (-250000000, 10000000),#(-170,   -120)},
                      'bus_breakdown_reparam': (-3000,  -1450),
                      'movielens_reparam':     (-2400,  -940),}
            }
    
    zoomed_insets = [Zoomed_Inset(model_name='occupancy', methods=['qem', 'rws'],   K=best_Ks['occupancy'][0], metric_name='p_lls', xlims=(None, None), ylims=(-24800, -24725)),]
                    #  Zoomed_Inset(model_name='radon', methods=['qem', 'rws', 'vi'], K=best_Ks['radon'][0],     metric_name='p_lls', xlims=(None, None), ylims=(-25000000, -1000000))]

    smoothing_window = 3

    plot_all_2row(Ks_to_plot=best_Ks,
                  num_lrs = 1,
                  filename_end = f"_K30_SMOOTH{smoothing_window}_TIME",
                  x_axis_iters = False, 
                  x_lim = iteration_x_lim, 
                  error_bars = True, 
                  save_pdf = True, 
                  ylims = ylims,
                  zoomed_insets = zoomed_insets,
                  smoothing_window=smoothing_window)
    
    plot_all_2row(Ks_to_plot=best_Ks,
                  num_lrs = 1, 
                  filename_end = f"_K30_SMOOTH{smoothing_window}_ITER", 
                  x_axis_iters = True, 
                  x_lim = iteration_x_lim, 
                  error_bars = True, 
                  save_pdf = True, 
                  ylims = ylims,
                  zoomed_insets = zoomed_insets,
                  smoothing_window=smoothing_window)
    
    plot_avg_iter_time_per_K(save_pdf=True)




    # ########## GLOBAL VS MP ##########
    # validation_iter_number = 50
    # iteration_x_lim = 250#2*validation_iter_number

    # smoothing_window = 1

    # model_names = ['bus_breakdown', 'chimpanzees', 'movielens', 'occupancy']

    # for model_name in model_names:
    #     if model_name == 'occupancy':
    #         preprocess.get_best_results(model_name, validation_iter_number=validation_iter_number, method_names=['qem', 'rws', 'global_rws'])
    #     else:
    #         preprocess.get_best_results(model_name, validation_iter_number=validation_iter_number, method_names=['qem', 'rws', 'vi', 'global_rws', 'global_vi'])

    # best_Ks = {'bus_breakdown': [10], 'chimpanzees': [10], 'movielens': [10], 'occupancy': [10], 'radon': [10]}
    # # ylims = {'elbo': {},
    # #          'p_ll': {}}

    # ylims = {'elbo': {'bus_breakdown': (-6000,  -1500),
    #                   'chimpanzees':   (-750,   -235),
    #                   'movielens':     (-9000,  -1000),
    #                   'occupancy':     (-150000, -49250),#(-51400, -49250)
    #                   'radon':         (-5000, 0)},#(-494,   -484)},
    #          'p_ll': {'bus_breakdown': (-6000,  -1800),
    #                   'chimpanzees':   (-100,    -39),
    #                   'movielens':     (-1250,  -950),
    #                   'occupancy':     (-100000, -24750),#(-25500, -24750),
    #                   'radon':         (-1000000, -100000)},#(-170,   -120)},
    #         }

    # force_lrs = {'bus_breakdown': {'rws': [0.1], 'vi': [0.1], 'global_rws': [0.1], 'global_vi': [0.1]},
    #                 'chimpanzees': {'rws': [0.1], 'vi': [0.1], 'global_rws': [0.1], 'global_vi': [0.1]},}
    #                 # 'movielens': {'rws': [0.1], 'vi': [0.1], 'global_rws': [0.1], 'global_vi': [0.1]},
    #                 # 'occupancy': {'rws': [0.1], 'vi': [0.1], 'global_rws': [0.1], 'global_vi': [0.1]},
    #                 # 'radon': {'rws': [0.1], 'vi': [0.1], 'global_rws': [0.1], 'global_vi': [0.1]}
    #                 # }
    
    # time_x_lim = {'bus_breakdown': 1, 'chimpanzees': 1, 'movielens': 1, 'occupancy': 1, 'radon': 1}
    # # time_x_lim = 'auto'

    # short_labels = True

    # plot_all_2row_plus_global(Ks_to_plot=best_Ks, model_names=model_names,
    #                num_lrs=1, filename_end="L_talk_TIME", x_axis_iters=False, error_bars=False, save_pdf=True, ylims=ylims, x_lim=iteration_x_lim, x_lim_time=time_x_lim, smoothing_window=smoothing_window, force_lrs_per_model_method = force_lrs, short_labels=short_labels)
    # plot_all_2row_plus_global(Ks_to_plot=best_Ks, model_names=model_names,
    #                num_lrs=1, filename_end="L_talk_ITER", x_axis_iters=True, error_bars=False, save_pdf=True, ylims=ylims, x_lim=iteration_x_lim, x_lim_time=time_x_lim, smoothing_window=smoothing_window, force_lrs_per_model_method = force_lrs, short_labels=short_labels)
    
    # # plot_avg_iter_time_per_K(save_pdf=True)
        
