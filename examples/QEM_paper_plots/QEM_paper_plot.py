import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import torch as t 
import preprocess

# ALL_MODEL_NAMES = ['bus_breakdown', 'chimpanzees', 'movielens', 'occupancy', 'radon']
# ALL_MODEL_NAMES = ['bus_breakdown', 'bus_breakdown_reparam', 'chimpanzees', 'movielens', 'movielens_reparam', 'occupancy', 'radon']
ALL_MODEL_NAMES = ['bus_breakdown', 'bus_breakdown_reparam', 'movielens', 'movielens_reparam',
                  'occupancy', 'occupancy_reparam', 'radon', 'radon_reparam', 'covid']

REPARAM_MODELS  = ['bus_breakdown_reparam', 'movielens_reparam', 'occupancy_reparam', 'radon_reparam']#, 'covid_reparam']

REPARAM_MODELS  = ['bus_breakdown_reparam', 'movielens_reparam', 'occupancy_reparam', 'radon_reparam']#, 'covid_reparam']

SHORT_LABEL_DICT = {'qem': 'QEM', 'rws': 'MP RWS', 'vi': 'MP VI', 'qem_nonmp': 'Global QEM', 'global_rws': 'Global RWS', 'global_vi': 'IWAE', 'HMC': 'HMC'}

DEFAULT_ALPHA_FUNC = lambda i, num_lrs: 1 if i == 0 else 1 - 0.5*i/(num_lrs-1)

DEFAULT_COLOURS = {'qem': '#e7298a', 'qem_nonmp' : '#7570b3',
                   'rws': '#1b9e77', 'global_rws': '#1b9e77', 
                   'vi' : '#d95f02', 'global_vi' : '#d95f02',
                   'HMC': '#000000'}

def load_results(model_name):
    with open(f'../experiments/results/{model_name}/best.pkl', 'rb') as f:
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
    def __init__(self, model_name, methods, K, metric_name, xlims, ylims, position='bottom', compare_reparams=False):
        self.model_name = model_name
        self.methods = methods
        self.K = K
        self.metric_name = metric_name
        self.xlims = xlims
        self.ylims = ylims
        self.position = position
        self.compare_reparams = compare_reparams

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
                        short_labels = True,
                        show_labels = True,
                        label_transform = lambda x: x,
                        HMC = False):
        
    if not HMC:
        metric     = model_results[method][K][metric_name]
        stderrs    = model_results[method][K][f'{metric_name[:-1]}_stderrs']
        iter_times = model_results[method][K]['iter_times'].cumsum(axis=1) 
        lrs = model_results[method][K]['lrs']

        if force_lrs is not None:
            # order the 0th (lr) dimension of metric, stderrs and iter_times to match the order of force_lrs
            lr_order = np.array([lrs.tolist().index(lr) for lr in force_lrs])
            metric = metric[lr_order, :]
            stderrs = stderrs[lr_order, :]
            iter_times = iter_times[lr_order, :]
            lrs = force_lrs
            num_lrs = len(lrs)

    else:
        metric     = model_results[method][metric_name]
        stderrs    = model_results[method][f'{metric_name[:-1]}_stderrs']
        iter_times = model_results[method]['iter_times'].cumsum(axis=0)
        lrs = []

        # add a dummy lr dimension 
        metric = metric[None, :]
        stderrs = stderrs[None, :]
        iter_times = iter_times[None, :]

    min_x_val = np.inf

    for i in range(num_lrs):
        if HMC or i < metric.shape[0]:
            smoothed_metric = smooth(metric[i, :], smoothing_window)

            alpha_val = alpha_func(i, num_lrs)

            xs = np.arange(smoothed_metric.shape[0]) if x_axis_iters else iter_times[i]

            if x_lim_iters is None:
                x_lim_iters = len(xs)

            if not show_labels:
                label = None
            elif short_labels:
                label = SHORT_LABEL_DICT[method]
            else:
                label=f'{method.upper()}: lr={lrs[i]}'

            label = label_transform(label)

            ax.plot(xs[:x_lim_iters], smoothed_metric[:x_lim_iters], label=label, color=colour, alpha=alpha_val)

            if error_bars:
                ax.fill_between(xs[:x_lim_iters], (smoothed_metric - stderrs[i, :])[:x_lim_iters], (smoothed_metric + stderrs[i, :])[:x_lim_iters], color=colour, alpha=0.2*alpha_val)

            if not HMC:  # don't count HMC when computing min_x_val
                if xs[x_lim_iters-1] < min_x_val:
                    min_x_val = xs[x_lim_iters-1]

    return min_x_val
    
def plot_all_2row(model_names   = ALL_MODEL_NAMES,
                  Ks_to_plot    = 'largest',
                  num_lrs       = 1,
                  x_axis_iters  = True,
                  x_lim         = 50,
                  smoothing_window = 1,
                  error_bars    = False,
                  alpha_func    = DEFAULT_ALPHA_FUNC,
                  colours_dict  = DEFAULT_COLOURS,
                  short_labels  = True,
                  ylims         = {'elbo': {}, 'p_ll': {}},
                  yscale        = 'linear',
                  zoomed_insets = [],
                  compare_reparams = False,
                  match_reparam_lrs = False,
                  save_pdf      = False,
                  filename_end  = ""):
    
    results = {model_name: load_results(model_name) for model_name in model_names}

    if compare_reparams and all([model_name.endswith('_reparam') for model_name in model_names]):
        # load results for the original parameterisation
        original_param_model_names = [model_name[:-8] for model_name in model_names]
        original_param_results = {model_name+"_reparam": load_results(model_name) for model_name in original_param_model_names}

        if match_reparam_lrs:
            # for each model, reorder the reparam results' lr-dimension to match the original param results
            for model_name in model_names:
                for method_name in results[model_name].keys():
                    if method_name != 'HMC':
                        Ks = list(results[model_name][method_name].keys())
                        for K in Ks:
                            original_lrs = original_param_results[model_name][method_name][K]['lrs']
                            reparam_lrs  = results[model_name][method_name][K]['lrs']

                            reorder_idxs = [reparam_lrs.tolist().index(lr) for lr in original_lrs]

                            for metric_name in ['elbos', 'p_lls']:
                                results[model_name][method_name][K][metric_name] = results[model_name][method_name][K][metric_name][reorder_idxs, :]
                                results[model_name][method_name][K][f'{metric_name[:-1]}_stderrs'] = results[model_name][method_name][K][f'{metric_name[:-1]}_stderrs'][reorder_idxs, :]
                            results[model_name][method_name][K]['iter_times'] = results[model_name][method_name][K]['iter_times'][reorder_idxs, :]
                            results[model_name][method_name][K]['iter_times_stderrs'] = results[model_name][method_name][K]['iter_times_stderrs'][reorder_idxs, :]

                            results[model_name][method_name][K]['lrs'] = [reparam_lrs[i] for i in reorder_idxs]

    if Ks_to_plot == 'largest':
        Ks_to_plot = {model_name: [max(results[model_name]['qem'].keys())] for model_name in model_names}
    elif Ks_to_plot == 'smallest':
        Ks_to_plot = {model_name: [min(results[model_name]['qem'].keys())] for model_name in model_names}
    elif Ks_to_plot == 'all':
        Ks_to_plot = {model_name: list(results[model_name]['qem'].keys()) for model_name in model_names}

    num_cols = sum([len(Ks_to_plot[model_name]) for model_name in model_names])

    fig, axs = plt.subplots(2, num_cols, figsize=(max(num_cols*3, 12), 7), sharex=x_axis_iters)

    col_counter = 0
    for i, model_name in enumerate(model_names):
        for k, K in enumerate(Ks_to_plot[model_name]):

            axs[0,col_counter].set_title(f'{model_name.upper().replace("_", " ")}\nK={K}')

            for j, method_name in enumerate(results[model_name].keys()):
                colour = colours_dict[method_name]

                if method_name != 'HMC':
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
                                        alpha_func = alpha_func,
                                        short_labels=short_labels)
                
                if not (method_name == 'HMC' and x_axis_iters):
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
                                        alpha_func = alpha_func,
                                        short_labels=short_labels,
                                        HMC = method_name == 'HMC')
                    
                if compare_reparams:
                    if method_name != 'HMC':
                        plot_method_K_lines(ax = axs[0,col_counter], 
                                            model_results = original_param_results[model_name],
                                            method = method_name,
                                            K = K,
                                            metric_name = 'elbos',
                                            num_lrs = num_lrs,
                                            colour = colour,
                                            x_axis_iters = x_axis_iters,
                                            smoothing_window = smoothing_window,
                                            x_lim_iters=x_lim,
                                            error_bars = error_bars,
                                            alpha_func = lambda *args: 0.5*alpha_func(*args),
                                            short_labels=short_labels,
                                            show_labels=False,)
    
                    if not (method_name == 'HMC' and x_axis_iters):
                        plot_method_K_lines(ax = axs[1,col_counter], 
                                            model_results = original_param_results[model_name],
                                            method = method_name,
                                            K = K,
                                            metric_name = 'p_lls',
                                            num_lrs = num_lrs,
                                            colour = colour,
                                            x_axis_iters = x_axis_iters,
                                            smoothing_window = smoothing_window,
                                            x_lim_iters=x_lim,
                                            error_bars = error_bars,
                                            alpha_func = lambda *args: 0.5*alpha_func(*args),
                                            HMC = method_name == 'HMC',
                                            short_labels=short_labels,
                                            show_labels=False,)
                

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


            axs[0,col_counter].set_yscale(yscale)
            axs[1,col_counter].set_yscale(yscale)


            for inset in zoomed_insets:
                if inset.model_name == model_name and inset.K == K:
                    row = 0 if inset.metric_name == 'elbos' else 1

                    bbox_to_anchor = (0.615,0.705) if inset.position == 'top' else (0.60,0.35)

                    axins = inset_axes(axs[row,col_counter], 1.5,1, loc='center', bbox_to_anchor=bbox_to_anchor, bbox_transform=axs[row,col_counter].transAxes)
                    
                    for method_name in inset.methods:
                        colour = colours_dict[method_name]

                        plot_method_K_lines(ax = axins, 
                                            model_results = results[model_name],
                                            method = method_name,
                                            K = K if method_name != 'HMC' else -1, 
                                            metric_name = inset.metric_name,
                                            num_lrs = num_lrs,
                                            colour = colour,
                                            x_axis_iters = x_axis_iters,
                                            smoothing_window = smoothing_window,
                                            x_lim_iters=x_lim,
                                            error_bars = error_bars,
                                            alpha_func = alpha_func,
                                            short_labels=short_labels,
                                            show_labels=False,),

                        if inset.compare_reparams:
                            if method_name != 'HMC' and inset.metric_name == 'elbos':
                                plot_method_K_lines(ax = axins, 
                                                    model_results = original_param_results[model_name],
                                                    method = method_name,
                                                    K = K,
                                                    metric_name = 'elbos',
                                                    num_lrs = num_lrs,
                                                    colour = colour,
                                                    x_axis_iters = x_axis_iters,
                                                    smoothing_window = smoothing_window,
                                                    x_lim_iters=x_lim,
                                                    error_bars = error_bars,
                                                    alpha_func = lambda *args: 0.5*alpha_func(*args),
                                                    short_labels=short_labels,
                                                    show_labels=False,)
            
                            if not (method_name == 'HMC' and x_axis_iters) and inset.metric_name == 'p_lls':
                                plot_method_K_lines(ax = axins, 
                                                    model_results = original_param_results[model_name],
                                                    method = method_name,
                                                    K = K,
                                                    metric_name = 'p_lls',
                                                    num_lrs = num_lrs,
                                                    colour = colour,
                                                    x_axis_iters = x_axis_iters,
                                                    smoothing_window = smoothing_window,
                                                    x_lim_iters=x_lim,
                                                    error_bars = error_bars,
                                                    alpha_func = lambda *args: 0.5*alpha_func(*args),
                                                    HMC = method_name == 'HMC',
                                                    short_labels=short_labels,
                                                    show_labels=False,)
                        
                    # axins.set_xlim(*inset.xlims)
                    axins.set_ylim(*inset.ylims)

                    # remove y ticks
                    # axins.yaxis.set_tick_params(labelleft=False)
                    # axins.set_yticks([])

                    mark_inset(axs[row,col_counter], axins, loc1=1, loc2=3, facecolor="none", edgecolor="none") 


            axs[1,col_counter].set_xlabel('Iterations' if x_axis_iters else 'Time (s)')

            # axs[1,col_counter].legend()

            col_counter += 1

    if short_labels:
        axs[1,0].legend()
    else:
        for i in range(col_counter):
            axs[1,i].legend()

    axs[0,0].set_ylabel('ELBO')
    axs[1,0].set_ylabel('Predictive log-likelihood')

    fig.tight_layout()

    plt.savefig(f'plots/{filename_end}.png')
    if save_pdf:
        plt.savefig(f'plots/pdfs/{filename_end}.pdf')

def plot_elbo_only_reparams(model_names   = ALL_MODEL_NAMES,
            Ks_to_plot    = 'largest',
            num_lrs       = 1,
            x_axis_iters  = True,
            x_lim         = 50,
            smoothing_window = 1,
            error_bars    = False,
            alpha_func    = DEFAULT_ALPHA_FUNC,
            colours_dict  = DEFAULT_COLOURS,
            short_labels  = True,
            ylims         = {'elbo': {}},
            yscale        = 'linear',
            match_reparam_lrs = True,
            shift_legends_y = [],
            save_pdf      = True,
            filename_end  = ""):
    
    results = {model_name: load_results(model_name) for model_name in model_names}

    # load results for the original parameterisation
    original_param_model_names = [model_name[:-8] for model_name in model_names]
    original_param_results = {model_name+"_reparam": load_results(model_name) for model_name in original_param_model_names}

    model_method_Ks_with_altered_lrs = []

    if match_reparam_lrs:
        # for each model, reorder the reparam results' lr-dimension to match the original param results
        for model_name in model_names:
            for method_name in results[model_name].keys():
                if method_name != 'HMC':
                    Ks = list(results[model_name][method_name].keys())
                    for K in Ks:
                        original_lrs = original_param_results[model_name][method_name][K]['lrs']
                        reparam_lrs  = results[model_name][method_name][K]['lrs']

                        reorder_idxs = [reparam_lrs.tolist().index(lr) for lr in original_lrs]

                        for metric_name in ['elbos', 'p_lls']:
                            results[model_name][method_name][K][metric_name] = results[model_name][method_name][K][metric_name][reorder_idxs, :]
                            results[model_name][method_name][K][f'{metric_name[:-1]}_stderrs'] = results[model_name][method_name][K][f'{metric_name[:-1]}_stderrs'][reorder_idxs, :]
                        results[model_name][method_name][K]['iter_times'] = results[model_name][method_name][K]['iter_times'][reorder_idxs, :]
                        results[model_name][method_name][K]['iter_times_stderrs'] = results[model_name][method_name][K]['iter_times_stderrs'][reorder_idxs, :]

                        results[model_name][method_name][K]['lrs'] = [reparam_lrs[i] for i in reorder_idxs]

                        if reorder_idxs[:num_lrs] != list(range(num_lrs)):
                            model_method_Ks_with_altered_lrs.append((model_name, method_name, K))

    if Ks_to_plot == 'largest':
        Ks_to_plot = {model_name: [max(results[model_name]['qem'].keys())] for model_name in model_names}
    elif Ks_to_plot == 'smallest':
        Ks_to_plot = {model_name: [min(results[model_name]['qem'].keys())] for model_name in model_names}
    elif Ks_to_plot == 'all':
        Ks_to_plot = {model_name: list(results[model_name]['qem'].keys()) for model_name in model_names}

    num_cols = sum([len(Ks_to_plot[model_name]) for model_name in model_names])

    fig, axs = plt.subplots(3, num_cols, figsize=(max(num_cols*3, 12), 10), sharex=x_axis_iters)

    col_counter = 0
    for i, model_name in enumerate(model_names):
        for k, K in enumerate(Ks_to_plot[model_name]):

            axs[0,col_counter].set_title(f'{model_name.upper().replace("_", " ")}\nK={K}')

            for j, method_name in enumerate(results[model_name].keys()):
                colour = colours_dict[method_name]

                num_reparam_lrs_to_plot = 1
                if match_reparam_lrs and (model_name, method_name, K) in model_method_Ks_with_altered_lrs:
                    num_reparam_lrs_to_plot = 2

                # plot original param results
                plot_method_K_lines(ax = axs[j,col_counter], 
                                    model_results = original_param_results[model_name],
                                    method = method_name,
                                    K = K,
                                    metric_name = 'elbos',
                                    num_lrs = num_lrs,
                                    colour = 'black',#colour,
                                    x_axis_iters = x_axis_iters,
                                    smoothing_window = smoothing_window,
                                    x_lim_iters=x_lim,
                                    error_bars = error_bars,
                                    # alpha_func = lambda i, n: 0.5*alpha_func(i + num_reparam_lrs_to_plot, n + num_reparam_lrs_to_plot),
                                    alpha_func = lambda *args: 1,
                                    short_labels=short_labels,
                                    show_labels=True,
                                    label_transform=lambda x: f'{x.split(" ")[1]} (original)')
                
                # plot reparam results
                plot_method_K_lines(ax = axs[j,col_counter], 
                                    model_results = results[model_name],
                                    method = method_name,
                                    K = K,
                                    metric_name = 'elbos',
                                    num_lrs = num_reparam_lrs_to_plot,
                                    colour = colour,
                                    x_axis_iters = x_axis_iters,
                                    smoothing_window = smoothing_window,
                                    x_lim_iters=x_lim,
                                    error_bars = error_bars,
                                    alpha_func = alpha_func,
                                    short_labels=short_labels,
                                    label_transform=lambda x: f'{x.split(" ")[1]}')

                
                if x_axis_iters:
                    axs[j,col_counter].set_xlim(0, x_lim)
                # else:
                #     axs[0,col_counter].set_xscale('log')
  
                ylim_for_model = ylims['elbo'].get(model_name, {}).get(method_name, (None, None))
                axs[j,col_counter].set_ylim(*ylim_for_model)

                axs[j,col_counter].set_yscale(yscale)

                axs[j, 0].set_ylabel('ELBO')

                # if (model_name, method_name) in shift_legends_y:
                #     axs[j,col_counter].legend(title=SHORT_LABEL_DICT[method_name], bbox_to_anchor=(1.05, 1))
                # else:
                #     axs[j,col_counter].legend(title=SHORT_LABEL_DICT[method_name])#, loc='lower right')
                leg = axs[j,col_counter].legend(title=SHORT_LABEL_DICT[method_name])#, loc='lower right')

                plt.draw() # Draw the figure so you can find the positon of the legend. 

                if (model_name, method_name) in shift_legends_y:
                    
                    # Get the bounding box of the original legend
                    bb = leg.get_bbox_to_anchor().transformed(axs[j,col_counter].transAxes.inverted())

                    # Change to location of the legend. 
                    yOffset = 0.22 if model_name == 'radon_reparam' else 0.16
                    bb.y0 += yOffset
                    bb.y1 += yOffset
                    leg.set_bbox_to_anchor(bb, transform = axs[j,col_counter].transAxes)


            axs[-1,col_counter].set_xlabel('Iterations' if x_axis_iters else 'Time (s)')

            # axs[1,col_counter].legend()

            col_counter += 1

    # delete occupancy vi axes
    axs[2,model_names.index("occupancy_reparam")].axis('off')

    fig.tight_layout()

    plt.savefig(f'plots/reparams{filename_end}.png')
    if save_pdf:
        plt.savefig(f'plots/pdfs/reparams{filename_end}.pdf')

def plot_HMC_vs_QEM_pll(model_names   = ALL_MODEL_NAMES,
                        Ks_to_plot    = 'largest',
                        num_lrs       = 1,
                        x_axis_iters  = True,
                        x_lim         = 50,
                        smoothing_window = 1,
                        error_bars    = False,
                        alpha_func    = DEFAULT_ALPHA_FUNC,
                        colours_dict  = DEFAULT_COLOURS,
                        short_labels  = True,
                        ylims         = {'p_ll': {}},
                        yscale        = 'linear',
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

    fig, axs = plt.subplots(1, num_cols, figsize=(max(num_cols*3, 12), 7), sharex=x_axis_iters)

    col_counter = 0
    for i, model_name in enumerate(model_names):
        for k, K in enumerate(Ks_to_plot[model_name]):

            axs[col_counter].set_title(f'{model_name.upper().replace("_", " ")}\nK={K}')

            for j, method_name in enumerate(results[model_name].keys()):
                colour = colours_dict[method_name]
                
                plot_method_K_lines(ax = axs[col_counter], 
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
                                    short_labels=short_labels,
                                    HMC = method_name == 'HMC')

            if x_axis_iters:
                axs[col_counter].set_xlim(0, x_lim)
            # else:
            #     axs[col_counter].set_xscale('log')

            ylim_for_model = ylims['p_ll'].get(model_name, (None, None))
            axs[col_counter].set_ylim(*ylim_for_model)

            axs[col_counter].set_yscale(yscale)

            axs[col_counter].set_xlabel('Iterations' if x_axis_iters else 'Time (s)')

            # axs[1,col_counter].legend()

            col_counter += 1

    if short_labels:
        axs[0].legend()
    else:
        for i in range(col_counter):
            axs[1,i].legend()

    axs[0].set_ylabel('Predictive log-likelihood')

    fig.tight_layout()

    plt.savefig(f'plots/{filename_end}.png')
    if save_pdf:
        plt.savefig(f'plots/pdfs/{filename_end}.pdf')

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

                axs[0,col_counter].set_title(f'{model_name.upper().replace("_", " ")}\nK={K}')

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

        methods = list(results[model_name].keys())
        width = 1/(1+len(methods) if 'HMC' not in methods else len(methods)-1)  # the width of the bars
        multiplier = 0

        nan_Ks = [K for K in Ks if np.isnan(np.nanmean(results[model_name]['qem'][K]['iter_times']))]
        valid_Ks = [K for K in Ks if K not in nan_Ks]
        x = np.arange(len(valid_Ks))  # the label locations
        
        for j, method_name in enumerate(results[model_name].keys()):
            if method_name == 'HMC':
                continue
            colour = colours_dict[method_name]

            avg_iter_times_per_K = [np.nanmean(results[model_name][method_name][K]['iter_times']) for K in valid_Ks]

            # std_errs_per_K = [np.nanstd(results[model_name][method_name][K]['iter_times'])/np.sqrt(np.prod(results[model_name][method_name][K]['iter_times'].shape)) for K in valid_Ks]
            std_devs_per_K = [np.nanstd(results[model_name][method_name][K]['iter_times']) for K in valid_Ks]
            
            offset = width * multiplier
            
            rects = axs[i].bar(x + offset, avg_iter_times_per_K, width=width, yerr=std_devs_per_K, label=SHORT_LABEL_DICT[method_name], color=colour)
            # axs[i].bar_label(rects, padding=3)
            multiplier += 1

        axs[i].set_xlabel('K')
        axs[i].set_title(f'{model_name.upper().replace("_", " ")}')
        axs[i].set_xticks(x + width, valid_Ks)

    axs[0].legend(loc='upper left')
    axs[0].set_ylabel('Average iteration time (s)')

    fig.tight_layout()

    plt.savefig(f'plots/avg_iter_time_per_K{filename_end}.png')
    if save_pdf:
        plt.savefig(f'plots/pdfs/avg_iter_time_per_K{filename_end}.pdf')

        

if __name__ == "__main__":

    # bool to control whether to rerun the preprocessing or not
    run_preprocessing = True

    # bools to control which plots are generated
    make_time_per_iteration_plots = True
    make_elbo_p_ll_plots = True
    make_reparam_elbo_plots = True
    make_HMC_vs_QEM_pll_plots = False

    # whether to ignore NaNs in the results or not
    # i.e. if True, then the results are averaged over all runs, even if some runs have NaNs (runs which failed after some number of iterations)
    #      if False, then the results are averaged over all runs until the first NaN is encountered (only reports results up to the first failure of any run)
    #                (this helps avoid weird leaps in the plots when some runs fail early and others don't)
    ignore_nans = False

    # validation_iter_number = 225
    # iteration_x_lim = 250 

    validation_iter_number = 125
    iteration_x_lim = 2*validation_iter_number

    plot_HMC = False
    plot_global_QEM = False

    basic_methods = ['qem', 'rws']
    # if plot_global_QEM:
    #     basic_methods.append('qem_nonmp')

    # model2method = {'bus_breakdown': ['qem', 'rws', 'vi', 'qem_nonmp'],
    #                 'bus_breakdown_reparam': ['qem', 'rws', 'vi', 'qem_nonmp'],
    #                 'chimpanzees': ['qem', 'rws', 'vi', 'qem_nonmp'],
    #                 'movielens': ['qem', 'rws', 'vi', 'qem_nonmp'],
    #                 'movielens_reparam': ['qem', 'rws', 'vi', 'qem_nonmp'],
    #                 'occupancy': ['qem', 'rws', 'qem_nonmp'],
    #                 'radon': ['qem', 'rws', 'vi', 'qem_nonmp'],
    #                 'covid': ['qem', 'rws', 'vi', 'qem_nonmp']}

    model2method = {model_name: [*basic_methods] for model_name in ALL_MODEL_NAMES}
    
    for model_name, methods in model2method.items():
        if 'occupancy' not in model_name:
            methods.append('vi')
        if plot_HMC and model_name not in ('occupancy', 'radon', 'occupancy_reparam', 'radon_reparam'):
            methods.append('HMC')
        if plot_global_QEM:
            methods.append('qem_nonmp')
            
    if run_preprocessing:
        for model_name in ALL_MODEL_NAMES:
            preprocess.get_best_results(model_name, validation_iter_number=validation_iter_number, method_names=model2method[model_name], ignore_nans=ignore_nans)
    
    sub_model_collections = {'standard': ['bus_breakdown', 'movielens', 'occupancy', 'radon', 'covid'],
                             'standard_no_covid': ['bus_breakdown', 'movielens', 'occupancy', 'radon'],
                             'reparams': ['bus_breakdown_reparam', 'movielens_reparam', 'radon_reparam']}
    
    ##################### TIME-PER-ITERATION PLOTS #####################
    if make_time_per_iteration_plots:
        plot_avg_iter_time_per_K(save_pdf=True)
        plot_avg_iter_time_per_K(save_pdf=True, model_names=sub_model_collections['standard'], filename_end='_standardONLY')
        plot_avg_iter_time_per_K(save_pdf=True, model_names=sub_model_collections['standard_no_covid'], filename_end='_standard_no_covidONLY')


    #####################     ELBO/P_LL PLOTS      #####################
    best_Ks = {'bus_breakdown': [30], 'bus_breakdown_reparam': [30], 'chimpanzees': [30], 'movielens': [30], 'movielens_reparam': [30],
               'occupancy': [30], 'occupancy_reparam': [30], 'radon': [30], 'radon_reparam': [30], 'covid': [10]}
    smoothing_window = 8
    short_labels = True

    # YLIMS FOCUSING ON END OF TRAINING (WILL OFTEN IGNORE GLOBAL QEM) #
    ylims ={'elbo': {'bus_breakdown': (-500,  -400),
                     'chimpanzees':   (-255,   -244),
                     'movielens':     (-1060,  -985),
                     'occupancy':     (-49300, -49050),
                     'radon':         (-290,-276),
                     'bus_breakdown_reparam': (-500,  -400),
                     'movielens_reparam':     (-1060,  -985),
                     'occupancy_reparam':     (-49300, -49050),
                     'radon_reparam':         (-310,-276),
                     'covid': (-1400000, -500000)},
            'p_ll': {'bus_breakdown': (-500,  -425),
                     'chimpanzees':   (-45,    -39.5),
                     'movielens':     (-965,  -943),
                     'occupancy':     (-24600, -24550),
                     'radon':         (-600, -450),
                     'bus_breakdown_reparam':  (-500,  -425),
                     'movielens_reparam':     (-965,  -943),
                     'occupancy_reparam':     (-24590, -24550),
                     'radon_reparam':         (-600, -450),
                     'covid': (-30000000, 2000000)}
           }

    if make_elbo_p_ll_plots:
        for x_axis_iters in [True, False]:
            x_axis_str = 'ITER' if x_axis_iters else 'TIME'
            plot_all_2row(model_names=ALL_MODEL_NAMES,
                        Ks_to_plot=best_Ks,
                        num_lrs = 2,
                        filename_end = f"K30_SMOOTH{smoothing_window}_{x_axis_str}",
                        x_axis_iters = x_axis_iters, 
                        x_lim = iteration_x_lim, 
                        error_bars = True, 
                        save_pdf = True, 
                        ylims = ylims,
                        short_labels=short_labels,
                        smoothing_window=smoothing_window)
            
            for name, sub_models in sub_model_collections.items():
                plot_all_2row(model_names=sub_models,
                            Ks_to_plot=best_Ks,
                            num_lrs = 1,
                            filename_end = f"{name}ONLY_K30_SMOOTH{smoothing_window}_{x_axis_str}",
                            x_axis_iters = x_axis_iters, 
                            x_lim = iteration_x_lim, 
                            error_bars = name != 'reparams', 
                            compare_reparams = name == 'reparams',
                            save_pdf = True, 
                            ylims = ylims,
                            short_labels=short_labels,
                            smoothing_window=smoothing_window)
                
        # do a reparam iteration plot with extended extended y-axes (and ZoomedInsets) to show the full range of values
        # (i.e. show VI and RWS performing terribly)
        ylims_extended = {'elbo': {**ylims['elbo']}, 'p_ll': {**ylims['p_ll']}}
        ylims_extended['elbo']['movielens_reparam'] = (-40000, -50)
        ylims_extended['p_ll']['movielens_reparam'] = (-4000, -750)
        ylims_extended['elbo']['radon_reparam'] = (-1e8, 5e6)
        ylims_extended['p_ll']['radon_reparam'] = (-40000, 1000)

        zoomed_insets = [Zoomed_Inset(model_name='movielens_reparam', compare_reparams=True, methods=['qem', 'rws', 'vi'],  K=best_Ks['movielens_reparam'][0], metric_name='elbos', xlims=(None, None), position='bottom', ylims=(-1050,  -985)),  
                        Zoomed_Inset(model_name='movielens_reparam', compare_reparams=True, methods=['qem', 'rws', 'vi'],  K=best_Ks['movielens_reparam'][0], metric_name='p_lls', xlims=(None, None), position='top',    ylims=ylims['p_ll']['movielens']),      
                        Zoomed_Inset(model_name='radon_reparam',     compare_reparams=True, methods=['qem', 'rws', 'vi'],  K=best_Ks['radon_reparam'][0],     metric_name='elbos', xlims=(None, None), position='top',    ylims=ylims['elbo']['radon']),
                        Zoomed_Inset(model_name='radon_reparam',     compare_reparams=True, methods=['qem', 'rws', 'vi'],  K=best_Ks['radon_reparam'][0],     metric_name='p_lls', xlims=(None, None), position='top',    ylims=(-790, -525)),
                        ]

        plot_all_2row(model_names=sub_model_collections['reparams'],
                    Ks_to_plot=best_Ks,
                    num_lrs = 1,
                    filename_end = f"reparamsONLY_K30_SMOOTH{smoothing_window}_ITER_EXTENDED_y",
                    x_axis_iters = True, 
                    x_lim = iteration_x_lim, 
                    error_bars = False, 
                    compare_reparams = True,
                    save_pdf = True, 
                    ylims = ylims_extended,
                    yscale = 'linear',
                    zoomed_insets=zoomed_insets,
                    short_labels=short_labels,
                    smoothing_window=smoothing_window)

    ###################  3x3 ELBO-ONLY REPARAM PLOT  ####################
    reparam_ylims = {'elbo': {'bus_breakdown_reparam': {'rws': (-1000, -350),     'vi': (-1000, -400)},
                              'movielens_reparam':     {'rws': (-20000, 500),    'vi': (-200000, 10000)},
                              'occupancy_reparam':     {'rws': (-100000, -45000), 'vi': (None, None)},
                              'radon_reparam':         {'rws': (-10000, 500),       'vi': (-900, -250)}}
                    }

    # set all qem ylims to be the same as in the standard plots
    for model_name in ['bus_breakdown_reparam', 'movielens_reparam', 'occupancy_reparam', 'radon_reparam']:
        reparam_ylims['elbo'][model_name]['qem'] = ylims['elbo'][model_name]
    
    #except radon needs to be a bit higher
    reparam_ylims['elbo']['radon_reparam']['qem'] = (-285, -275)

    # note which legends to shift up to avoid overlap
    shift_legends_y = [('movielens_reparam', 'vi'), ('radon_reparam', 'vi'),]

    if make_reparam_elbo_plots:
        plot_elbo_only_reparams(model_names=REPARAM_MODELS,
                                Ks_to_plot=best_Ks,
                                num_lrs = 1,
                                filename_end = f"",
                                x_axis_iters = True, 
                                x_lim = iteration_x_lim, 
                                error_bars = True, 
                                match_reparam_lrs=True,
                                save_pdf = True, 
                                ylims = reparam_ylims,
                                short_labels=False,
                                smoothing_window=smoothing_window,
                                shift_legends_y=shift_legends_y)

    #####################  QEM vs. HMC P_LL PLOTS  #####################
    # if make_HMC_vs_QEM_pll_plots:
    #     for model_name in ALL_MODEL_NAMES:
    #         model2method[model_name].append('HMC')
    #         preprocess.get_best_results(model_name, validation_iter_number=validation_iter_number, method_names=model2method[model_name], ignore_nans=ignore_nans)

    #     for x_axis_iters in [True, False]:
    #         x_axis_str = 'ITER' if x_axis_iters else 'TIME'

    #         plot_HMC_vs_QEM_pll(model_names=ALL_MODEL_NAMES,
    #                             Ks_to_plot=best_Ks,
    #                             num_lrs = 2,
    #                             filename_end = f"HMC_vs_QEM_K30_SMOOTH{smoothing_window}_{x_axis_str}",
    #                             x_axis_iters = x_axis_iters, 
    #                             x_lim = iteration_x_lim, 
    #                             error_bars = True, 
    #                             save_pdf = True, 
    #                             ylims = ylims,
    #                             short_labels=short_labels,
    #                             smoothing_window=smoothing_window)
            
    #         for name, sub_models in sub_model_collections.items():
    #             if name != 'reparams':
    #                 plot_HMC_vs_QEM_pll(model_names=sub_models,
    #                                     Ks_to_plot=best_Ks,
    #                                     num_lrs = 1,
    #                                     filename_end = f"{name}ONLY_HMC_vs_QEM_K30_SMOOTH{smoothing_window}_{x_axis_str}",
    #                                     x_axis_iters = x_axis_iters, 
    #                                     x_lim = iteration_x_lim, 
    #                                     error_bars = True, 
    #                                     save_pdf = True, 
    #                                     ylims = ylims,
    #                                     short_labels=short_labels,
    #                                     smoothing_window=smoothing_window)

    
    # PLOTS W/ ZOOMED INSETS TO ALWAYS SHOW GLOBAL QEM BUT KEEP DETAIL #

    # ylims_for_zoomed_insets = {'elbo': {'bus_breakdown': (-2500,  -1240),
    #                                     'chimpanzees':   (-270,   -243),
    #                                     'movielens':     (-3000,  -900),
    #                                     'occupancy':     (-54000, -49300),
    #                                     'radon':         (-16000, 0), #(-494,   -484)},
    #                                     'bus_breakdown_reparam': (-2500,  -1240),
    #                                     'movielens_reparam':     (-10000,  -900),},
    #                             'p_ll': {'bus_breakdown': (-3000,  -1450),
    #                                     'chimpanzees':   (-45,    -39),
    #                                     'movielens':     (-1250,  -940),
    #                                     'occupancy':     (-26000, -24700),
    #                                     'radon':         (-450000000, 10000000),#(-170,   -120)},
    #                                     'bus_breakdown_reparam': (-3000,  -1450),
    #                                     'movielens_reparam':     (-2400,  -940),}
    #                             }
    
    # _zoomed_insets = [Zoomed_Inset(model_name='occupancy', methods=['qem', 'rws'],       K=best_Ks['occupancy'][0], metric_name='p_lls', xlims=(None, None), ylims=(-24800,    -24725)),    
    #                  Zoomed_Inset(model_name='radon',     methods=['qem', 'rws', 'vi'], K=best_Ks['radon'][0],     metric_name='p_lls', xlims=(None, None), ylims=(-25000000, -1000000))]

    # for zoom in [True, False]:
    #     zoomed_insets = _zoomed_insets if zoom else []

    #     plot_all_2row(Ks_to_plot=best_Ks,
    #                   num_lrs = 1,
    #                   filename_end = f"_K30_SMOOTH{smoothing_window}_TIME{'_nozoom' if not zoom else ''}",
    #                   x_axis_iters = False, 
    #                   x_lim = iteration_x_lim, 
    #                   error_bars = True, 
    #                   save_pdf = True, 
    #                   ylims = ylims_for_zoomed_insets,
    #                   zoomed_insets = zoomed_insets,
    #                   smoothing_window=smoothing_window)
        
    #     plot_all_2row(Ks_to_plot=best_Ks,
    #                   num_lrs = 1, 
    #                   filename_end = f"_K30_SMOOTH{smoothing_window}_ITER{'_nozoom' if not zoom else ''}", 
    #                   x_axis_iters = True, 
    #                   x_lim = iteration_x_lim, 
    #                   error_bars = True, 
    #                   save_pdf = True, 
    #                   ylims = ylims_for_zoomed_insets,
    #                   zoomed_insets = zoomed_insets,
    #                   smoothing_window=smoothing_window)

    #     for name, sub_models in sub_model_collections.items():

    #         if zoom and set(sub_models).intersection(set([inset.model_name for inset in zoomed_insets])) == set():
    #             continue
            
    #         plot_all_2row(model_names=sub_models,
    #                       Ks_to_plot=best_Ks,
    #                       num_lrs = 1,
    #                       filename_end = f"_{name}ONLY_K30_SMOOTH{smoothing_window}_TIME{'_nozoom' if not zoom else ''}",
    #                       x_axis_iters = False, 
    #                       x_lim = iteration_x_lim, 
    #                       error_bars = True, 
    #                       save_pdf = True, 
    #                       ylims = ylims_for_zoomed_insets,
    #                       zoomed_insets = zoomed_insets,
    #                       smoothing_window=smoothing_window)
            
    #         plot_all_2row(model_names=sub_models,
    #                       Ks_to_plot=best_Ks,
    #                       num_lrs = 1, 
    #                       filename_end = f"_{name}ONLY_K30_SMOOTH{smoothing_window}_ITER{'_nozoom' if not zoom else ''}", 
    #                       x_axis_iters = True, 
    #                       x_lim = iteration_x_lim, 
    #                       error_bars = True, 
    #                       save_pdf = True, 
    #                       ylims = ylims_for_zoomed_insets,
    #                       zoomed_insets = zoomed_insets,
    #                       smoothing_window=smoothing_window)



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
        
