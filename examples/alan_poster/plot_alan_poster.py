import pickle
import matplotlib.pyplot as plt
import torch as t
import numpy as np

DEFAULT_COLOURS = {'qem':      '#4daf4a',
                   'vi': '#e41a1c',
                   'vi_nonmp':        '#ff7f00',
                   'rws':     '#984ea3', 
                   'rws_nonmp':    '#377eb8'}
                #    'rws':       '#ffff33',
                #    'HMC':       '#FF69B4'}


DEFAULT_MODEL_METHOD_LRS_TO_IGNORE = {'covid': {'vi': [0.1]}}

DEFAULT_ALPHA_FUNC = lambda lr: {0.3: 1, 0.1: 0.83, 0.03: 0.67, 0.01: 0.5}.get(lr, 0.1)
DEFAULT_ALPHA_FUNC = lambda i, num_lrs: 1 if i == 0 or num_lrs <= 1 else max(1 - 0.5*i/(num_lrs-1), 0.5)
DEFAULT_ALPHA_FUNC = lambda lr: 1

def smooth(x, window):
    # result = np.convolve(x, np.ones(window)/window, mode='valid')
    
    result = np.zeros_like(x)

    result[0] = x[0]

    for i in range(1,len(x)):
        if x[i] != np.nan:
            result[i] = x[max(i-window, 0):i].mean()
        # result[i,:] = np.nanmean(x[max(i-window, 0):i,:], 1)

    return result

def dict_copy(d):
    output = {}
    for key in d:
        if isinstance(d[key], dict):
            output[key] = dict_copy(d[key])
        else:
            output[key] = d[key]
    return output

def load_results(model_name, method_name, dataset_seed=0, results_subfolder=''):
    with open(f'../{model_name}/results/{results_subfolder}{method_name}{dataset_seed}.pkl', 'rb') as f:
        return pickle.load(f)

def plot_some_model_metric_Ks(models_metrics_Ks_methods : list, window_size=25, dataset_seeds=[0], results_subfolder='', _model_method_lrs_to_ignore=DEFAULT_MODEL_METHOD_LRS_TO_IGNORE, x_lim_iters=10, x_time=True, log_x_time=False, log_y=False, xlims={}, ylims={}, auto_xlim=False, save_pdf=True, only_best_lr=True, elbo_validation_iter=100, alpha_function=DEFAULT_ALPHA_FUNC, filename=""):

    all_models = [model for model, _, _, _ in models_metrics_Ks_methods]
    all_metrics = [metric for _, metric, _, _ in models_metrics_Ks_methods]
    all_Ks = [K for _, _, K, _ in models_metrics_Ks_methods]
    all_method_names = [methods for _, _, _, methods, in models_metrics_Ks_methods]

    fig, ax = plt.subplots(2 if x_time else 1, len(models_metrics_Ks_methods), figsize=(4*len(models_metrics_Ks_methods), 8 if x_time else 4))
    
    for m, model_name in enumerate(all_models):
        metric = all_metrics[m]
        K = all_Ks[m]
        method_names = all_method_names[m]
    
        all_results = {}

        model_method_lrs_to_ignore = dict_copy(_model_method_lrs_to_ignore)

        if model_method_lrs_to_ignore.get(model_name, None) is None:
            model_method_lrs_to_ignore[model_name] = {}
            
        for method in method_names:
            all_results[method] = {}
            # if not (model_name == 'occupancy' and (method not in ['mpis', 'global_is'] or method[:3] != 'rws')):
            if model_name == 'occupancy' and (method[:2] == 'vi' or method[:3] == 'HMC'):
                continue

            for dataset_seed in dataset_seeds:
                temp = load_results(model_name, method, dataset_seed, results_subfolder=results_subfolder)
                if len(all_results[method]) == 0:
                    all_results[method] = temp 
                else:
                    for key in ['Ks', 'lrs', 'num_runs', 'num_iters']:
                        assert np.all(all_results[method][key] == temp[key])
                    
                    for key in ['elbos', 'p_lls', 'iter_times']:
                        all_results[method][key] += temp[key]

            for key in ['elbos', 'p_lls', 'iter_times']:
                all_results[method][key] /= len(dataset_seeds)

            K_idx = all_results[method]['Ks'].index(K)

            if only_best_lr:
                valid_lr_idxs = np.array([i for i, lr in enumerate(all_results[method]['lrs']) if lr not in model_method_lrs_to_ignore[model_name].get(method, [])])
                best_lr = np.array(all_results[method]['lrs'])[valid_lr_idxs][int(np.argmax(all_results[method]['elbos'][K_idx].mean(-1)[valid_lr_idxs, elbo_validation_iter].numpy()))]
                model_method_lrs_to_ignore[model_name][method] = [lr for lr in all_results[method]['lrs'] if lr != best_lr]

        shortest_time = np.inf

        for i, method in enumerate(method_names):
            colour = DEFAULT_COLOURS[method]

            method_name = method.upper()
            if method_name == 'VI_NONMP':
                method_name = 'Global VI'
            if method_name == 'RWS_NONMP':
                method_name = 'Global RWS'
            if method_name in ['VI', 'RWS']:
                method_name = 'MP ' + method_name

            if model_name == 'occupancy' and (method[:2] == 'vi' or method[:3] == 'HMC'):
                continue

            else:
                for lr_idx, lr in enumerate(all_results[method]['lrs']):
                    # alpha = alpha_function(lr_idx, len(all_results[method]['lrs']) - len(model_method_lrs_to_ignore[model_name].get(method, [])))
                    alpha = alpha_function(lr)

                    label = f"{method_name} lr={lr}" if not only_best_lr else f"{method_name}"

                    if lr in model_method_lrs_to_ignore[model_name].get(method, []):
                        # ax[0,m].plot([],[], label=label, color=colour, alpha=alpha)
                        # ax[1,m].plot([],[], label=label, color=colour, alpha=alpha)
                        continue
                    
                    xs = all_results[method]['iter_times'][K_idx, lr_idx].mean(-1).cumsum(0)
                    ys = all_results[method][metric][K_idx, lr_idx].mean(-1)

                    ys = smooth(ys, window_size)

                    if xs[x_lim_iters-1] < shortest_time:
                        shortest_time = xs[x_lim_iters-1]

                    if x_time:
                        ax[0,m].plot(xs[:x_lim_iters], ys[:x_lim_iters], label=label, color=colour, alpha=alpha)
                        ax[1,m].plot(np.arange(x_lim_iters), ys[:x_lim_iters], label=label, color=colour, alpha=alpha)
                    else:
                        ax[m].plot(xs[:x_lim_iters], ys[:x_lim_iters], label=label, color=colour, alpha=alpha)

        if x_time:
            ax[0,m].set_title(f"{model_name.upper()} [K={K}]")

            ax[0,m].set_xlabel('Time (s)')
            ax[1,m].set_xlabel('Iteration')

            metric_name = 'ELBO' if metric == 'elbos' else 'Predictive Log-Likelihood'
            ax[0,m].set_ylabel(metric_name)
            ax[1,m].set_ylabel(metric_name)

            if log_x_time:
                ax[0,m].set_xscale('log')
            
            if auto_xlim:
                ax[0,m].set_xlim(right=shortest_time)
                    
            else:
                ax[0,m].set_xlim(*xlims.get(metric, {}).get(model_name, {}).get(K, (None, None)))

            ylims_ = ylims.get(metric, {}).get(model_name, {}).get(K, (None, None))
            ax[0,m].set_ylim(*ylims_)
            ax[1,m].set_ylim(*ylims_)

            if log_y:
                ax[0,m].set_yscale('log')
                ax[1,m].set_yscale('log')

            ax[1,m].legend()

        else:
            ax[m].set_title(f"{model_name.upper()} [K={K}]")

            ax[m].set_xlabel('Time (s)')

            metric_name = 'ELBO' if metric == 'elbos' else 'Predictive Log-Likelihood'
            ax[m].set_ylabel(metric_name)

            if log_x_time:
                ax[m].set_xscale('log')
            
            if auto_xlim:
                ax[m].set_xlim(right=shortest_time)
                    
            else:
                ax[m].set_xlim(*xlims.get(metric, {}).get(model_name, {}).get(K, (None, None)))

            ylims_ = ylims.get(metric, {}).get(model_name, {}).get(K, (None, None))
            ax[m].set_ylim(*ylims_)

            if log_y:
                ax[m].set_yscale('log')

            ax[m].legend()

    fig.tight_layout()

    if filename == "":
        filename = f"many_"
        for model, metric, K, _ in models_metrics_Ks_methods:
            filename += f"_{model}{K}{metric}"
        filename += f"{'_log_x' if log_x_time else ''}_{window_size}"

    plt.savefig(f"plots/{filename}.png")
    if save_pdf:
        plt.savefig(f"plots/pdfs/{filename}.pdf")

    plt.close()

if __name__ == '__main__':
    
    elbo_ylims_per_K = {'movielens':     {3: (-4000, -950), 10: (-3000, -950), 30: (-2000, -950)},
                        'bus_breakdown': {3: (-6000, None), 10: (-3300, None), 30: (-2750, None)},
                        'occupancy':     {3: (-70000, None), 5: (-55000, None), 10: (-50000, None)},
                        'radon':         {3: (-800, -450), 10: (-580, -480), 30: (-500, -480)},
                        'chimpanzees':   {5: (-500, -240),  15: (-500, -240)},
                        'covid':         {3: (-5000000, 1000), 10: (-200000, 0), 30: (None, None)}}

    pll_ylims_per_K  = {'movielens':     {3: (-1150, -940), 10: (-1100, -940), 30: (None, -940)}, #30: (-1060, -940)},
                        'bus_breakdown': {3: (-7000, None), 10: (-3500, None), 30: (-2800, -1750)},
                        'occupancy':     {3: (-35000, None), 5: (-28000, None), 10: (-24900, None)},
                        'radon':         {3: (-300, None), 10: (-155, -130), 30: (-150, -130)},
                        'chimpanzees':   {5: (-50, -40),    15: (-50, -40)}}

    ylims = {'elbos': elbo_ylims_per_K, 'p_lls': pll_ylims_per_K}
    # for K in [3, 10]:
    #     plot('covid', Ks_to_plot=[K], method_lrs_to_ignore={'qem': [0.001, 0.0001], 'rws': [0.001, 0.0001], 'vi': [0.001, 0.0001]},
    #          elbo_ylims=elbo_ylims_per_K['covid'][K])

    for x_time in [True, False]:
        for log_y in [True, False]:
            plot_some_model_metric_Ks((('covid', 'elbos', 3, ['qem', 'rws', 'vi']),
                                    ('movielens', 'p_lls', 30, ['qem', 'rws', 'vi', 'rws_nonmp', 'vi_nonmp'])),        
                                    results_subfolder='alan_poster/', 
                                    x_lim_iters=1000,
                                    x_time = x_time,
                                    log_x_time=False, 
                                    ylims=ylims,
                                    auto_xlim=False, 
                                    only_best_lr=True, 
                                    elbo_validation_iter=100, 
                                    alpha_function=DEFAULT_ALPHA_FUNC, 
                                    filename=f"alan_poster{'_logy' if log_y else ''}{'_1row' if not x_time else ''}"
            )
