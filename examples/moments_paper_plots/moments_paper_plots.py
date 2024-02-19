import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch as t 

ALL_MODEL_NAMES = ['bus_breakdown', 'chimpanzees', 'movielens', 'occupancy']

DEFAULT_COLOURS = {'mpis':      '#e41a1c',
                   'global_is': '#377eb8',
                   'vi':        '#4daf4a',
                   'vi10K':     '#984ea3', 
                   'rws10K':    '#ff7f00',
                   'rws':       '#ffff33',
                   'HMC':       '#FF69B4'}

DEFAULT_ALPHA_FUNC = lambda i, num_lrs: 1 if i == 0 or num_lrs <= 1 else max(1 - 0.5*i/(num_lrs-1), 0.5)

DEFAULT_ALPHA_FUNC = lambda lr: {0.3: 1, 0.1: 0.83, 0.03: 0.67, 0.01: 0.5}.get(lr, 0.1)


# These learning rates are ignored for each model and method because the corresponding
# results are not reliable (e.g. the method failed or encountered numerical issues)
DEFAULT_MODEL_METHOD_LRS_TO_IGNORE = {'bus_breakdown': {'vi': [], 'rws10K': []},
                                      'chimpanzees': {'vi': [], 'rws10K': [0.3]},
                                      'movielens': {'vi': [0.3], 'rws10K': [0.3,0.1], 'vi10K': [0.3]},
                                      'occupancy': {'rws10K': [0.3,0.1]}}

# DEFAULT_MODEL_METHOD_LRS_TO_IGNORE = {model_name: {} for model_name in ALL_MODEL_NAMES}
# DEFAULT_MODEL_METHOD_LRS_TO_IGNORE = {model_name: {'rws': [0.1, 0.3]} for model_name in ALL_MODEL_NAMES}


DEFAULT_ELBO_VALIDATION_ITER = 200

def dict_copy(d):
    output = {}
    for key in d:
        if isinstance(d[key], dict):
            output[key] = dict_copy(d[key])
        else:
            output[key] = d[key]
    return output

def load_results(model_name, method_name, fake_data, dataset_seed=0):
    with open(f'../{model_name}/results/moments/{method_name}{dataset_seed}{"_FAKE_DATA" if fake_data else ""}.pkl', 'rb') as f:
        return pickle.load(f)
    
def choose_MSEs(results, latent_name='all'):
    if latent_name == 'all':
        total = None
        for key in results['MSEs']:
            if total is None:
                total = results['MSEs'][key]
            else:
                total += results['MSEs'][key]
        return total
    else:
        return results['MSEs'][latent_name]

def remove_failed_Ks(results, method):
    # remove any K-values that are all zeros (i.e. the method failed to sample anything)
    valid_K_idx = results[method]['MSEs'].cpu() != 0
    results[method]['Ks'] = [str(K) for k, K in enumerate(results[method]['Ks']) if valid_K_idx[k]]
    for key in results[method]:
        if key in ['elbos', 'p_lls']:
            results[method][key] = results[method][key][results[method][key].sum(1) != 0]
        elif key in ['MSEs', 'MSEs_fake']:
            results[method][key] = results[method][key][results[method][key] != 0]
        elif key == 'times':
            for subkey in results[method][key]:
                results[method][key][subkey] = results[method][key][subkey][results[method][key][subkey].sum(1) != 0]

    return results[method]

    
def plot_IS_per_K_one_model(model_name, save_pdf=False, scatter=False, MSE_latent = 'all', filename=""):
    all_results = {}
    for method in ['mpis', 'global_is']:
        all_results[method] = load_results(model_name, method, False)
        all_results[method]['MSEs'] = choose_MSEs(all_results[method], MSE_latent)
        all_results[method]['MSEs_fake'] = choose_MSEs(load_results(model_name, method, True), MSE_latent)

        all_results[method] = remove_failed_Ks(all_results, method)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    for i, method in enumerate(['mpis', 'global_is']):
        colour = DEFAULT_COLOURS[method]
        for j, key in enumerate(['elbos', 'p_lls', 'MSEs', 'MSEs_fake']):
            if key in ['elbos', 'p_lls']:
                if scatter:
                    ax[j].scatter(all_results[method]['Ks'], all_results[method][key].mean(1), label=method.upper(), marker='x', color=colour)
                else:
                    ax[j].plot(all_results[method]['Ks'], all_results[method][key].mean(1), label=method.upper(), color=colour)
                ax[j].errorbar(all_results[method]['Ks'], all_results[method][key].mean(1), yerr=all_results[method][key].std(1)/np.sqrt(all_results[method]['num_runs']), fmt='x', color=colour)
            else:
                # if scatter:
                ax[j].scatter(all_results[method]['Ks'], all_results[method][key].cpu(), label=method.upper(), marker='x', color=colour)
                if not scatter:
                    ax[j].plot(all_results[method]['Ks'], all_results[method][key].cpu(), label=method.upper(), color=colour)
                ax[j].errorbar(all_results[method]['Ks'], all_results[method][key].cpu(), yerr=0, label=method.upper(), color=colour)
            ax[j].set_xlabel('K')
            ax[j].tick_params(axis='x', rotation=45)

    fig.suptitle("IS Comparison for " + model_name.upper())
    ax[0].set_ylabel('ELBO')
    ax[1].set_ylabel('Predictive Log-Likelihood')
    ax[2].set_ylabel(f'Total Variance{" of variable " + MSE_latent if MSE_latent != "all" else ""}')
    ax[3].set_ylabel(f'MSE{" of variable " + MSE_latent if MSE_latent != "all" else ""}')
    ax[0].legend()

    plt.xticks(rotation=70)

    fig.tight_layout()

    if filename == "":
        filename = f"{model_name}_IS_per_K{'_MSE_' + MSE_latent if MSE_latent != 'all' else ''}"

    plt.savefig(f"plots/{filename}.png")
    if save_pdf:
        plt.savefig(f"plots/pdfs/{filename}.pdf")

    plt.close()

def plots_IS_per_K_all_models(save_pdf=False, scatter=False, x_axis_time=False, MSE_latent = 'all', Ks_to_ignore=[1], filename=""):
    all_results = {model_name: {} for model_name in ALL_MODEL_NAMES}
    for model_name in ALL_MODEL_NAMES:
        for method in ['mpis', 'global_is']:
            all_results[model_name][method] = load_results(model_name, method, False)
            all_results[model_name][method]['MSEs'] = choose_MSEs(all_results[model_name][method], MSE_latent)
            all_results[model_name][method]['MSEs_fake'] = choose_MSEs(load_results(model_name, method, True), MSE_latent)

            all_results[model_name][method] = remove_failed_Ks(all_results[model_name], method)

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))

    for i, model_name in enumerate(ALL_MODEL_NAMES):
        for j, method in enumerate(['mpis', 'global_is']):
            colour = DEFAULT_COLOURS[method]
            for k, key in enumerate(['elbos', 'p_lls']):
                # if key in ['elbos', 'p_lls']:
                Ks_to_ignore_idxs = [all_results[model_name][method]['Ks'].index(str(K)) for K in Ks_to_ignore if str(K) in all_results[model_name][method]['Ks']]
                Ks_to_keep = np.array([i for i in range(len(all_results[model_name][method]['Ks'])) if i not in Ks_to_ignore_idxs])
                if x_axis_time:
                    xs = all_results[model_name][method]['times'][key if k == 0 else 'p_ll'].mean(-1)
                else:
                    xs = all_results[model_name][method]['Ks']
                    xs = [int(K) for K in xs]
                    ax[k,i].set_xticks(xs)
                    ax[k,i].set_xscale('log')
                xs = [xs[i] for i in range(len(xs)) if i not in Ks_to_ignore_idxs]
                if scatter:
                    ax[k, i].scatter(xs, all_results[model_name][method][key].mean(1)[Ks_to_keep], label=method.upper(), marker='x', color=colour)
                else:
                    # breakpoint()
                    ax[k, i].plot(xs, all_results[model_name][method][key].mean(1)[Ks_to_keep], label=method.upper(), color=colour)
                ax[k, i].errorbar(xs, all_results[model_name][method][key].mean(1)[Ks_to_keep], yerr=all_results[model_name][method][key].std(1)[Ks_to_keep]/np.sqrt(all_results[model_name][method]['num_runs']), fmt='x', color=colour)
                
                if x_axis_time:
                    ax[k, i].set_xlabel('Time (s)')
                else:
                    ax[k, i].set_xlabel('K')
                    ax[k, i].tick_params(axis='x', rotation=45)

        ax[0, i].set_title(model_name.upper())

    ax[0,0].set_ylabel('ELBO')
    ax[1,0].set_ylabel('Predictive Log-Likelihood')
    ax[0,0].legend()

    if x_axis_time:
        ax[0,0].set_ylim(bottom=-6100)
        ax[0,1].set_ylim(bottom=-750)

        ax[1,0].set_ylim(bottom=-6100)
        ax[1,1].set_ylim(bottom=-150)


    fig.tight_layout()

    # fig.suptitle("IS Comparison for All Models")

    if filename == "":
        filename = f"all_IS_per_K{'_TIME' if x_axis_time else ''}"

    plt.savefig(f"plots/{filename}.png")
    if save_pdf:
        plt.savefig(f"plots/pdfs/{filename}.pdf")

    plt.close()

def plot_iterative_vs_IS_all_K_one_model(model_name, iterative_methods = ['vi', 'rws', 'HMC'], save_pdf=False, x_lim_iters=10, log_x=False, model_method_lrs_to_ignore=DEFAULT_MODEL_METHOD_LRS_TO_IGNORE, alpha_function=DEFAULT_ALPHA_FUNC, MSE_latent = 'all', xlims={}, ylims={}, auto_xlim=False, filename=""):
    all_results = {}
    for method in ['mpis', 'global_is'] + iterative_methods:
        all_results[method] = load_results(model_name, method, False)
        all_results[method]['MSEs'] = choose_MSEs(all_results[method], MSE_latent)

        fake_results = load_results(model_name, method, True)
        all_results[method]['MSEs_fake'] = choose_MSEs(fake_results, MSE_latent)
        all_results[method]['times']['moments_fake'] = fake_results['times']['moments']

        if method in ['mpis', 'global_is']:
            all_results[method] = remove_failed_Ks(all_results, method)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    for i, method in enumerate(['mpis', 'global_is']):
        colour = DEFAULT_COLOURS[method]
        ax[0].plot(all_results[method]['times']['elbos'].mean(-1), all_results[method]['elbos'].mean(1), label=method.upper(), color=colour)
        ax[1].plot(all_results[method]['times']['p_ll'].mean(-1), all_results[method]['p_lls'].mean(1), label=method.upper(), color=colour)
        ax[2].plot(all_results[method]['times']['moments'].mean(-1), all_results[method]['MSEs'].cpu(), label=method.upper(), color=colour)
        ax[3].plot(all_results[method]['times']['moments_fake'].mean(-1), all_results[method]['MSEs_fake'].cpu(), label=method.upper(), color=colour)

    for i, method in enumerate(iterative_methods):
        colour = DEFAULT_COLOURS[method]

        method_name = method.upper()
        if method_name == 'VI10K':
            method_name = 'IWAE'
        if method_name == 'RWS10K':
            method_name = 'RWS'

        for lr_idx, lr in enumerate(all_results[method]['lrs']):
            if lr in model_method_lrs_to_ignore[model_name].get(method, []):
                continue
            # alpha = alpha_function(lr_idx, len(all_results[method]['lrs']) - len(model_method_lrs_to_ignore[model_name].get(method, [])))
            alpha = alpha_function(lr)

            ax[0].plot(all_results[method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['elbos'][lr_idx].mean(-1)[:x_lim_iters], label=f"{method_name} lr={lr}", color=colour, alpha=alpha)
            ax[1].plot(all_results[method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['p_lls'][lr_idx].mean(-1)[:x_lim_iters], label=f"{method_name} lr={lr}", color=colour, alpha=alpha)
            ax[2].plot(all_results[method]['times']['moments'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['MSEs'][lr_idx].cpu()[:x_lim_iters], label=f"{method_name} lr={lr}", color=colour, alpha=alpha)
            ax[3].plot(all_results[method]['times']['moments_fake'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['MSEs_fake'][lr_idx].cpu()[:x_lim_iters], label=f"{method_name} lr={lr}", color=colour, alpha=alpha)

    for j in range(4):
        ax[j].set_xlabel('Time (s)')
        # ax[j].tick_params(axis='x', rotation=45)

        if log_x:
            ax[j].set_xscale('log')

    fig.suptitle("Iterative vs IS for " + model_name.upper())
    ax[0].set_ylabel('ELBO')
    ax[1].set_ylabel('Predictive Log-Likelihood')
    ax[2].set_ylabel(f'Total Variance{" of variable " + MSE_latent if MSE_latent != "all" else ""}')
    ax[3].set_ylabel(f'MSE')

    ax[0].set_xlim(*xlims.get(model_name, {}).get('elbos', (None, None)))
    ax[1].set_xlim(*xlims.get(model_name, {}).get('p_lls', (None, None)))
    ax[2].set_xlim(*xlims.get(model_name, {}).get('vars',  (None, None)))
    ax[3].set_xlim(*xlims.get(model_name, {}).get('MSEs',  (None, None)))

    ax[0].set_ylim(*ylims.get(model_name, {}).get('elbos', (None, None)))
    ax[1].set_ylim(*ylims.get(model_name, {}).get('p_lls', (None, None)))
    ax[2].set_ylim(*ylims.get(model_name, {}).get('vars',  (None, None)))
    ax[3].set_ylim(*ylims.get(model_name, {}).get('MSEs',  (None, None)))

    ax[-1].legend()

    fig.tight_layout()

    if filename == "":
        filename = f"{model_name}_iterative_vs_IS_all_K{'_log_x' if log_x else ''}{'_MSE_' + MSE_latent if MSE_latent != 'all' else ''}"

    plt.savefig(f"plots/{filename}.png")
    if save_pdf:
        plt.savefig(f"plots/pdfs/{filename}.pdf")

    plt.close()

def plot_iterative_vs_IS_single_K_one_model(model_name, iterative_methods = ['vi', 'rws', 'HMC'], mpis_K = 10, global_is_K = 1000, save_pdf=False, x_lim_iters=10, log_x=False, model_method_lrs_to_ignore=DEFAULT_MODEL_METHOD_LRS_TO_IGNORE, only_best_lr=False, elbo_validation_iter=DEFAULT_ELBO_VALIDATION_ITER, alpha_function=DEFAULT_ALPHA_FUNC, MSE_latent = 'all', xlims={}, ylims={}, auto_xlim=False, filename=""):
    all_results = {}

    # only_best_lr = model_method_lrs_to_ignore == 'all_but_best'
    # if only_best_lr:
    #     model_method_lrs_to_ignore = {model_name: {}}

    model_method_lrs_to_ignore = dict_copy(model_method_lrs_to_ignore)
        
    for method in ['mpis', 'global_is'] + iterative_methods:
        all_results[method] = load_results(model_name, method, False)
        all_results[method]['MSEs'] = choose_MSEs(all_results[method], MSE_latent)

        fake_results = load_results(model_name, method, True)
        all_results[method]['MSEs_fake'] = choose_MSEs(fake_results, MSE_latent)
        all_results[method]['times']['moments_fake'] = fake_results['times']['moments']

        if method in ['mpis', 'global_is']:
            all_results[method] = remove_failed_Ks(all_results, method)

        if only_best_lr and method in iterative_methods and method != 'HMC':
            valid_lr_idxs = np.array([i for i, lr in enumerate(all_results[method]['lrs']) if lr not in model_method_lrs_to_ignore[model_name].get(method, [])])
            best_lr = all_results[method]['lrs'][int(np.argmax(all_results[method]['elbos'].mean(-1)[valid_lr_idxs, elbo_validation_iter].numpy()))]
            model_method_lrs_to_ignore[model_name][method] = [lr for lr in all_results[method]['lrs'] if lr != best_lr]

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    final_xs = {'elbos': [], 'p_lls': [], 'vars': [], 'MSEs': []}

    for i, method in enumerate(['mpis', 'global_is']):
        colour = DEFAULT_COLOURS[method]

        Ks = all_results[method]['Ks']
        K_idx = Ks.index(str(mpis_K) if method == 'mpis' else str(global_is_K))

        ax[0].scatter(all_results[method]['times']['elbos'].mean(-1)[K_idx], all_results[method]['elbos'].mean(1)[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)
        ax[1].scatter(all_results[method]['times']['p_ll'].mean(-1)[K_idx], all_results[method]['p_lls'].mean(1)[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)
        ax[2].scatter(all_results[method]['times']['moments'].mean(-1)[K_idx], all_results[method]['MSEs'].cpu()[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)
        ax[3].scatter(all_results[method]['times']['moments_fake'].mean(-1)[K_idx], all_results[method]['MSEs_fake'].cpu()[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)

    for i, method in enumerate(iterative_methods):
        colour = DEFAULT_COLOURS[method]

        method_name = method.upper()
        if method_name == 'VI10K':
            method_name = 'IWAE'
        if method_name == 'RWS10K':
            method_name = 'RWS'

        if method == 'HMC':
            # NOTE: times are already cumulative for HMC (no need for cumsum(0))
            ax[1].plot(all_results[method]['times']['p_ll'].mean(-1)[:x_lim_iters], all_results[method]['p_lls'].mean(-1)[:x_lim_iters], label=f"{method_name}", color=colour, alpha=alpha)
            ax[2].plot(all_results[method]['times']['moments'].mean(-1)[:x_lim_iters], all_results[method]['MSEs'][:x_lim_iters], label=f"{method_name}", color=colour, alpha=alpha)
            ax[3].plot(all_results[method]['times']['moments_fake'].mean(-1)[:x_lim_iters], all_results[method]['MSEs_fake'][:x_lim_iters], label=f"{method_name}", color=colour, alpha=alpha)

            final_xs['p_lls'].append(all_results[method]['times']['p_ll'].mean(-1).cumsum(0)[:x_lim_iters][-1])
            final_xs['vars'].append(all_results[method]['times']['moments'].mean(-1).cumsum(0)[:x_lim_iters][-1])
            final_xs['MSEs'].append(all_results[method]['times']['moments_fake'].mean(-1).cumsum(0)[:x_lim_iters][-1])

        else:
            for lr_idx, lr in enumerate(all_results[method]['lrs']):
                if lr in model_method_lrs_to_ignore[model_name].get(method, []):
                    continue
                # alpha = alpha_function(lr_idx, len(all_results[method]['lrs']) - len(model_method_lrs_to_ignore[model_name].get(method, [])))
                alpha = alpha_function(lr)

                label = f"{method_name} lr={lr}" if not only_best_lr else f"{method_name}"

                ax[0].plot(all_results[method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['elbos'][lr_idx].mean(-1)[:x_lim_iters], label=label, color=colour, alpha=alpha)
                ax[1].plot(all_results[method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['p_lls'][lr_idx].mean(-1)[:x_lim_iters], label=label, color=colour, alpha=alpha)
                ax[2].plot(all_results[method]['times']['moments'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['MSEs'][lr_idx].cpu()[:x_lim_iters], label=label, color=colour, alpha=alpha)
                ax[3].plot(all_results[method]['times']['moments_fake'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['MSEs_fake'][lr_idx].cpu()[:x_lim_iters], label=label, color=colour, alpha=alpha)

                final_xs['elbos'].append(all_results[method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])
                final_xs['p_lls'].append(all_results[method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])
                final_xs['vars'].append(all_results[method]['times']['moments'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])
                final_xs['MSEs'].append(all_results[method]['times']['moments_fake'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])

    for j in range(4):
        ax[j].set_xlabel('Time (s)')
        # ax[j].tick_params(axis='x', rotation=45)

        if log_x:
            ax[j].set_xscale('log')

    if auto_xlim:
        ax[0].set_xlim(right=min(final_xs['elbos']))
        ax[1].set_xlim(right=min(final_xs['p_lls']))
        ax[2].set_xlim(right=min(final_xs['vars']))
        ax[3].set_xlim(right=min(final_xs['MSEs']))

    fig.suptitle("Iterative vs IS for " + model_name.upper())
    ax[0].set_ylabel('ELBO')
    ax[1].set_ylabel('Predictive Log-Likelihood')
    ax[2].set_ylabel(f'Total Variance{" of variable " + MSE_latent if MSE_latent != "all" else ""}')
    ax[3].set_ylabel(f'MSE{" of variable " + MSE_latent if MSE_latent != "all" else ""}')

    ax[2].set_yscale('log')
    ax[3].set_yscale('log')

    ax[0].set_xlim(*xlims.get(model_name, {}).get('elbos', (None, None)))
    ax[1].set_xlim(*xlims.get(model_name, {}).get('p_lls', (None, None)))
    ax[2].set_xlim(*xlims.get(model_name, {}).get('vars',  (None, None)))
    ax[3].set_xlim(*xlims.get(model_name, {}).get('MSEs',  (None, None)))

    ax[0].set_ylim(*ylims.get(model_name, {}).get('elbos', (None, None)))
    ax[1].set_ylim(*ylims.get(model_name, {}).get('p_lls', (None, None)))
    ax[2].set_ylim(*ylims.get(model_name, {}).get('vars',  (None, None)))
    ax[3].set_ylim(*ylims.get(model_name, {}).get('MSEs',  (None, None)))

    ax[-1].legend()

    fig.tight_layout()

    if filename == "":
        filename = f"{model_name}_iterative_vs_IS_K{mpis_K}-{global_is_K}{'_log_x' if log_x else ''}{'_MSE_' + MSE_latent if MSE_latent != 'all' else ''}"

    plt.savefig(f"plots/{filename}.png")
    if save_pdf:
        plt.savefig(f"plots/pdfs/{filename}.pdf")

    plt.close()

def plot_iterative_vs_IS_single_K_one_model_per_row(all_models = ALL_MODEL_NAMES, iterative_methods = ['vi', 'rws', 'HMC'], mpis_K = 10, global_is_K = 1000, save_pdf=False, x_lim_iters=10, log_x=False, _model_method_lrs_to_ignore=DEFAULT_MODEL_METHOD_LRS_TO_IGNORE, only_best_lr=False, elbo_validation_iter=DEFAULT_ELBO_VALIDATION_ITER, alpha_function=DEFAULT_ALPHA_FUNC, MSE_latent = 'all', xlims={}, ylims={}, auto_xlim=False, filename=""):
    fig, ax = plt.subplots(len(all_models), 4, figsize=(16, 4*len(all_models)))

    
    for m, model_name in enumerate(all_models):
    
        all_results = {}

        # only_best_lr = model_method_lrs_to_ignore == 'all_but_best'
        # if only_best_lr:
        #     model_method_lrs_to_ignore = {model_name: {}}

        model_method_lrs_to_ignore = dict_copy(_model_method_lrs_to_ignore)
            
        for method in ['mpis', 'global_is'] + iterative_methods:
            # if not (model_name == 'occupancy' and (method not in ['mpis', 'global_is'] or method[:3] != 'rws')):
            if model_name == 'occupancy' and (method[:2] == 'vi' or method[:3] == 'HMC'):
                continue
            all_results[method] = load_results(model_name, method, False)
            all_results[method]['MSEs'] = choose_MSEs(all_results[method], MSE_latent)

            fake_results = load_results(model_name, method, True)
            all_results[method]['MSEs_fake'] = choose_MSEs(fake_results, MSE_latent)
            all_results[method]['times']['moments_fake'] = fake_results['times']['moments']

            if method in ['mpis', 'global_is']:
                all_results[method] = remove_failed_Ks(all_results, method)

            if only_best_lr and method in iterative_methods and method != 'HMC':
                valid_lr_idxs = np.array([i for i, lr in enumerate(all_results[method]['lrs']) if lr not in model_method_lrs_to_ignore[model_name].get(method, [])])
                best_lr = np.array(all_results[method]['lrs'])[valid_lr_idxs][int(np.argmax(all_results[method]['elbos'].mean(-1)[valid_lr_idxs, elbo_validation_iter].numpy()))]
                model_method_lrs_to_ignore[model_name][method] = [lr for lr in all_results[method]['lrs'] if lr != best_lr]

        final_xs = {'elbos': [], 'p_lls': [], 'vars': [], 'MSEs': []}

        for i, method in enumerate(['mpis', 'global_is']):
            colour = DEFAULT_COLOURS[method]

            Ks = all_results[method]['Ks']
            K_idx = Ks.index(str(mpis_K) if method == 'mpis' else str(global_is_K))

            ax[m,0].scatter(all_results[method]['times']['elbos'].mean(-1)[K_idx], all_results[method]['elbos'].mean(1)[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)
            ax[m,1].scatter(all_results[method]['times']['p_ll'].mean(-1)[K_idx], all_results[method]['p_lls'].mean(1)[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)
            ax[m,2].scatter(all_results[method]['times']['moments'].mean(-1)[K_idx], all_results[method]['MSEs'].cpu()[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)
            ax[m,3].scatter(all_results[method]['times']['moments_fake'].mean(-1)[K_idx], all_results[method]['MSEs_fake'].cpu()[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)

        for i, method in enumerate(iterative_methods):
            colour = DEFAULT_COLOURS[method]

            method_name = method.upper()
            if method_name == 'VI10K':
                method_name = 'IWAE'
            if method_name == 'RWS10K':
                method_name = 'RWS'

            if model_name == 'occupancy' and (method[:2] == 'vi' or method[:3] == 'HMC'):
                continue

            if method == 'HMC':
                # NOTE: times are already cumulative for HMC (no need for cumsum(0))
                ax[m,1].plot(all_results[method]['times']['p_ll'].mean(-1)[:x_lim_iters], all_results[method]['p_lls'].mean(-1)[:x_lim_iters], label=f"{method_name}", color=colour, alpha=alpha)
                ax[m,2].plot(all_results[method]['times']['moments'].mean(-1)[:x_lim_iters], all_results[method]['MSEs'][:x_lim_iters], label=f"{method_name}", color=colour, alpha=alpha)
                ax[m,3].plot(all_results[method]['times']['moments_fake'].mean(-1)[:x_lim_iters], all_results[method]['MSEs_fake'][:x_lim_iters], label=f"{method_name}", color=colour, alpha=alpha)

                final_xs['p_lls'].append(all_results[method]['times']['p_ll'].mean(-1).cumsum(0)[:x_lim_iters][-1])
                final_xs['vars'].append(all_results[method]['times']['moments'].mean(-1).cumsum(0)[:x_lim_iters][-1])
                final_xs['MSEs'].append(all_results[method]['times']['moments_fake'].mean(-1).cumsum(0)[:x_lim_iters][-1])

            else:
                for lr_idx, lr in enumerate(all_results[method]['lrs']):
                    if lr in model_method_lrs_to_ignore[model_name].get(method, []):
                        continue
                    # alpha = alpha_function(lr_idx, len(all_results[method]['lrs']) - len(model_method_lrs_to_ignore[model_name].get(method, [])))
                    alpha = alpha_function(lr)

                    label = f"{method_name} lr={lr}" if not only_best_lr else f"{method_name}"

                    ax[m,0].plot(all_results[method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['elbos'][lr_idx].mean(-1)[:x_lim_iters], label=label, color=colour, alpha=alpha)
                    ax[m,1].plot(all_results[method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['p_lls'][lr_idx].mean(-1)[:x_lim_iters], label=label, color=colour, alpha=alpha)
                    ax[m,2].plot(all_results[method]['times']['moments'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['MSEs'][lr_idx].cpu()[:x_lim_iters], label=label, color=colour, alpha=alpha)
                    ax[m,3].plot(all_results[method]['times']['moments_fake'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['MSEs_fake'][lr_idx].cpu()[:x_lim_iters], label=label, color=colour, alpha=alpha)

                    final_xs['elbos'].append(all_results[method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])
                    final_xs['p_lls'].append(all_results[method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])
                    final_xs['vars'].append(all_results[method]['times']['moments'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])
                    final_xs['MSEs'].append(all_results[method]['times']['moments_fake'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])

        for j in range(4):
            ax[m,j].set_xlabel('Time (s)')
            # ax[j].tick_params(axis='x', rotation=45)

            if log_x:
                ax[m,j].set_xscale('log')

        if auto_xlim:
            ax[m,0].set_xlim(right=min(final_xs['elbos']))
            ax[m,1].set_xlim(right=min(final_xs['p_lls']))
            ax[m,2].set_xlim(right=min(final_xs['vars']))
            ax[m,3].set_xlim(right=min(final_xs['MSEs']))

        ax[m,0].set_ylabel(f'{model_name.upper()}\nELBO')
        ax[m,1].set_ylabel('Predictive Log-Likelihood')
        ax[m,2].set_ylabel(f'Total Variance{" of variable " + MSE_latent if MSE_latent != "all" else ""}')
        ax[m,3].set_ylabel(f'MSE{" of variable " + MSE_latent if MSE_latent != "all" else ""}')

        ax[m,2].set_yscale('log')
        ax[m,3].set_yscale('log')

        ax[m,0].set_xlim(*xlims.get(model_name, {}).get('elbos', (None, None)))
        ax[m,1].set_xlim(*xlims.get(model_name, {}).get('p_lls', (None, None)))
        ax[m,2].set_xlim(*xlims.get(model_name, {}).get('vars',  (None, None)))
        ax[m,3].set_xlim(*xlims.get(model_name, {}).get('MSEs',  (None, None)))

        ax[m,0].set_ylim(*ylims.get(model_name, {}).get('elbos', (None, None)))
        ax[m,1].set_ylim(*ylims.get(model_name, {}).get('p_lls', (None, None)))
        ax[m,2].set_ylim(*ylims.get(model_name, {}).get('vars',  (None, None)))
        ax[m,3].set_ylim(*ylims.get(model_name, {}).get('MSEs',  (None, None)))

        ax[m,-1].legend()

    # fig.suptitle("Iterative vs IS for " + model_name.upper())

    fig.tight_layout()

    if filename == "":
        filename = f"many_models_iterative_vs_IS_K{mpis_K}-{global_is_K}{'_log_x' if log_x else ''}{'_MSE_' + MSE_latent if MSE_latent != 'all' else ''}"

    plt.savefig(f"plots/{filename}.png")
    if save_pdf:
        plt.savefig(f"plots/pdfs/{filename}.pdf")

    plt.close()

def plot_iterative_vs_IS_single_K_one_model_per_col_no_var_mse(all_models = ALL_MODEL_NAMES, iterative_methods = ['vi', 'rws', 'HMC'], mpis_K = 10, global_is_K = 1000, save_pdf=False, x_lim_iters=10, log_x=False, _model_method_lrs_to_ignore=DEFAULT_MODEL_METHOD_LRS_TO_IGNORE, only_best_lr=False, elbo_validation_iter=DEFAULT_ELBO_VALIDATION_ITER, alpha_function=DEFAULT_ALPHA_FUNC, MSE_latent = 'all', xlims={}, ylims={}, auto_xlim=False, filename=""):
    fig, ax = plt.subplots(2, len(all_models), figsize=(4*len(all_models), 8))

    
    for m, model_name in enumerate(all_models):
    
        all_results = {}

        # only_best_lr = model_method_lrs_to_ignore == 'all_but_best'
        # if only_best_lr:
        #     model_method_lrs_to_ignore = {model_name: {}}

        model_method_lrs_to_ignore = dict_copy(_model_method_lrs_to_ignore)
            
        for method in ['mpis', 'global_is'] + iterative_methods:
            # if not (model_name == 'occupancy' and (method not in ['mpis', 'global_is'] or method[:3] != 'rws')):
            if model_name == 'occupancy' and (method[:2] == 'vi' or method[:3] == 'HMC'):
                continue
            all_results[method] = load_results(model_name, method, False)
            all_results[method]['MSEs'] = choose_MSEs(all_results[method], MSE_latent)

            fake_results = load_results(model_name, method, True)
            all_results[method]['MSEs_fake'] = choose_MSEs(fake_results, MSE_latent)
            all_results[method]['times']['moments_fake'] = fake_results['times']['moments']

            if method in ['mpis', 'global_is']:
                all_results[method] = remove_failed_Ks(all_results, method)

            if only_best_lr and method in iterative_methods and method != 'HMC':
                valid_lr_idxs = np.array([i for i, lr in enumerate(all_results[method]['lrs']) if lr not in model_method_lrs_to_ignore[model_name].get(method, [])])
                best_lr = np.array(all_results[method]['lrs'])[valid_lr_idxs][int(np.argmax(all_results[method]['elbos'].mean(-1)[valid_lr_idxs, elbo_validation_iter].numpy()))]
                model_method_lrs_to_ignore[model_name][method] = [lr for lr in all_results[method]['lrs'] if lr != best_lr]

        final_xs = {'elbos': [], 'p_lls': [], 'vars': [], 'MSEs': []}

        for i, method in enumerate(['mpis', 'global_is']):
            colour = DEFAULT_COLOURS[method]

            Ks = all_results[method]['Ks']
            K_idx = Ks.index(str(mpis_K) if method == 'mpis' else str(global_is_K))

            ax[0,m].scatter(all_results[method]['times']['elbos'].mean(-1)[K_idx], all_results[method]['elbos'].mean(1)[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)
            ax[1,m].scatter(all_results[method]['times']['p_ll'].mean(-1)[K_idx], all_results[method]['p_lls'].mean(1)[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)
            # ax[m,2].scatter(all_results[method]['times']['moments'].mean(-1)[K_idx], all_results[method]['MSEs'].cpu()[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)
            # ax[m,3].scatter(all_results[method]['times']['moments_fake'].mean(-1)[K_idx], all_results[method]['MSEs_fake'].cpu()[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)

        for i, method in enumerate(iterative_methods):
            colour = DEFAULT_COLOURS[method]

            method_name = method.upper()
            if method_name == 'VI10K':
                method_name = 'IWAE'
            if method_name == 'RWS10K':
                method_name = 'RWS'

            if model_name == 'occupancy' and (method[:2] == 'vi' or method[:3] == 'HMC'):
                continue

            if method == 'HMC':
                # NOTE: times are already cumulative for HMC (no need for cumsum(0))
                ax[1,m].plot(all_results[method]['times']['p_ll'].mean(-1)[:x_lim_iters], all_results[method]['p_lls'].mean(-1)[:x_lim_iters], label=f"{method_name}", color=colour, alpha=alpha)
                # ax[m,2].plot(all_results[method]['times']['moments'].mean(-1)[:x_lim_iters], all_results[method]['MSEs'][:x_lim_iters], label=f"{method_name}", color=colour, alpha=alpha)
                # ax[m,3].plot(all_results[method]['times']['moments_fake'].mean(-1)[:x_lim_iters], all_results[method]['MSEs_fake'][:x_lim_iters], label=f"{method_name}", color=colour, alpha=alpha)

                final_xs['p_lls'].append(all_results[method]['times']['p_ll'].mean(-1).cumsum(0)[:x_lim_iters][-1])
                final_xs['vars'].append(all_results[method]['times']['moments'].mean(-1).cumsum(0)[:x_lim_iters][-1])
                final_xs['MSEs'].append(all_results[method]['times']['moments_fake'].mean(-1).cumsum(0)[:x_lim_iters][-1])

            else:
                for lr_idx, lr in enumerate(all_results[method]['lrs']):
                    # alpha = alpha_function(lr_idx, len(all_results[method]['lrs']) - len(model_method_lrs_to_ignore[model_name].get(method, [])))
                    alpha = alpha_function(lr)

                    label = f"{method_name} lr={lr}" if not only_best_lr else f"{method_name}"

                    if lr in model_method_lrs_to_ignore[model_name].get(method, []):
                        ax[0,m].plot([],[], label=label, color=colour, alpha=alpha)
                        ax[1,m].plot([],[], label=label, color=colour, alpha=alpha)
                        continue
                    

                    ax[0,m].plot(all_results[method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['elbos'][lr_idx].mean(-1)[:x_lim_iters], label=label, color=colour, alpha=alpha)
                    ax[1,m].plot(all_results[method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['p_lls'][lr_idx].mean(-1)[:x_lim_iters], label=label, color=colour, alpha=alpha)
                    # ax[2,m].plot(all_results[method]['times']['moments'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['MSEs'][lr_idx].cpu()[:x_lim_iters], label=label, color=colour, alpha=alpha)
                    # ax[3,m].plot(all_results[method]['times']['moments_fake'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['MSEs_fake'][lr_idx].cpu()[:x_lim_iters], label=label, color=colour, alpha=alpha)

                    final_xs['elbos'].append(all_results[method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])
                    final_xs['p_lls'].append(all_results[method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])
                    final_xs['vars'].append(all_results[method]['times']['moments'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])
                    final_xs['MSEs'].append(all_results[method]['times']['moments_fake'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])

        for j in range(2):
            ax[j,m].set_xlabel('Time (s)')
            # ax[j].tick_params(axis='x', rotation=45)

            if log_x:
                ax[j,m].set_xscale('log')

        if auto_xlim:
            ax[0,m].set_xlim(right=min(final_xs['elbos']))
            ax[1,m].set_xlim(right=min(final_xs['p_lls']))
            # ax[m,2].set_xlim(right=min(final_xs['vars']))
            # ax[m,3].set_xlim(right=min(final_xs['MSEs']))

        ax[0, m].set_title(model_name.upper())


        # ax[m,2].set_yscale('log')
        # ax[m,3].set_yscale('log')

        ax[0,m].set_xlim(*xlims.get(model_name, {}).get('elbos', (None, None)))
        ax[1,m].set_xlim(*xlims.get(model_name, {}).get('p_lls', (None, None)))
        # ax[m,2].set_xlim(*xlims.get(model_name, {}).get('vars',  (None, None)))
        # ax[m,3].set_xlim(*xlims.get(model_name, {}).get('MSEs',  (None, None)))

        ax[0,m].set_ylim(*ylims.get(model_name, {}).get('elbos', (None, None)))
        ax[1,m].set_ylim(*ylims.get(model_name, {}).get('p_lls', (None, None)))
        # ax[m,2].set_ylim(*ylims.get(model_name, {}).get('vars',  (None, None)))
        # ax[m,3].set_ylim(*ylims.get(model_name, {}).get('MSEs',  (None, None)))

        if len(iterative_methods) == 1:
            legend_locs = {'vi': 'upper center', 'vi10K': 'upper center', 'rws10K': 'center left'}
            ax[1,-1].legend(loc=legend_locs[iterative_methods[0]])
        else:
            ax[1,-1].legend()

    # fig.suptitle("Iterative vs IS for " + model_name.upper())
    ax[0,0].set_ylabel('ELBO')
    ax[1,0].set_ylabel('Predictive Log-Likelihood')
    # ax[m,2].set_ylabel(f'Total Variance{" of variable " + MSE_latent if MSE_latent != "all" else ""}')
    # ax[m,3].set_ylabel(f'MSE{" of variable " + MSE_latent if MSE_latent != "all" else ""}')

    fig.tight_layout()

    if filename == "":
        filename = f"many_models_iterative_vs_IS_K{mpis_K}-{global_is_K}{'_log_x' if log_x else ''}{'_MSE_' + MSE_latent if MSE_latent != 'all' else ''}"

    plt.savefig(f"plots/{filename}.png")
    if save_pdf:
        plt.savefig(f"plots/pdfs/{filename}.pdf")

    plt.close()


def plot_iterative_vs_IS_single_K_all_models(iterative_methods = ['vi', 'rws', 'HMC'], mpis_K = 10, global_is_K = 1000, save_pdf=False, x_lim_iters=10, log_x=False, _model_method_lrs_to_ignore=DEFAULT_MODEL_METHOD_LRS_TO_IGNORE, only_best_lr=False, elbo_validation_iter=DEFAULT_ELBO_VALIDATION_ITER, alpha_function=DEFAULT_ALPHA_FUNC, MSE_latent = 'all', all_model_names=ALL_MODEL_NAMES, Ks_to_ignore=[1], xlims={}, ylims={}, auto_xlim=False, filename=""):
    all_results = {model_name: {} for model_name in all_model_names}

    model_method_lrs_to_ignore = dict_copy(_model_method_lrs_to_ignore)

    # only_best_lr = model_method_lrs_to_ignore == 'all_but_best'
    # if only_best_lr:
    #     model_method_lrs_to_ignore = {model_name: {} for model_name in all_model_names}

    if isinstance(mpis_K, int):
        temp = mpis_K
        mpis_K = {method: temp for method in all_model_names}
    if isinstance(global_is_K, int):
        temp = global_is_K
        global_is_K = {method: temp for method in all_model_names}

    for model_name in all_model_names:
        for method in ['mpis', 'global_is'] + iterative_methods:
            if (method[:2] == 'vi' or method == 'HMC') and model_name == 'occupancy':
                continue
            all_results[model_name][method] = load_results(model_name, method, False)
            all_results[model_name][method]['MSEs'] = choose_MSEs(all_results[model_name][method], MSE_latent)

            fake_results = load_results(model_name, method, True)
            all_results[model_name][method]['MSEs_fake'] = choose_MSEs(fake_results, MSE_latent)
            all_results[model_name][method]['times']['moments_fake'] = fake_results['times']['moments']

            if method in ['mpis', 'global_is']:
                all_results[model_name][method] = remove_failed_Ks(all_results[model_name], method)

            if only_best_lr and method in iterative_methods and method != 'HMC':
                valid_lr_idxs = np.array([i for i, lr in enumerate(all_results[model_name][method]['lrs']) if lr not in model_method_lrs_to_ignore[model_name].get(method, [])])
                best_lr = np.array(all_results[model_name][method]['lrs'])[valid_lr_idxs][int(np.argmax(all_results[model_name][method]['elbos'].mean(-1)[valid_lr_idxs, elbo_validation_iter].numpy()))]
                model_method_lrs_to_ignore[model_name][method] = [lr for lr in all_results[model_name][method]['lrs'] if lr != best_lr]

    fig, ax = plt.subplots(2, len(all_model_names), figsize=(16, 8))

    final_xs = {model_name: {'elbos': [], 'p_lls': []} for model_name in all_model_names}

    for i, model_name in enumerate(all_model_names):
        for j, method in enumerate(['mpis', 'global_is']):
            colour = DEFAULT_COLOURS[method]

            Ks = all_results[model_name][method]['Ks']
            K_idx = Ks.index(str(mpis_K[model_name]) if method == 'mpis' else str(global_is_K[model_name]))

            ax[0, i].scatter(all_results[model_name][method]['times']['elbos'].mean(-1)[K_idx], all_results[model_name][method]['elbos'].mean(1)[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)
            ax[1, i].scatter(all_results[model_name][method]['times']['p_ll'].mean(-1)[K_idx], all_results[model_name][method]['p_lls'].mean(1)[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)

        for j, method in enumerate(iterative_methods):
            colour = DEFAULT_COLOURS[method]

            method_name = method.upper()
            if method_name == 'VI10K':
                method_name = 'IWAE'
            if method_name == 'RWS10K':
                method_name = 'RWS'

            if model_name == 'occupancy' and method[:3] != 'rws':
                if method[:2] == 'vi': 
                    for lr_idx, lr in enumerate(all_results['chimpanzees'][method]['lrs']):
                        if lr in model_method_lrs_to_ignore['chimpanzees'].get(method, []):
                            continue

                        # alpha = alpha_function(lr_idx, len(all_results['chimpanzees'][method]['lrs']) - len(model_method_lrs_to_ignore['chimpanzees'].get(method, [])))
                        alpha = alpha_function(lr)

                        label = f"{method_name} lr={lr}" if not only_best_lr else f"{method_name}"

                        ax[0,i].plot([], [], label=label, color=colour, alpha=alpha)
                        ax[1,i].plot([], [], label=label, color=colour, alpha=alpha)
                    
                elif method == 'HMC':
                    ax[0,i].plot([], [], label=f"{method_name}", color=colour, alpha=alpha)
                    ax[1,i].plot([], [], label=f"{method_name}", color=colour, alpha=alpha)
                
                continue

            if method == 'HMC':
                # NOTE: times are already cumulative for HMC (no need for cumsum(0))
                ax[1, i].plot(all_results[model_name][method]['times']['p_ll'].mean(-1)[:x_lim_iters], all_results[model_name][method]['p_lls'].mean(-1)[:x_lim_iters], label=f"{method_name}", color=colour, alpha=alpha)
                final_xs[model_name]['p_lls'].append(all_results[model_name][method]['times']['p_ll'].mean(-1).cumsum(0)[:x_lim_iters][-1])

            else:
                for lr_idx, lr in enumerate(all_results[model_name][method]['lrs']):
                    # alpha = alpha_function(lr_idx, len(all_results[model_name][method]['lrs']) - len(model_method_lrs_to_ignore[model_name].get(method, [])))
                    alpha = alpha_function(lr)

                    label = f"{method_name} lr={lr}" if not only_best_lr else f"{method_name}"
                    if lr in model_method_lrs_to_ignore[model_name].get(method, []):

                        # ax[0,i].plot([], [], label=label, color=colour, alpha=alpha)
                        # ax[1,i].plot([], [], label=label, color=colour, alpha=alpha)
                        continue
                    

                        # ax[0, i].plot(all_results[model_name][method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:80], all_results[model_name][method]['elbos'][lr_idx].mean(-1)[:80], label=f"{method_name} lr={lr}", color=colour, alpha=alpha)
                        # ax[1, i].plot(all_results[model_name][method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:80], all_results[model_name][method]['p_lls'][lr_idx].mean(-1)[:80], label=f"{method_name} lr={lr}", color=colour, alpha=alpha)
                    
                    ax[0, i].plot(all_results[model_name][method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[model_name][method]['elbos'][lr_idx].mean(-1)[:x_lim_iters], label=label, color=colour, alpha=alpha)
                    ax[1, i].plot(all_results[model_name][method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[model_name][method]['p_lls'][lr_idx].mean(-1)[:x_lim_iters], label=label, color=colour, alpha=alpha)
                    
                    final_xs[model_name]['elbos'].append(all_results[model_name][method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])
                    final_xs[model_name]['p_lls'].append(all_results[model_name][method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])

        ax[0, i].set_title(model_name.upper())

        ax[1, i].set_xlabel('Time (s)')
        # ax[j].tick_params(axis='x', rotation=45)

        if log_x:
            ax[0, i].set_xscale('log')
            ax[1, i].set_xscale('log')

        if auto_xlim:
            ax[0, i].set_xlim(right=min(final_xs[model_name]['elbos']))
            ax[1, i].set_xlim(right=min(final_xs[model_name]['p_lls']))
            # print(max(final_xs[model_name]['elbos']), max(final_xs[model_name]['p_lls']))
        else:
            ax[0,i].set_xlim(*xlims.get(model_name, {}).get('elbos', (None, None)))
            ax[1,i].set_xlim(*xlims.get(model_name, {}).get('p_lls', (None, None)))

        ax[0,i].set_ylim(*ylims.get(model_name, {}).get('elbos', (None, None)))
        ax[1,i].set_ylim(*ylims.get(model_name, {}).get('p_lls', (None, None)))

        

    # fig.suptitle("Iterative vs IS for " + model_name.upper())
    ax[0,0].set_ylabel('ELBO')
    ax[1,0].set_ylabel('Predictive Log-Likelihood')

    ax[0,-1].legend()

    fig.tight_layout()

    if filename == "":
        filename = f"all_iterative_vs_IS_single_K{'_log_x' if log_x else ''}"

    plt.savefig(f"plots/{filename}.png")
    if save_pdf:
        plt.savefig(f"plots/pdfs/{filename}.pdf")

    plt.close()

def plot_for_poster(model_name, iterative_methods = ['vi', 'rws', 'HMC'], mpis_K = 10, global_is_K = 1000, save_pdf=False, x_lim_iters=10, log_x=False, _model_method_lrs_to_ignore=DEFAULT_MODEL_METHOD_LRS_TO_IGNORE, only_best_lr=False, elbo_validation_iter=DEFAULT_ELBO_VALIDATION_ITER, alpha_function=DEFAULT_ALPHA_FUNC, MSE_latent = 'all', Ks_to_ignore=[1], xlims={}, ylims={}, auto_xlim=False, filename=""):
    all_results = {model_name: {}}

    model_method_lrs_to_ignore = dict_copy(_model_method_lrs_to_ignore)

    # only_best_lr = model_method_lrs_to_ignore == 'all_but_best'
    # if only_best_lr:
    #     model_method_lrs_to_ignore = {model_name: {} for model_name in all_model_names}

    if isinstance(mpis_K, int):
        temp = mpis_K
        mpis_K = {model_name: temp}
    if isinstance(global_is_K, int):
        temp = global_is_K
        global_is_K = {model_name: temp}

    for method in ['mpis', 'global_is'] + iterative_methods:
        if (method[:2] == 'vi' or method == 'HMC') and model_name == 'occupancy':
            continue
        all_results[model_name][method] = load_results(model_name, method, False)
        all_results[model_name][method]['MSEs'] = choose_MSEs(all_results[model_name][method], MSE_latent)

        fake_results = load_results(model_name, method, True)
        all_results[model_name][method]['MSEs_fake'] = choose_MSEs(fake_results, MSE_latent)
        all_results[model_name][method]['times']['moments_fake'] = fake_results['times']['moments']

        if method in ['mpis', 'global_is']:
            all_results[model_name][method] = remove_failed_Ks(all_results[model_name], method)

        if only_best_lr and method in iterative_methods and method != 'HMC':
            valid_lr_idxs = np.array([i for i, lr in enumerate(all_results[model_name][method]['lrs']) if lr not in model_method_lrs_to_ignore[model_name].get(method, [])])
            best_lr = np.array(all_results[model_name][method]['lrs'])[valid_lr_idxs][int(np.argmax(all_results[model_name][method]['elbos'].mean(-1)[valid_lr_idxs, elbo_validation_iter].numpy()))]
            model_method_lrs_to_ignore[model_name][method] = [lr for lr in all_results[model_name][method]['lrs'] if lr != best_lr]

    fig, ax = plt.subplots(3, 1, figsize=(6, 7.5))

    final_xs = {model_name: {'elbos': [], 'p_lls': []}}

    
    
    ax[0].set_title(f"{model_name.upper()} Dataset/Model")

    # Plot 1 & 2: MPIS vs globalIS against K & against time (respectively)
    scatter = False
    for i, x_axis_time in enumerate([False, True]):
        for j, method in enumerate(['mpis', 'global_is']):
            colour = DEFAULT_COLOURS[method]

            Ks_to_ignore_idxs = [all_results[model_name][method]['Ks'].index(str(K)) for K in Ks_to_ignore if str(K) in all_results[model_name][method]['Ks']]
            Ks_to_keep = np.array([i for i in range(len(all_results[model_name][method]['Ks'])) if i not in Ks_to_ignore_idxs])
            if x_axis_time:
                xs = all_results[model_name][method]['times']['p_ll'].mean(-1)
            else:
                xs = all_results[model_name][method]['Ks']
                xs = [int(K) for K in xs]
                ax[i].set_xticks(xs)
                ax[i].set_xscale('log')
            xs = [xs[i] for i in range(len(xs)) if i not in Ks_to_ignore_idxs]
            if scatter:
                ax[i].scatter(xs, all_results[model_name][method]['p_lls'].mean(1)[Ks_to_keep], label=method.upper(), marker='x', color=colour)
            else:
                # breakpoint()
                ax[i].plot(xs, all_results[model_name][method]['p_lls'].mean(1)[Ks_to_keep], label=method.upper(), color=colour)
            ax[i].errorbar(xs, all_results[model_name][method]['p_lls'].mean(1)[Ks_to_keep], yerr=all_results[model_name][method]['p_lls'].std(1)[Ks_to_keep]/np.sqrt(all_results[model_name][method]['num_runs']), fmt='x', color=colour)
            
            if x_axis_time:
                ax[i].set_xlabel('Time (s)')
            else:
                ax[i].set_xlabel('K')
                # ax[i].tick_params(axis='x', rotation=45)

    ax[0].set_xlim(left=1)

    # Plot 3: globalIS vs MPIS vs iterative methods
    for j, method in enumerate(['mpis', 'global_is']):
        colour = DEFAULT_COLOURS[method]

        Ks = all_results[model_name][method]['Ks']
        K_idx = Ks.index(str(mpis_K[model_name]) if method == 'mpis' else str(global_is_K[model_name]))

        ax[2].scatter(all_results[model_name][method]['times']['p_ll'].mean(-1)[K_idx], all_results[model_name][method]['p_lls'].mean(1)[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*', s=150, color=colour)

    for j, method in enumerate(iterative_methods):
        colour = DEFAULT_COLOURS[method]

        method_name = method.upper()
        if method_name == 'VI10K':
            method_name = 'IWAE'
        if method_name == 'RWS10K':
            method_name = 'RWS'

        if model_name == 'occupancy' and method[:3] != 'rws':
            if method[:2] == 'vi': 
                for lr_idx, lr in enumerate(all_results['chimpanzees'][method]['lrs']):
                    if lr in model_method_lrs_to_ignore['chimpanzees'].get(method, []):
                        continue

                    # alpha = alpha_function(lr_idx, len(all_results['chimpanzees'][method]['lrs']) - len(model_method_lrs_to_ignore['chimpanzees'].get(method, [])))
                    alpha = alpha_function(lr)

                    label = f"{method_name} lr={lr}" if not only_best_lr else f"{method_name}"

                    ax[2].plot([], [], label=label, color=colour, alpha=alpha)
                
            elif method == 'HMC':
                ax[2].plot([], [], label=f"{method_name}", color=colour, alpha=alpha)
            
            continue

        if method == 'HMC':
            # NOTE: times are already cumulative for HMC (no need for cumsum(0))
            ax[2].plot(all_results[model_name][method]['times']['p_ll'].mean(-1)[:x_lim_iters], all_results[model_name][method]['p_lls'].mean(-1)[:x_lim_iters], label=f"{method_name}", color=colour, alpha=alpha)
            final_xs[model_name]['p_lls'].append(all_results[model_name][method]['times']['p_ll'].mean(-1).cumsum(0)[:x_lim_iters][-1])

        else:
            for lr_idx, lr in enumerate(all_results[model_name][method]['lrs']):
                # alpha = alpha_function(lr_idx, len(all_results[model_name][method]['lrs']) - len(model_method_lrs_to_ignore[model_name].get(method, [])))
                alpha = alpha_function(lr)

                label = f"{method_name} lr={lr}" if not only_best_lr else f"{method_name}"
                if lr in model_method_lrs_to_ignore[model_name].get(method, []):

                    # ax[0,i].plot([], [], label=label, color=colour, alpha=alpha)
                    # ax[1,i].plot([], [], label=label, color=colour, alpha=alpha)
                    continue
                

                    # ax[0, i].plot(all_results[model_name][method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:80], all_results[model_name][method]['elbos'][lr_idx].mean(-1)[:80], label=f"{method_name} lr={lr}", color=colour, alpha=alpha)
                    # ax[1, i].plot(all_results[model_name][method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:80], all_results[model_name][method]['p_lls'][lr_idx].mean(-1)[:80], label=f"{method_name} lr={lr}", color=colour, alpha=alpha)
                
                ax[2].plot(all_results[model_name][method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[model_name][method]['p_lls'][lr_idx].mean(-1)[:x_lim_iters], label=label, color=colour, alpha=alpha)
                
                final_xs[model_name]['p_lls'].append(all_results[model_name][method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters][-1])


        ax[2].set_xlabel('Time (s)')

        # ax[j].tick_params(axis='x', rotation=45)

        if log_x:
            ax[2].set_xscale('log')

        if auto_xlim:
            ax[2].set_xlim(right=min(final_xs[model_name]['p_lls']))
            # print(max(final_xs[model_name]['elbos']), max(final_xs[model_name]['p_lls']))
        else:
            ax[2].set_xlim(*xlims.get(model_name, {}).get('p_lls', (None, None)))

        # ax[0].set_ylim(*ylims.get(model_name, {}).get('p_lls', (None, None)))
        # ax[1].set_ylim(*ylims.get(model_name, {}).get('p_lls', (None, None)))
        ax[2].set_ylim(*ylims.get(model_name, {}).get('p_lls', (None, None)))

    # fig.suptitle("Iterative vs IS for " + model_name.upper())
    ax[0].set_ylabel('Predictive Log-Likelihood')
    ax[1].set_ylabel('Predictive Log-Likelihood')
    ax[2].set_ylabel('Predictive Log-Likelihood')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    fig.tight_layout()

    if filename == "":
        filename = f"poster{'_log_x' if log_x else ''}"

    plt.savefig(f"plots/{filename}.png")
    if save_pdf:
        plt.savefig(f"plots/pdfs/{filename}.pdf")

    plt.close()


if __name__ == '__main__':
    xlims = {'bus_breakdown': {'elbos': (None, None), 'p_lls': (None, None), 'vars': (None, None), 'MSEs': (None, None)},
             'chimpanzees':   {'elbos': (None, None), 'p_lls': (None, None), 'vars': (None, None), 'MSEs': (None, None)},
             'movielens':     {'elbos': (None, None), 'p_lls': (None, None), 'vars': (None, None), 'MSEs': (None, None)},
             'occupancy':     {'elbos': (None, None), 'p_lls': (None, None), 'vars': (None, None), 'MSEs': (None, None)}}
    
    ylims = {'bus_breakdown': {},
             'chimpanzees':   {'elbos': (-5000, 0)},
             'movielens':     {'elbos': (-20000, 0)},
             'occupancy':     {}}

    # xlims = {}
    # ylims = {}

    mpis_K      = {'bus_breakdown': 15,   'chimpanzees': 15,    'movielens': 15,    'occupancy': 15}
    global_is_K = {'bus_breakdown': 10000,'chimpanzees': 10000, 'movielens': 10000, 'occupancy': 10000}

    final_latents = {'bus_breakdown': 'alpha', 'chimpanzees': 'alpha_block', 'movielens': 'z', 'occupancy': 'z'}

    # print("Making main summary plot")
    # plot_iterative_vs_IS_single_K_all_models(iterative_methods=['vi', 'vi10K', 'rws10K', 'HMC'],  # not including HMC
    #                                          mpis_K = mpis_K,
    #                                          global_is_K = global_is_K,
    #                                          _model_method_lrs_to_ignore=DEFAULT_MODEL_METHOD_LRS_TO_IGNORE,
    #                                          only_best_lr=True,
    #                                          x_lim_iters=1000,
    #                                          log_x=True,
    #                                          Ks_to_ignore=[],
    #                                          xlims=xlims,
    #                                          ylims=ylims,
    #                                          auto_xlim=True,
    #                                          save_pdf=True,
    #                                          filename="summary")
    #                                         #  all_model_names=['bus_breakdown', 'chimpanzees', 'movielens'])
    
    
    # # plot_iterative_vs_IS_single_K_one_model_per_row(all_models=['bus_breakdown', 'chimpanzees', 'movielens', 'occupancy'], iterative_methods=['vi', 'vi10K', 'rws10K', 'HMC'], mpis_K=15, global_is_K=10000, only_best_lr=True, x_lim_iters=1000, log_x=True, MSE_latent='all', xlims=xlims, ylims=ylims, auto_xlim=True, save_pdf=True, filename=f"all_models_summary")
    # print("Making summary plot for VI lrs")
    # plot_iterative_vs_IS_single_K_one_model_per_col_no_var_mse(all_models=['bus_breakdown', 'chimpanzees', 'movielens'], iterative_methods=['vi'], mpis_K=15, global_is_K=10000, only_best_lr=False, x_lim_iters=1000, log_x=True, MSE_latent='all', xlims=xlims, ylims=ylims, auto_xlim=True, save_pdf=True, filename=f"VI_summary")
    
    # ylims['chimpanzees']['elbos'] = (-1000, None)
    # ylims['movielens']['elbos'] = (-10000, None)
    # print("Making summary plot for IWAE lrs")
    # plot_iterative_vs_IS_single_K_one_model_per_col_no_var_mse(all_models=['bus_breakdown', 'chimpanzees', 'movielens'], iterative_methods=['vi10K'], mpis_K=15, global_is_K=10000, only_best_lr=False, x_lim_iters=1000, log_x=True, MSE_latent='all', xlims=xlims, ylims=ylims, auto_xlim=True, save_pdf=True, filename=f"IWAE_summary")
    # print("Making summary plot for RWS lrs")
    # plot_iterative_vs_IS_single_K_one_model_per_col_no_var_mse(all_models=['bus_breakdown', 'chimpanzees', 'movielens', 'occupancy'], iterative_methods=['rws10K'], mpis_K=15, global_is_K=10000, only_best_lr=False, x_lim_iters=1000, log_x=True, MSE_latent='all', xlims=xlims, ylims=ylims, auto_xlim=True, save_pdf=True, filename=f"RWS_summary")


    # # for model_name in ALL_MODEL_NAMES:
    # #     for MSE_latent in ['all']:#, final_latents[model_name]]:
    # #         # plot_IS_per_K_one_model(model_name, MSE_latent=MSE_latent)
    # #         if model_name == 'occupancy':
    # #             # plot_iterative_vs_IS_all_K_one_model(model_name, iterative_methods=['rws10K'], x_lim_iters=1000, MSE_latent=MSE_latent, save_pdf=True, filename=f"{model_name}_summary")
    # #             plot_iterative_vs_IS_single_K_one_model(model_name, iterative_methods=['rws10K'], mpis_K=mpis_K[model_name], global_is_K=global_is_K[model_name], only_best_lr=True, x_lim_iters=1000, log_x=True, MSE_latent=MSE_latent, xlims=xlims, ylims=ylims, auto_xlim=True, save_pdf=True, filename=f"{model_name}_summary")

    # #             plot_iterative_vs_IS_single_K_one_model(model_name, iterative_methods=['rws10K'], mpis_K=mpis_K[model_name], global_is_K=global_is_K[model_name], only_best_lr=False, x_lim_iters=1000, log_x=True, MSE_latent=MSE_latent, xlims=xlims, ylims=ylims, save_pdf=True, filename=f"{model_name}_RWS")

    # #         else:
    # #         # if model_name != 'occupancy':
    # #             # plot_iterative_vs_IS_all_K_one_model(model_name, iterative_methods=['vi'], MSE_latent=MSE_latent)
    # #             plot_iterative_vs_IS_single_K_one_model(model_name, iterative_methods=['vi', 'vi10K', 'rws10K', 'HMC'], mpis_K=mpis_K[model_name], global_is_K=global_is_K[model_name], only_best_lr=True, x_lim_iters=1000, log_x=True, MSE_latent=MSE_latent, xlims=xlims, ylims=ylims, auto_xlim=True, save_pdf=True, filename=f"{model_name}_summary")

    # #             for m, method in enumerate(['vi', 'vi10K', 'rws10K']):
    # #                 method_name = ['VI', 'IWAE', 'RWS'][m]
    # #                 plot_iterative_vs_IS_single_K_one_model(model_name, iterative_methods=[method], mpis_K=mpis_K[model_name], global_is_K=global_is_K[model_name], x_lim_iters=1000, log_x=True, only_best_lr=False, MSE_latent=MSE_latent, xlims=xlims, ylims=ylims, save_pdf=True, filename=f"{model_name}_{method_name}")
    
    # # don't plot IS methods with K=1 or K=3000000: the first is uninteresting (global IS == MP IS) and the second 
    # # comes from a faulty run of chimpanzees that was too unstable to finish
    # Ks_to_ignore = [1,3000000]
    # plots_IS_per_K_all_models(Ks_to_ignore=Ks_to_ignore, save_pdf=True, filename="IS_per_K")

    # plots_IS_per_K_all_models(x_axis_time=True, Ks_to_ignore=Ks_to_ignore, save_pdf=True, filename="IS_per_K_time")
    

    # # plot_iterative_vs_IS_single_K_all_models(iterative_methods=['vi', 'vi10K', 'rws10K', 'HMC'],
    # #                                          mpis_K = mpis_K,
    # #                                          global_is_K = global_is_K,
    # #                                          x_lim_iters=10,
    # #                                          Ks_to_ignore=[],
    # #                                          save_pdf=True)
    # #                                          all_model_names=['bus_breakdown', 'chimpanzees', 'movielens'])


    # POSTER PLOTS
    print("Making main summary plot for poster")
    plot_for_poster(model_name='bus_breakdown',
                    iterative_methods=['vi', 'vi10K', 'rws10K'],# 'HMC'],
                    mpis_K = mpis_K,
                    global_is_K = global_is_K,
                    _model_method_lrs_to_ignore=DEFAULT_MODEL_METHOD_LRS_TO_IGNORE,
                    only_best_lr=True,
                    x_lim_iters=1000,
                    log_x=True,
                    Ks_to_ignore=[1,3000000],
                    xlims=xlims,
                    ylims=ylims,
                    auto_xlim=True,
                    save_pdf=True,
                    filename="poster")
    