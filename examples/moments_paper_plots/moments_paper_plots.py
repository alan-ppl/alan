import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch as t 

ALL_MODEL_NAMES = ['bus_breakdown', 'chimpanzees', 'movielens', 'occupancy']

DEFAULT_ALPHA_FUNC = lambda i, num_lrs: 1 if i == 0 else 1 - 0.5*i/(num_lrs-1)

def load_results(model_name, method_name, fake_data, dataset_seed=0):
    with open(f'../{model_name}/results/moments/{method_name}{dataset_seed}{"_FAKE_DATA" if fake_data else ""}.pkl', 'rb') as f:
        return pickle.load(f)
    
def sum_all_MSEs(results):
    total = None
    for key in results['MSEs']:
        if total is None:
            total = results['MSEs'][key]
        else:
            total += results['MSEs'][key]
    return total

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

    
def plot_IS_per_K_one_model(model_name, save_pdf=False):
    all_results = {}
    for method in ['mpis', 'global_is']:
        all_results[method] = load_results(model_name, method, False)
        all_results[method]['MSEs'] = sum_all_MSEs(all_results[method])
        all_results[method]['MSEs_fake'] = sum_all_MSEs(load_results(model_name, method, True))

        all_results[method] = remove_failed_Ks(all_results, method)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    for i, method in enumerate(['mpis', 'global_is']):
        for j, key in enumerate(['elbos', 'p_lls', 'MSEs', 'MSEs_fake']):
            if key in ['elbos', 'p_lls']:
                ax[j].plot(all_results[method]['Ks'], all_results[method][key].mean(1), label=method.upper())
            else:
                ax[j].plot(all_results[method]['Ks'], all_results[method][key].cpu(), label=method.upper())
            ax[j].set_xlabel('K')
            ax[j].tick_params(axis='x', rotation=45)

    fig.suptitle("IS Comparison for " + model_name.upper())
    ax[0].set_ylabel('ELBO')
    ax[1].set_ylabel('Predictive Log-Likelihood')
    ax[2].set_ylabel('Total Variance')
    ax[3].set_ylabel('MSE')
    ax[-1].legend()

    plt.xticks(rotation=70)

    fig.tight_layout()

    plt.savefig(f"plots/{model_name}_IS_per_K.png")
    if save_pdf:
        plt.savefig(f"plots/{model_name}_IS_per_K.pdf")

def plots_IS_per_K_all_models(save_pdf=False):
    all_results = {model_name: {} for model_name in ALL_MODEL_NAMES}
    for model_name in ALL_MODEL_NAMES:
        for method in ['mpis', 'global_is']:
            all_results[model_name][method] = load_results(model_name, method, False)
            all_results[model_name][method]['MSEs'] = sum_all_MSEs(all_results[model_name][method])
            all_results[model_name][method]['MSEs_fake'] = sum_all_MSEs(load_results(model_name, method, True))

            all_results[model_name][method] = remove_failed_Ks(all_results[model_name], method)

    fig, ax = plt.subplots(2, 4, figsize=(20, 10))

    for i, model_name in enumerate(ALL_MODEL_NAMES):
        for j, method in enumerate(['mpis', 'global_is']):
            for k, key in enumerate(['elbos', 'p_lls']):
                if key in ['elbos', 'p_lls']:
                    ax[k, i].plot(all_results[model_name][method]['Ks'], all_results[model_name][method][key].mean(1), label=method.upper())
                else:
                    ax[k, i].plot(all_results[model_name][method]['Ks'], all_results[model_name][method][key].cpu(), label=method.upper())
                ax[k, i].set_xlabel('K')
                ax[k, i].tick_params(axis='x', rotation=45)

        ax[0, i].set_title(model_name.upper())

    ax[0,0].set_ylabel('ELBO')
    ax[1,0].set_ylabel('Predictive Log-Likelihood')
    ax[0,0].legend()


    fig.tight_layout()

    # fig.suptitle("IS Comparison for All Models")

    plt.savefig(f"plots/all_IS_per_K.png")
    if save_pdf:
        plt.savefig(f"plots/all_IS_per_K.pdf")


def plot_iterative_vs_IS_all_K_one_method(model_name, iterative_methods = ['vi', 'rws', 'HMC'], save_pdf=False, x_lim_iters=10):
    all_results = {}
    for method in ['mpis', 'global_is'] + iterative_methods:
        all_results[method] = load_results(model_name, method, False)
        all_results[method]['MSEs'] = sum_all_MSEs(all_results[method])

        fake_results = load_results(model_name, method, True)
        all_results[method]['MSEs_fake'] = sum_all_MSEs(fake_results)
        all_results[method]['times']['moments_fake'] = fake_results['times']['moments']

        if method in ['mpis', 'global_is']:
            all_results[method] = remove_failed_Ks(all_results, method)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    for i, method in enumerate(['mpis', 'global_is']):
        ax[0].plot(all_results[method]['times']['elbos'].mean(-1), all_results[method]['elbos'].mean(1), label=method.upper())
        ax[1].plot(all_results[method]['times']['p_ll'].mean(-1), all_results[method]['p_lls'].mean(1), label=method.upper())
        ax[2].plot(all_results[method]['times']['moments'].mean(-1), all_results[method]['MSEs'].cpu(), label=method.upper())
        ax[3].plot(all_results[method]['times']['moments_fake'].mean(-1), all_results[method]['MSEs_fake'].cpu(), label=method.upper())

    for i, method in enumerate(iterative_methods):
        colour = 'C' + str(i + 2)

        for lr_idx, lr in enumerate(all_results[method]['lrs']):
            alpha = DEFAULT_ALPHA_FUNC(lr_idx, len(all_results[method]['lrs']))

            ax[0].plot(all_results[method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['elbos'][lr_idx].mean(-1)[:x_lim_iters], label=f"{method.upper()} lr={lr}", color=colour, alpha=alpha)
            ax[1].plot(all_results[method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['p_lls'][lr_idx].mean(-1)[:x_lim_iters], label=f"{method.upper()} lr={lr}", color=colour, alpha=alpha)
            ax[2].plot(all_results[method]['times']['moments'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['MSEs'][lr_idx].cpu()[:x_lim_iters], label=f"{method.upper()} lr={lr}", color=colour, alpha=alpha)
            ax[3].plot(all_results[method]['times']['moments_fake'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['MSEs_fake'][lr_idx].cpu()[:x_lim_iters], label=f"{method.upper()} lr={lr}", color=colour, alpha=alpha)

    for j in range(4):
        ax[j].set_xlabel('Time (s)')
        # ax[j].tick_params(axis='x', rotation=45)

    fig.suptitle("Iterative vs IS for " + model_name.upper())
    ax[0].set_ylabel('ELBO')
    ax[1].set_ylabel('Predictive Log-Likelihood')
    ax[2].set_ylabel('Total Variance')
    ax[3].set_ylabel('MSE')

    ax[-1].legend()

    fig.tight_layout()

    plt.savefig(f"plots/{model_name}_iterative_vs_IS_all_K.png")
    if save_pdf:
        plt.savefig(f"plots/{model_name}_iterative_vs_IS_all_K.pdf")

def plot_iterative_vs_IS_single_K_one_model(model_name, iterative_methods = ['vi', 'rws', 'HMC'], mpis_K = 10, global_is_K = 1000, save_pdf=False, x_lim_iters=10):
    all_results = {}
    for method in ['mpis', 'global_is'] + iterative_methods:
        all_results[method] = load_results(model_name, method, False)
        all_results[method]['MSEs'] = sum_all_MSEs(all_results[method])

        fake_results = load_results(model_name, method, True)
        all_results[method]['MSEs_fake'] = sum_all_MSEs(fake_results)
        all_results[method]['times']['moments_fake'] = fake_results['times']['moments']

        if method in ['mpis', 'global_is']:
            all_results[method] = remove_failed_Ks(all_results, method)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    for i, method in enumerate(['mpis', 'global_is']):
        Ks = all_results[method]['Ks']
        K_idx = Ks.index(str(mpis_K) if method == 'mpis' else str(global_is_K))

        ax[0].plot(all_results[method]['times']['elbos'].mean(-1)[K_idx], all_results[method]['elbos'].mean(1)[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*')
        ax[1].plot(all_results[method]['times']['p_ll'].mean(-1)[K_idx], all_results[method]['p_lls'].mean(1)[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*')
        ax[2].plot(all_results[method]['times']['moments'].mean(-1)[K_idx], all_results[method]['MSEs'].cpu()[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*')
        ax[3].plot(all_results[method]['times']['moments_fake'].mean(-1)[K_idx], all_results[method]['MSEs_fake'].cpu()[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*')

    for i, method in enumerate(iterative_methods):
        colour = 'C' + str(i + 2)

        for lr_idx, lr in enumerate(all_results[method]['lrs']):
            alpha = DEFAULT_ALPHA_FUNC(lr_idx, len(all_results[method]['lrs']))

            ax[0].plot(all_results[method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['elbos'][lr_idx].mean(-1)[:x_lim_iters], label=f"{method.upper()} lr={lr}", color=colour, alpha=alpha)
            ax[1].plot(all_results[method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['p_lls'][lr_idx].mean(-1)[:x_lim_iters], label=f"{method.upper()} lr={lr}", color=colour, alpha=alpha)
            ax[2].plot(all_results[method]['times']['moments'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['MSEs'][lr_idx].cpu()[:x_lim_iters], label=f"{method.upper()} lr={lr}", color=colour, alpha=alpha)
            ax[3].plot(all_results[method]['times']['moments_fake'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[method]['MSEs_fake'][lr_idx].cpu()[:x_lim_iters], label=f"{method.upper()} lr={lr}", color=colour, alpha=alpha)

    for j in range(4):
        ax[j].set_xlabel('Time (s)')
        # ax[j].tick_params(axis='x', rotation=45)

    fig.suptitle("Iterative vs IS for " + model_name.upper())
    ax[0].set_ylabel('ELBO')
    ax[1].set_ylabel('Predictive Log-Likelihood')
    ax[2].set_ylabel('Total Variance')
    ax[3].set_ylabel('MSE')

    ax[-1].legend()

    fig.tight_layout()

    plt.savefig(f"plots/{model_name}_iterative_vs_IS_K{mpis_K}-{global_is_K}.png")
    if save_pdf:
        plt.savefig(f"plots/{model_name}_iterative_vs_IS_all_K{mpis_K}-{global_is_K}.pdf")

def plot_iterative_vs_IS_single_K_all_models(iterative_methods = ['vi', 'rws', 'HMC'], mpis_K = 10, global_is_K = 1000, save_pdf=False, x_lim_iters=10):
    all_results = {model_name: {} for model_name in ALL_MODEL_NAMES}

    if isinstance(mpis_K, int):
        mpis_K = {method: mpis_K for method in ['mpis', 'global_is']+iterative_methods}
    if isinstance(global_is_K, int):
        global_is_K = {method: global_is_K for method in ['mpis', 'global_is']+iterative_methods}

    for model_name in ALL_MODEL_NAMES:
        for method in ['mpis', 'global_is'] + iterative_methods:
            if method == 'vi' and model_name == 'occupancy':
                continue
            all_results[model_name][method] = load_results(model_name, method, False)
            all_results[model_name][method]['MSEs'] = sum_all_MSEs(all_results[model_name][method])

            fake_results = load_results(model_name, method, True)
            all_results[model_name][method]['MSEs_fake'] = sum_all_MSEs(fake_results)
            all_results[model_name][method]['times']['moments_fake'] = fake_results['times']['moments']

            if method in ['mpis', 'global_is']:
                all_results[model_name][method] = remove_failed_Ks(all_results[model_name], method)

    fig, ax = plt.subplots(2, 4, figsize=(20, 10))

    for i, model_name in enumerate(ALL_MODEL_NAMES):
        for j, method in enumerate(['mpis', 'global_is']):
            Ks = all_results[model_name][method]['Ks']
            K_idx = Ks.index(str(mpis_K[method]) if method == 'mpis' else str(global_is_K[method]))

            ax[0, i].plot(all_results[model_name][method]['times']['elbos'].mean(-1)[K_idx], all_results[model_name][method]['elbos'].mean(1)[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*')
            ax[1, i].plot(all_results[model_name][method]['times']['p_ll'].mean(-1)[K_idx], all_results[model_name][method]['p_lls'].mean(1)[K_idx], label=f"{method.upper()} K={Ks[K_idx]}", marker='*')
        
        for j, method in enumerate(iterative_methods):
            colour = 'C' + str(j + 2)

            if method == 'vi' and model_name == 'occupancy':
                for lr_idx, lr in enumerate(all_results['movielens'][method]['lrs']):
                    alpha = DEFAULT_ALPHA_FUNC(lr_idx, len(all_results['movielens'][method]['lrs']))

                    ax[0,i].plot([], [], label=f"{method.upper()} lr={lr}", color=colour, alpha=alpha)
                    ax[1,i].plot([], [], label=f"{method.upper()} lr={lr}", color=colour, alpha=alpha)
                continue

            for lr_idx, lr in enumerate(all_results[model_name][method]['lrs']):
                alpha = DEFAULT_ALPHA_FUNC(lr_idx, len(all_results[model_name][method]['lrs']))

                ax[0, i].plot(all_results[model_name][method]['times']['elbos'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[model_name][method]['elbos'][lr_idx].mean(-1)[:x_lim_iters], label=f"{method.upper()} lr={lr}", color=colour, alpha=alpha)
                ax[1, i].plot(all_results[model_name][method]['times']['p_ll'][lr_idx].mean(-1).cumsum(0)[:x_lim_iters], all_results[model_name][method]['p_lls'][lr_idx].mean(-1)[:x_lim_iters], label=f"{method.upper()} lr={lr}", color=colour, alpha=alpha)

        ax[0, i].set_title(model_name.upper())

        ax[1, i].set_xlabel('Time (s)')
        # ax[j].tick_params(axis='x', rotation=45)

    # fig.suptitle("Iterative vs IS for " + model_name.upper())
    ax[0,0].set_ylabel('ELBO')
    ax[1,0].set_ylabel('Predictive Log-Likelihood')

    ax[0,-1].legend()

    fig.tight_layout()

    plt.savefig(f"plots/all_iterative_vs_IS_single_K.png")
    if save_pdf:
        plt.savefig(f"plots/all_iterative_vs_IS_single_K.pdf")


if __name__ == '__main__':
    for model_name in ALL_MODEL_NAMES:
        # plot_IS_per_K_one_model(model_name)
        if model_name == 'occupancy':
            # plot_iterative_vs_IS_all_K_one_method(model_name, iterative_methods=['rws'])
            plot_iterative_vs_IS_single_K_one_model(model_name, iterative_methods=['rws'])
        else:
            # plot_iterative_vs_IS_all_K_one_method(model_name, iterative_methods=['vi', 'rws'])
            plot_iterative_vs_IS_single_K_one_model(model_name, iterative_methods=['vi', 'rws'])
    
    # plots_IS_per_K_all_models()
    plot_iterative_vs_IS_single_K_all_models(iterative_methods=['vi', 'rws'])
