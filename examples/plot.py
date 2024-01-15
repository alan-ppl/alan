import pickle
import matplotlib.pyplot as plt
import torch as t
import numpy as np

def smooth(x, window):
    # result = np.convolve(x, np.ones(window)/window, mode='valid')
    
    result = np.zeros_like(x)

    result[0] = x[0]

    for i in range(1,len(x)):
        if x[i] != np.nan:
            result[i] = x[max(i-window, 0):i].mean()

    return result

def plot(model_name, method_names=['vi','rws','qem'], window_sizes=[1, 5, 10, 25, 50], dataset_seeds=[0], results_subfolder='', Ks_to_plot='all', method_lrs_to_ignore={}, elbo_ylims=None, pll_ylims=None):

    print(f'Plotting {model_name} with Ks {Ks_to_plot}.')

    elbos, p_lls, iter_times, lrs = {}, {}, {}, {}
    Ks = None

    for method_name in method_names:
        for x in [elbos, p_lls, iter_times]:
            x[method_name] = []

        lrs[method_name] = None

        for dataset_seed in dataset_seeds:
            # Load the results from the pickle file
            with open(f'{model_name}/results/{results_subfolder}{method_name}{dataset_seed}.pkl', 'rb') as f:
                results = pickle.load(f)

            # Extract the relevant data
            elbos[method_name].append(results['elbos'])
            p_lls[method_name].append(results['p_lls'])
            iter_times[method_name].append(results['iter_times'])

            # Check that the learning rates and Ks are the same for this method throughout all seeds
            if lrs[method_name] is None:
                lrs[method_name] = results['lrs']
            else:
                assert np.all(lrs[method_name] == results['lrs'])

            if Ks is None:
                if Ks_to_plot == 'all':
                    Ks = results['Ks']
                    K_idxs = range(len(Ks))
                else:
                    Ks = [K for K in results['Ks'] if K in Ks_to_plot]
                    K_idxs = [i for i, K in enumerate(results['Ks']) if K in Ks_to_plot]
            else:
                if Ks_to_plot == 'all':
                    assert np.all(Ks == results['Ks'])
                else:
                    assert np.all([K for K in results['Ks'] if K in Ks_to_plot] == Ks)

        # Remove the learning rates that we don't want to plot
        if method_name in method_lrs_to_ignore:
            lrs[method_name] = [lr for lr in lrs[method_name] if lr not in method_lrs_to_ignore[method_name]]

        # Average out over the seeds and convert any 0 elbos or p_lls to NaNs
        elbos[method_name] = np.stack(elbos[method_name])
        p_lls[method_name] = np.stack(p_lls[method_name])

        elbos[method_name][elbos[method_name] == 0] = np.nan
        p_lls[method_name][p_lls[method_name] == 0] = np.nan

        # elbos[method_name] = np.stack(elbos[method_name]).mean(0)
        # p_lls[method_name] = np.stack(p_lls[method_name]).mean(0)
        elbos[method_name] = np.nanmean(np.stack(elbos[method_name]),0)  # ignore nans in mean calculation
        p_lls[method_name] = np.nanmean(np.stack(p_lls[method_name]),0)

        iter_times[method_name] = np.stack(iter_times[method_name]).mean(0)

    # Create the subplots (top row x=iter, bottom row x=time)
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))

    for window_size in window_sizes:
        # Plot for elbos
        axs[0,0].set_xlabel('Iteration')
        axs[0,0].set_ylabel('ELBO')
        axs[1,0].set_xlabel('Time (s)')
        axs[1,0].set_ylabel('ELBO')
        for k, K in enumerate(Ks):
            K_idx = K_idxs[k]
            for i, method_name in enumerate(method_names):
                colour = f'C{k*len(method_names) + i}'

                for j, lr in enumerate(lrs[method_name]):
                    # breakpoint()
                    mean_values = elbos[method_name][K_idx,j].mean(1)

                    smoothed_mean_values = smooth(mean_values, window_size)
                    
                    std_errs = elbos[method_name][K_idx,j].std(1)/np.sqrt(elbos[method_name].shape[3])
                    
                    times = iter_times[method_name][K_idx,j].mean(1).cumsum()

                    alpha_val = 1 - 0.5*j/len(lrs)

                    axs[0,0].plot(smoothed_mean_values, label=f'{method_name.upper()}: K={K}, lr={lr}', color=colour, alpha=alpha_val)
                    # axs[0,0].fill_between(range(len(smoothed_mean_values)), smoothed_mean_values - std_errs, smoothed_mean_values + std_errs, alpha=0.2*alpha_val, color=colour)

                    axs[1,0].plot(times, smoothed_mean_values, label=f'{method_name.upper()}: K={K}, lr={lr}', color=colour, alpha=alpha_val)
                    # axs[1,0].fill_between(times, smoothed_mean_values - std_errs, smoothed_mean_values + std_errs, alpha=0.2*alpha_val, color=colour)

                    if elbo_ylims is not None:
                        axs[0,0].set_ylim(elbo_ylims)
                        axs[1,0].set_ylim(elbo_ylims)
                    

        # Plot for p_lls
        axs[0,1].set_xlabel('Iteration')
        axs[0,1].set_ylabel('Predictive Log-Likelihood')
        axs[1,1].set_xlabel('Time (s)')
        axs[1,1].set_ylabel('Predictive Log-Likelihood')
        for k, K in enumerate(Ks):
            K_idx = K_idxs[k]
            for i, method_name in enumerate(method_names):
                colour = f'C{k*len(method_names) + i}'

                for j, lr in enumerate(lrs[method_name]):
                    mean_values = p_lls[method_name][K_idx,j].mean(1)

                    smoothed_mean_values = smooth(mean_values, window_size)

                    std_errs = p_lls[method_name][K_idx,j].std(1)/np.sqrt(p_lls[method_name].shape[3])
                    
                    times = iter_times[method_name][K_idx,j].mean(1).cumsum()

                    alpha_val = 1 - 0.5*j/len(lrs)

                    # print(mean_values)
                    # breakpoint()

                    axs[0,1].plot(smoothed_mean_values, label=f'{method_name.upper()}: K={K}, lr={lr}', color=colour, alpha=alpha_val)
                    # axs[0,1].fill_between(range(len(smoothed_mean_values)), smoothed_mean_values - std_errs, smoothed_mean_values + std_errs, alpha=0.1*alpha_val, color=colour)

                    axs[1,1].plot(times, smoothed_mean_values, label=f'{method_name.upper()}: K={K}, lr={lr}', color=colour, alpha=alpha_val)
                    # axs[1,1].fill_between(times, smoothed_mean_values - std_errs, smoothed_mean_values + std_errs, alpha=0.1*alpha_val, color=colour)

                    if pll_ylims is not None:
                        axs[0,1].set_ylim(pll_ylims)
                        axs[1,1].set_ylim(pll_ylims)   

        # Add title
        fig.suptitle(f'{model_name.upper()} with K={Ks}\n(Smoothing window size: {window_size})', x=0.3)


        # Add legend 
        if model_name == 'movielens' and results_subfolder == 'regular_version_final/':
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0, 3, 6, 1, 4, 2, 5]
            axs[0,0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1, 1.05), loc='lower left', borderaxespad=0., ncol=3)
        else:
            axs[0,0].legend(bbox_to_anchor=(1, 1.05), loc='lower left', borderaxespad=0., ncol=3)

        axs[0,1].set_zorder(-1)
        axs[1,1].set_zorder(-1)

        # Show the plots
        # plt.show()
        plt.savefig(f'{model_name}/plots/results_{window_size}{"_K" + str(Ks_to_plot) if Ks_to_plot != "all" else ""}.png')
        plt.savefig(f'{model_name}/plots/results_{window_size}.pdf')

        # Clear the plots
        for ax in axs.flatten():
            ax.clear()

if __name__ == '__main__':
    # plot('movielens', results_subfolder='regular_version_final/',
    #       method_lrs_to_ignore={'qem': [0.01], 'rws': [0.0001], 'vi': [0.0001]})
    # plot('bus_breakdown', results_subfolder='', 
    #      method_lrs_to_ignore={'qem': [0.01], 'rws': [0.0001], 'vi': [0.0001]})
    
    elbo_ylims_per_K = {'movielens':     {3: (-6500, None), 10: (-3000, -950), 30: (-2000, -950)},
                        'bus_breakdown': {3: (-6000, None), 10: (-3300, None), 30: (-2750, None)}}
    pll_ylims_per_K  = {'movielens':     {3: (-1150, None), 10: (-1100, -940), 30: (-1060, -940)},
                        'bus_breakdown': {3: (-7000, None), 10: (-3500, None), 30: (-2800, -1750)}}

    for K in [3, 10, 30]:

        # plot('movielens', method_names=['qem','vi'], results_subfolder='', Ks_to_plot=[K])
            #   elbo_ylims=elbo_ylims_per_K['movielens'][K], pll_ylims=pll_ylims_per_K['movielens'][K])

        plot('movielens', results_subfolder='regular_version_final_FULL/', Ks_to_plot=[K],
              method_lrs_to_ignore={'qem': [0.01,0.2], 'rws': [0.0001], 'vi': [0.0001,0.2]},
              elbo_ylims=elbo_ylims_per_K['movielens'][K], pll_ylims=pll_ylims_per_K['movielens'][K])

        # plot('movielens', results_subfolder='regular_version_final/', Ks_to_plot=[K],
        #       method_lrs_to_ignore={'qem': [0.01], 'rws': [0.0001], 'vi': [0.0001]},
        #       elbo_ylims=elbo_ylims_per_K['movielens'][K], pll_ylims=pll_ylims_per_K['movielens'][K])

        # plot('bus_breakdown', results_subfolder='', Ks_to_plot=[K],
        #       method_lrs_to_ignore={'qem': [0.01], 'rws': [0.0001], 'vi': [0.0001]},
        #       elbo_ylims=elbo_ylims_per_K['bus_breakdown'][K], pll_ylims=pll_ylims_per_K['bus_breakdown'][K])
