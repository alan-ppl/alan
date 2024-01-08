import pickle
import matplotlib.pyplot as plt
import torch as t
import numpy as np

def smooth(x, window):
    # result = np.convolve(x, np.ones(window)/window, mode='valid')
    
    result = np.zeros_like(x)

    for i in range(len(x)):
        result[i] = x[max(i-window, 0):i].mean()

    return result

def plot(model_name, method_names=['vi','rws','qem'], window_sizes=[1, 5, 10, 50], dataset_seeds=[0], results_subfolder='', Ks_to_plot='all'):

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
                else:
                    Ks = [K for K in results['Ks'] if K in Ks_to_plot]
            else:
                if Ks_to_plot == 'all':
                    assert np.all(Ks == results['Ks'])
                else:
                    assert np.all([K for K in results['Ks'] if K in Ks_to_plot] == Ks)

        # Average out over the seeds
        elbos[method_name] = np.stack(elbos[method_name]).mean(0)
        p_lls[method_name] = np.stack(p_lls[method_name]).mean(0)
        iter_times[method_name] = np.stack(iter_times[method_name]).mean(0)


    # Create the subplots (top row x=iter, bottom row x=time)
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    # Define the moving average window sizes
    window_sizes = [1, 5, 10, 50]

    for window_size in window_sizes:
        # Plot for elbos
        axs[0,0].set_xlabel('Iteration')
        axs[0,0].set_ylabel('ELBO')
        axs[1,0].set_xlabel('Time (s)')
        axs[1,0].set_ylabel('ELBO')
        for k, K in enumerate(Ks):
            for i, method_name in enumerate(method_names):
                colour = f'C{k*len(method_names) + i}'

                for j, lr in enumerate(lrs[method_name]):
                    mean_values = elbos[method_name][i,j].mean(1)

                    smoothed_mean_values = smooth(mean_values, window_size)
                    
                    std_errs = elbos[method_name][i, j].std(1)/np.sqrt(elbos[method_name].shape[3])
                    
                    times = iter_times[method_name][i,j].mean(1).cumsum()

                    axs[0,0].plot(smoothed_mean_values, label=f'{method_name.upper()}: K={K}, lr={lr}', color=colour, alpha=(j+1)/len(lrs))
                    # axs[0,0].fill_between(range(len(smoothed_mean_values)), smoothed_mean_values - std_errs, smoothed_mean_values + std_errs, alpha=0.2*(j+1)/len(lrs), color=colour)

                    axs[1,0].plot(times, smoothed_mean_values, label=f'{method_name.upper()}: K={K}, lr={lr}', color=colour, alpha=(j+1)/len(lrs))
                    # axs[1,0].fill_between(times, range(len(smoothed_mean_values)), smoothed_mean_values - std_errs, smoothed_mean_values + std_errs, alpha=0.2*(j+1)/len(lrs), color=colour)

                    

                # Plot for p_lls
                axs[0,1].set_xlabel('Iteration')
                axs[0,1].set_ylabel('Predictive Log-Likelihood')
                axs[1,1].set_xlabel('Time (s)')
                axs[1,1].set_ylabel('Predictive Log-Likelihood')
                for k, K in enumerate(Ks):
                    for i, method_name in enumerate(method_names):
                        colour = f'C{k*len(method_names) + i}'

                        for j, lr in enumerate(lrs[method_name]):
                            mean_values = p_lls[method_name][i,j].mean(1)

                            smoothed_mean_values = smooth(mean_values, window_size)

                            std_errs = p_lls[method_name][i, j].std(1)/np.sqrt(p_lls[method_name].shape[3])
                            
                            times = iter_times[method_name][i,j].mean(1).cumsum()

                            axs[0,1].plot(smoothed_mean_values, label=f'{method_name.upper()}: K={K}, lr={lr}', color=colour, alpha=(j+1)/len(lrs))
                            # axs[0,1].fill_between(range(len(smoothed_mean_values)), smoothed_mean_values - std_errs, smoothed_mean_values + std_errs, alpha=0.1*(j+1)/len(lrs), color=colour)

                            axs[0,1].plot(times, smoothed_mean_values, label=f'{method_name.upper()}: K={K}, lr={lr}', color=colour, alpha=(j+1)/len(lrs))
                            # axs[0,1].fill_between(times, range(len(smoothed_mean_values)), smoothed_mean_values - std_errs, smoothed_mean_values + std_errs, alpha=0.1*(j+1)/len(lrs), color=colour)


                # Add title
                fig.suptitle(f'{method_name.upper()} on {model_name} (Smoothing window size: {window_size}){" (K=" + str(Ks_to_plot) + ")" if Ks_to_plot != "all" else ""}')


        # Add legend outside the subplot to the right-hand side with two columns
        # NOTE: When we get p_ll working we'll need to rethink legend positioning/design
        axs[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)

        axs[0,1].set_zorder(-1)
        axs[1,1].set_zorder(-1)
        axs[0,1].clear()
        axs[1,1].clear()

        # Show the plots
        # plt.show()
        plt.savefig(f'{model_name}/plots/results_{window_size}{"_K" + str(Ks_to_plot) if Ks_to_plot != "all" else ""}.png')
        plt.savefig(f'{model_name}/plots/results_{window_size}.pdf')

        # Clear the plots
        for ax in axs.flatten():
            ax.clear()

if __name__ == '__main__':
    plot('movielens', results_subfolder='')
    plot('bus_breakdown', results_subfolder='')

    for K in [3, 10, 30]:
        plot('movielens', results_subfolder='', Ks_to_plot=[K])
        plot('bus_breakdown', results_subfolder='', Ks_to_plot=[K])