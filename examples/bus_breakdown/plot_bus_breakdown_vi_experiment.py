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

# Load the results from the pickle file
with open('results/results.pkl', 'rb') as f:
    results = pickle.load(f)

# Extract the relevant data
elbos = results['elbos']
p_lls = results['p_lls']
Ks = results['Ks']
lrs = results['lrs']

# Create the subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Define the moving average window sizes
window_sizes = [1, 5, 10, 50]

for window_size in window_sizes:
    # Plot for elbos
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('ELBO')
    for i, K in enumerate(Ks):
        for j, lr in enumerate(lrs):
            mean_values = t.mean(elbos[i, j], axis=1)

            # padded_start = mean_values[0]*np.ones(window_size-1)
            # mean_padded = np.concatenate([padded_start, mean_values])
            # smoothed_mean_values = np.convolve(mean_values, np.ones(window_size)/window_size, mode='valid')

            smoothed_mean_values = smooth(mean_values, window_size)
            
            std_errs = t.std(elbos[i, j], axis=1).numpy()/np.sqrt(elbos.shape[3])
            
            axs[0].plot(smoothed_mean_values, label=f'K={K}, lr={lr}', color=f'C{i}', alpha=(j+1)/len(lrs))
            axs[0].fill_between(range(len(smoothed_mean_values)), smoothed_mean_values - std_errs, smoothed_mean_values + std_errs, alpha=0.2*(j+1)/len(lrs), color=f'C{i}')

    # Plot for p_lls
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Predictive Log-Likelihood')
    for i, K in enumerate(Ks):
        for j, lr in enumerate(lrs):
            mean_values = t.mean(p_lls[i, j], axis=1)

            # padded_start = mean_values[0]*np.ones(window_size-1)
            # mean_padded = np.concatenate([padded_start, mean_values])
            # smoothed_mean_values = np.convolve(mean_values, np.ones(window_size)/window_size, mode='valid')

            smoothed_mean_values = smooth(mean_values, window_size)

            std_errs = t.std(p_lls[i, j], axis=1).numpy()/np.sqrt(p_lls.shape[3])
            
            axs[1].plot(smoothed_mean_values, label=f'K={K}, lr={lr}', color=f'C{i}', alpha=(j+1)/len(lrs))
            axs[1].fill_between(range(len(smoothed_mean_values)), smoothed_mean_values - std_errs, smoothed_mean_values + std_errs, alpha=0.1*(j+1)/len(lrs), color=f'C{i}')

    # Add legend
    axs[0].legend(loc='lower right')
    axs[1].legend(loc='lower right')

    # Add title
    fig.suptitle(f'VI On Bus Breakdown (Smoothing window size: {window_size})')

    # Show the plots
    # plt.show()
    plt.savefig(f'plots/results_{window_size}.png')
    plt.savefig(f'plots/results_{window_size}.pdf')

    # Clear the plots
    axs[0].clear()
    axs[1].clear()
