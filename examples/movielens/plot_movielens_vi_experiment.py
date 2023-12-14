import pickle
import matplotlib.pyplot as plt
import torch as t
import numpy as np

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
window_sizes = [1, 10, 50]

for window_size in window_sizes:
    # Plot for elbos
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('ELBO')
    for i, K in enumerate(Ks):
        for j, lr in enumerate(lrs):
            mean_values = t.mean(elbos[i, j], axis=1)
            smoothed_mean_values = np.convolve(mean_values, np.ones(window_size)/window_size, mode='valid')
            std_values = t.std(elbos[i, j], axis=1)
            axs[0].plot(smoothed_mean_values, label=f'K={K}, lr={lr}', color=f'C{i}', alpha=(j+1)/len(lrs))
            # axs[0].fill_between(range(len(mean_values)), mean_values - std_values, mean_values + std_values, alpha=0.2)

    # Plot for p_lls
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Predictive Log-Likelihood')
    for i, K in enumerate(Ks):
        for j, lr in enumerate(lrs):
            mean_values = t.mean(p_lls[i, j], axis=1)
            smoothed_mean_values = np.convolve(mean_values, np.ones(window_size)/window_size, mode='valid')
            std_values = t.std(p_lls[i, j], axis=1)
            axs[1].plot(smoothed_mean_values, label=f'K={K}, lr={lr}', color=f'C{i}', alpha=(j+1)/len(lrs))
            # axs[1].fill_between(range(len(mean_values)), mean_values - std_values, mean_values + std_values, alpha=0.2)

    # Add legend
    axs[0].legend()
    axs[1].legend()

    # Add title
    fig.suptitle(f'VI On MovieLens (Smoothing window size: {window_size})')

    # Show the plots
    # plt.show()
    plt.savefig(f'plots/results_{window_size}.png')
    plt.savefig(f'plots/results_{window_size}.pdf')

    # Clear the plots
    axs[0].clear()
    axs[1].clear()
