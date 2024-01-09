import pickle
import matplotlib.pyplot as plt
import numpy as np
comp_modes = ['no_checkpoint', 'checkpoint', 'Split=10']


p_sizes = [30,100,300,1000, 3000]

Ks = [10,30,100,300, 1000, 3000]

with open('mem_usage.pkl', 'rb') as f:
    mem_usage = pickle.load(f)
    
with open('time_usage.pkl', 'rb') as f:
    time_usage = pickle.load(f)
        
for mode in comp_modes:

    #Plot surface defined by numpy arrays
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_title(mode)
    ax.set_xlabel('Plate size')
    ax.set_ylabel('K')
    ax.set_zlabel('Memory usage (GB)')

    ax.set_xticks(np.log10(Ks))
    ax.set_xticklabels(Ks)
    ax.set_yticks(np.log10(p_sizes))
    ax.set_yticklabels(p_sizes)
    ax.set_zlim(0,9)
    ps, ks = np.meshgrid(np.log10(p_sizes), np.log10(Ks), indexing='ij')
    mem_usage[mode][np.isnan(mem_usage[mode])] = np.nan
    
    ax.plot_surface(ps, ks, np.array(mem_usage[mode]/1024**3), cmap='viridis')
    
    plt.savefig('mem_usage_{}.png'.format(mode))
    
        #Plot surface defined by numpy arrays
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_title(mode)
    ax.set_xlabel('Plate size')
    ax.set_ylabel('K')
    ax.set_zlabel('Time')

    ax.set_xticks(np.log10(Ks))
    ax.set_xticklabels(Ks)
    ax.set_yticks(np.log10(p_sizes))
    ax.set_yticklabels(p_sizes)
    ps, ks = np.meshgrid(np.log10(p_sizes), np.log10(Ks), indexing='ij')
    mem_usage[mode][np.isnan(mem_usage[mode])] = np.nan
    
    ax.plot_surface(ps, ks, np.array(time_usage[mode]/1024**3), cmap='viridis')
    
    plt.savefig('time_usage_{}.png'.format(mode))
    
