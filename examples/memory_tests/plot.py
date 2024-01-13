import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
comp_modes = ['no_checkpoint', 'checkpoint', 'Split=10']

matplotlib.rcParams.update({'font.size': 17})

p_sizes = [30,100,300,1000, 3000]

Ks = [10,30,100,300, 1000, 3000]

# with open('mem_usage.pkl', 'rb') as f:
#     mem_usage = pickle.load(f)
    
# with open('time_usage.pkl', 'rb') as f:
#     time_usage = pickle.load(f)

titles = ['Checkpointing in reduce_Ks', 'No checkpointing in reduce_Ks']  
for mode in comp_modes:

    fig, ax = plt.subplots(1,2,subplot_kw={"projection": "3d"})
    fig.set_size_inches(18.5, 10.5)
    for i in range(2):
        if i == 0:     
            with open('with_checkpointing_reduce_Ks/mem_usage.pkl', 'rb') as f:
                mem_usage = pickle.load(f)
                
            with open('with_checkpointing_reduce_Ks/time_usage.pkl', 'rb') as f:
                time_usage = pickle.load(f)
        else:
            with open('without_checkpointing_reduce_Ks/mem_usage.pkl', 'rb') as f:
                mem_usage = pickle.load(f)
                
            with open('without_checkpointing_reduce_Ks/time_usage.pkl', 'rb') as f:
                time_usage = pickle.load(f)
        #Plot surface defined by numpy arrays

        ax[i].set_title(titles[i])
        ax[i].set_xlabel('Plate size')
        ax[i].set_ylabel('K')
        ax[i].set_zlabel('Memory usage (GB)')

        ax[i].set_xticks(np.log10(Ks))
        ax[i].set_xticklabels(Ks)
        ax[i].set_yticks(np.log10(p_sizes))
        ax[i].set_yticklabels(p_sizes)
        ax[i].set_zlim(0,9)
        ps, ks = np.meshgrid(np.log10(p_sizes), np.log10(Ks), indexing='ij')
        mem_usage[mode][np.isnan(mem_usage[mode])] = np.nan
        
        ax[i].plot_surface(ps, ks, np.array(mem_usage[mode]/1024**3), cmap='viridis')
    

    fig.suptitle(f"Memory usage with computation strategy: {mode}")
    plt.savefig('mem_usage_{}.png'.format(mode))
    
    fig, ax = plt.subplots(1,2,subplot_kw={"projection": "3d"})
    fig.set_size_inches(18.5, 10.5)
    for i in range(2):  
        
        if i == 0:     
            with open('with_checkpointing_reduce_Ks/mem_usage.pkl', 'rb') as f:
                mem_usage = pickle.load(f)
                
            with open('with_checkpointing_reduce_Ks/time_usage.pkl', 'rb') as f:
                time_usage = pickle.load(f)
        else:
            with open('without_checkpointing_reduce_Ks/mem_usage.pkl', 'rb') as f:
                mem_usage = pickle.load(f)
                
            with open('without_checkpointing_reduce_Ks/time_usage.pkl', 'rb') as f:
                time_usage = pickle.load(f) 
            #Plot surface defined by numpy arrays
        ax[i].set_title(titles[i])
        ax[i].set_xlabel('Plate size')
        ax[i].set_ylabel('K')
        ax[i].set_zlabel('Time')

        ax[i].set_xticks(np.log10(Ks))
        ax[i].set_xticklabels(Ks)
        ax[i].set_yticks(np.log10(p_sizes))
        ax[i].set_yticklabels(p_sizes)
        # ax[i].set_zlim(0,3e-8)
        ps, ks = np.meshgrid(np.log10(p_sizes), np.log10(Ks), indexing='ij')
        mem_usage[mode][np.isnan(mem_usage[mode])] = np.nan
        
        ax[i].plot_surface(ps, ks, np.array(time_usage[mode]/1024**3), cmap='viridis')
    

    fig.suptitle(f"Time to calculate moment with computation strategy: {mode}")
    plt.savefig('time_usage_{}.png'.format(mode))
    
