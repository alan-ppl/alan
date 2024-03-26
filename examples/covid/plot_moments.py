import pickle
import matplotlib.pyplot as plt
import torch as t
import numpy as np

from pathlib import Path

models = ['covid']# 'poisson_only_wearing_mobility', 'poisson_only_npis']

for mod in models:
    #Get moments from file
    with open(f'results/{mod}/qem_moments_30_0.1.pkl', 'rb') as f:
        moments = pickle.load(f)
        
    #Get predictive samples from file
    with open(f'results/{mod}/qem_predictive_samples_30_0.1.pkl', 'rb') as f:
        predicted_obs = pickle.load(f)
        
    #Get standard deviation from moments
    moments = moments[0]
    std = np.sqrt(moments['CM_ex2'] - moments['CM_mean']**2)

    std_w = np.sqrt(moments['Wearing_ex2'] - moments['Wearing_mean']**2)

    std_m = np.sqrt(moments['Mobility_ex2'] - moments['Mobility_mean']**2)


    #NPIS
    NPIS = ['C1', 'C1_full', 'C2', 'C2_full', 'C4_3plus', 'C6', 'C7', 'C4_2plus', 'C4_full']

    #plot bar chart of moments, plotting CMs separately
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    #Divide by number of samples (100)
    ax[0,0].bar(np.arange(9), moments['CM_mean'], yerr=std)
    ax[0,0].set_title('CMs')
    ax[0,0].set_ylabel('Mean (Moment)')
    #hide xticks
    ax[0,0].set_xticks([])




    #plot bar chart of moments, plotting Wearing and Mobility separately
    ax[0,1].bar(np.arange(2), [moments['Wearing_mean'], moments['Mobility_mean']], yerr=[std_w, std_m])
    ax[0,1].set_title('Wearing and Mobility')
    ax[0,1].set_xticks([])
    

    #plot predictive samples as timeseries

    #load obs
    obs = t.load('data/obs.pt').numpy()

    #plot
    #5 colors
    colors = ['b', 'g', 'r', 'c', 'm']

    print(predicted_obs['obs'].shape)
    print(obs.shape)
    ax[0,2].plot(predicted_obs['obs'].mean(0)[:1,:].T, color='k', label='Predicted data')
    ax[0,2].plot(obs[:1,:].T, linestyle='--', color='k', label='Real data')
    ax[0,2].set_title('Predicted vs Real data (Training)')
    ax[0,2].set_xlabel('Week')
    ax[0,2].set_ylabel('Observed cases')
    ax[0,2].legend()



    
    #make same plot using parameters from the last iteration
    #Get model parameters from file
    params = t.load(f'results/{mod}/qem_0_30_0.1.pth', map_location='cpu')
    
    CM_mean = params['Q._qem_means.CM_alpha_mean'].numpy()
    CM_std = np.sqrt(params['Q._qem_means.CM_alpha_mean2'].numpy() - CM_mean**2)
    #plot parameter means
    ax[1,0].bar(np.arange(9), CM_mean, yerr=CM_std)
    NPIS = ['Some schools', 'All schools', 'Some business', 'All non-essential business', '11-1000 people', 'No leaving house', 'Internal movement restrictions', '101-1000 people', '-10 people']
    ax[1,0].set_xticks(np.arange(9))
    ax[1,0].set_xticklabels(NPIS)
    ax[1,0].set_ylabel('Mean (Parameter)')


    
    #Rotate labels and set position
    for tick in ax[1,0].get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right')


    #plot bar chart of moments, plotting Wearing and Mobility separately
    wearing_mean = params['Q._qem_means.Wearing_alpha_mean'].numpy()
    wearing_std = np.sqrt(params['Q._qem_means.Wearing_alpha_mean2'].numpy() - wearing_mean**2)
    
    mobility_mean = params['Q._qem_means.Mobility_alpha_mean'].numpy()
    mobility_std = np.sqrt(params['Q._qem_means.Mobility_alpha_mean2'].numpy() - mobility_mean**2)
    
    
    ax[1,1].bar(np.arange(2), [wearing_mean, mobility_mean], yerr=[wearing_std, mobility_std])
    ax[1,1].set_xticks(np.arange(2))
    
    
    ax[1,1].set_xticklabels(['Wearing', 'Mobility'])
    for tick in ax[1,1].get_xticklabels():
        tick.set_rotation(45)
    
    
    
    ax[1][2].set_visible(False)

    # ax[1][0].set_position([0.24,0.125,0.228,0.343])
    # ax[1][1].set_position([0.55,0.125,0.228,0.343])

    Path(f"plots/{mod}/").mkdir(parents=True, exist_ok=True)
    plt.savefig(f'plots/{mod}/moments.png')
    


