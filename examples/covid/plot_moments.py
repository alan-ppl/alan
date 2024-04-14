import pickle
import matplotlib.pyplot as plt
import torch as t
import numpy as np

from pathlib import Path



with open('results/covid_only_npis_wearing/rws_0_30_0.1_moments.pkl', 'rb') as f:
    psi = pickle.load(f)

models = ['covid', 'covid_only_wearing_mobility', 'covid_only_npis_wearing']
model = ['covid_only_npis_wearing']

for mod in models:
    #Get moments from file
    with open(f'results/{mod}/qem_0_30_0.1_moments.pkl', 'rb') as f:
        moments = pickle.load(f)
        
    #Get predictive samples from file
    with open(f'results/{mod}/qem_predictive_samples_30_0.1.pkl', 'rb') as f:
        predicted_obs = pickle.load(f)
        
    #Get standard deviation from moments

    means = moments['means'][-1]
    means2 = moments['means2'][-1]
    if mod == 'covid' or mod == 'covid_only_npis_wearing':
        std = np.sqrt(means2['CM_alpha'] - means['CM_alpha']**2)
    
    std_w = np.sqrt(means2['Wearing_alpha'] - means['Wearing_alpha']**2)

    if mod == 'covid' or mod == 'covid_only_wearing_mobility':
        std_m = np.sqrt(means2['Mobility_alpha'] - means['Mobility_alpha']**2)


    #NPIS
    NPIS = ['C1', 'C1_full', 'C2', 'C2_full', 'C4_3plus', 'C6', 'C7', 'C4_2plus', 'C4_full']

    #plot bar chart of moments, plotting CMs separately
    fig, ax = plt.subplots(1, 3, figsize=(15, 8))
    #Divide by number of samples (100)
    if mod == 'covid' or mod == 'covid_only_npis_wearing':
        ax[0].bar(np.arange(9), means['CM_alpha'], yerr=std)
    ax[0].set_title('CMs')
    ax[0].set_ylabel('Mean (Moment)')
    #hide xticks
    ax[0].set_xticks([])




    #plot bar chart of moments, plotting Wearing and Mobility separately
    if mod == 'covid' or mod == 'covid_only_wearing_mobility':
        ax[1].bar(np.arange(2), [means['Wearing_alpha'], means['Mobility_alpha']], yerr=[std_w, std_m])
        ax[1].set_title('Wearing and Mobility')
        ax[1].set_xticks(np.arange(2))
        ax[1].set_xticklabels(['Wearing', 'Mobility'])
    else:
        ax[1].bar(np.arange(1), [means['Wearing_alpha']], yerr=[std_w])
        ax[1].set_title('Wearing')
        ax[1].set_xticks(np.arange(1))
        ax[1].set_xticklabels(['Wearing'])
    
    NPIS = ['Some schools', 'All schools', 'Some business', 'All non-essential business', '11-1000 people', 'No leaving house', 'Internal movement restrictions', '101-1000 people', '-10 people']
    ax[0].set_xticks(np.arange(9))
    ax[0].set_xticklabels(NPIS)
    
    # #Rotate labels and set position
    for tick in ax[0].get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right')
    
    #plot predictive samples as timeseries

    #load obs
    obs = t.load('data/obs.pt').numpy()

    #plot
    #5 colors
    colors = ['b', 'g', 'r', 'c', 'm']

    ax[2].plot(predicted_obs['obs'].mean(0)[:1,:].T, color='k', label='Predicted data')
    ax[2].plot(obs[:1,:].T, linestyle='--', color='k', label='Real data')
    ax[2].set_title('Predicted vs Real data (Training)')
    ax[2].set_xlabel('Week')
    ax[2].set_ylabel('Observed cases')
    ax[2].legend()



    
    #make same plot using parameters from the last iteration
    #Get model parameters from file
    # params = t.load(f'results/{mod}/qem_0_30_0.1.pth', map_location='cpu')
    
    # CM_mean = params['Q._qem_means.CM_alpha_mean'].numpy()
    # CM_std = np.sqrt(params['Q._qem_means.CM_alpha_mean2'].numpy() - CM_mean**2)
    # #plot parameter means
    # ax[1,0].bar(np.arange(9), CM_mean, yerr=CM_std)
    # NPIS = ['Some schools', 'All schools', 'Some business', 'All non-essential business', '11-1000 people', 'No leaving house', 'Internal movement restrictions', '101-1000 people', '-10 people']
    # ax[1,0].set_xticks(np.arange(9))
    # ax[1,0].set_xticklabels(NPIS)
    # ax[1,0].set_ylabel('Mean (Parameter)')


    
    # #Rotate labels and set position
    # for tick in ax[1,0].get_xticklabels():
    #     tick.set_rotation(45)
    #     tick.set_ha('right')


    # #plot bar chart of moments, plotting Wearing and Mobility separately
    # wearing_mean = params['Q._qem_means.Wearing_alpha_mean'].numpy()
    # wearing_std = np.sqrt(params['Q._qem_means.Wearing_alpha_mean2'].numpy() - wearing_mean**2)
    
    # mobility_mean = params['Q._qem_means.Mobility_alpha_mean'].numpy()
    # mobility_std = np.sqrt(params['Q._qem_means.Mobility_alpha_mean2'].numpy() - mobility_mean**2)
    
    
    # ax[1,1].bar(np.arange(2), [wearing_mean, mobility_mean], yerr=[wearing_std, mobility_std])
    # ax[1,1].set_xticks(np.arange(2))
    
    
    # ax[1,1].set_xticklabels(['Wearing', 'Mobility'])
    # for tick in ax[1,1].get_xticklabels():
    #     tick.set_rotation(45)
    
    
    
    # ax[1][2].set_visible(False)

    # ax[1][0].set_position([0.24,0.125,0.228,0.343])
    # ax[1][1].set_position([0.55,0.125,0.228,0.343])

    Path(f"plots/{mod}/").mkdir(parents=True, exist_ok=True)
    plt.suptitle(mod)
    plt.tight_layout()
    plt.savefig(f'plots/{mod}/moments.png')
    


