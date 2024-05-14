import pickle
import matplotlib.pyplot as plt
import torch as t
import numpy as np

from pathlib import Path


def smooth(x, window):
    # result = np.convolve(x, np.ones(window)/window, mode='valid')
    
    result = np.zeros_like(x)

    result[0] = x[0]

    for i in range(1,len(x)):
        if x[i] != np.nan:
            result[i] = x[max(i-window, 0):i].mean()
        # result[i,:] = np.nanmean(x[max(i-window, 0):i,:], 1)

    return result

models = ['movielens', 'radon', 'bus_breakdown', 'movielens_reparam', 'bus_breakdown_reparam']# 'covid']# 'bus_breakdown']# 'covid']# 'radon', 'chimpanzees']

paths = {'movielens': ['experiments/moments/movielens/qem_0_30_0.1_moments.pkl', 'experiments/moments/movielens/rws_0_30_0.03_moments.pkl', 'experiments/moments/movielens/vi_0_30_0.03_moments.pkl', 'experiments/results/movielens/blackjax_moments0.pkl', 'experiments/results/movielens/blackjax0.pkl'],
         'bus_breakdown': ['experiments/moments/bus_breakdown/qem_0_30_0.03_moments.pkl', 'experiments/moments/bus_breakdown/rws_0_30_0.1_moments.pkl', 'experiments/moments/bus_breakdown/vi_0_30_0.1_moments.pkl', 'experiments/results/bus_breakdown/blackjax_moments0.pkl', 'experiments/results/bus_breakdown/blackjax0.pkl'],
         'radon': ['experiments/moments/radon/qem_0_30_0.1_moments.pkl', 'experiments/moments/radon/rws_0_30_0.1_moments.pkl', 'experiments/moments/radon/vi_0_30_0.1_moments.pkl', 'experiments/results/radon/blackjax_moments0.pkl', 'experiments/results/radon/blackjax0.pkl'],
         'chimpanzees': ['experiments/moments/qem_0_30_0.1_moments.pkl', 'experiments/moments/rws_0_30_0.1_moments.pkl', 'experiments/moments/vi_0_30_0.1_moments.pkl', 'experiments/results/moments/blackjax_moments0.pkl', 'experiments/results/moments/blackjax0.pkl'],
         'covid': ['experiments/moments/covid/qem_0_30_0.1_moments.pkl', 'experiments/moments/covid/rws_0_30_0.1_moments.pkl', 'experiments/moments/covid/vi_0_30_0.1_moments.pkl', 'experiments/results/covid/blackjax_moments0.pkl', 'experiments/results/covid/blackjax0.pkl'],
         'movielens_reparam': ['experiments/moments/movielens_reparam/qem_0_30_0.1_moments.pkl', 'experiments/moments/movielens_reparam/rws_0_30_0.03_moments.pkl', 'experiments/moments/movielens_reparam/vi_0_30_0.03_moments.pkl', 'experiments/results/movielens/blackjax_moments0.pkl', 'experiments/results/movielens/blackjax0.pkl'],
         'bus_breakdown_reparam': ['experiments/moments/bus_breakdown_reparam/qem_0_30_0.03_moments.pkl', 'experiments/moments/bus_breakdown_reparam/rws_0_30_0.1_moments.pkl', 'experiments/moments/bus_breakdown_reparam/vi_0_30_0.1_moments.pkl', 'experiments/results/bus_breakdown/blackjax_moments0.pkl', 'experiments/results/bus_breakdown/blackjax0.pkl'],}

#reparam paths 
# paths = {'movielens': ['experiments/moments/movielens_reparam/qem_0_30_0.1_moments.pkl', 'experiments/moments/movielens_reparam/rws_0_30_0.03_moments.pkl', 'experiments/moments/movielens_reparam/vi_0_30_0.03_moments.pkl', 'experiments/results/movielens/blackjax_moments0.pkl'],
#          'bus_breakdown': ['experiments/moments/bus_breakdown_reparam/qem_0_30_0.1_moments.pkl', 'experiments/moments/bus_breakdown_reparam/rws_0_30_0.1_moments.pkl', 'experiments/moments/bus_breakdown_reparam/vi_0_30_0.1_moments.pkl', 'experiments/results/bus_breakdown_reparam/blackjax_moments0.pkl'],
#             'radon': ['experiments/moments/radon_reparam/qem_0_30_0.1_moments.pkl', 'experiments/moments/radon_reparam/rws_0_30_0.1_moments.pkl', 'experiments/moments/radon_reparam/vi_0_30_0.1_moments.pkl', 'experiments/results/radon_reparam/blackjax_moments0.pkl'],
#             'chimpanzees': ['experiments/moments/chimpanzees_reparam/qem_0_30_0.1_moments.pkl', 'experiments/moments/chimpanzees_reparam/rws_0_30_0.1_moments.pkl', 'experiments/moments/chimpanzees_reparam/vi_0_30_0.1_moments.pkl', 'experiments/results/chimpanzees_reparam/blackjax_moments0.pkl'],
#             'covid': ['experiments/moments/covid_reparam/qem_0_30_0.1_moments.pkl', 'experiments/moments/covid_reparam/rws_0_30_0.1_moments.pkl', 'experiments/moments/covid_reparam/vi_0_30_0.1_moments.pkl', 'experiments/results/covid_reparam/blackjax_moments0.pkl']}
         
# Three colours for each method
colours = ['b', 'r', 'g']

fig, ax = plt.subplots(2, len(models), figsize=(15, 4*len(models)))

# with open(f'movielensexperiments/moments/moments/HMC0.pkl', 'rb') as f:
#     hmc = pickle.load(f)
    

for model in models:
    print(model)
    pth = paths[model]
    with open(f'{pth[0]}', 'rb') as f:
        QEM_moments = pickle.load(f)

    with open(f'{pth[1]}', 'rb') as f:
        RWS_moments = pickle.load(f)

    with open(f'{pth[2]}', 'rb') as f:
        VI_moments = pickle.load(f)

    with open(f'{pth[3]}', 'rb') as f:
        HMC = pickle.load(f)


    # use the first 200 HMC iterations
    # for key in HMC.keys():
    #     print(key)
    #     print(HMC[key].shape)
        
    HMC_short = {key: HMC[key] for key in HMC}
    
    HMC_means = {key: HMC[key].mean(0) for key in HMC_short}
 
    QEM_diffs = {key: [] for key in QEM_moments['means'].keys()}
    for key in QEM_diffs.keys():
        diff = 0
        diff += ((HMC_means[key] - QEM_moments['means'][key])**2).mean(axis=tuple(range(1, QEM_moments['means'][key].ndim)))
        QEM_diffs[key] = diff


    RWS_diffs = {key: [] for key in RWS_moments['means'].keys()}
    if model == 'bus_breakdown_reparam':
        print(RWS_moments['means']['alpha'])
    for key in RWS_diffs.keys():
        diff = 0
        diff += ((HMC_means[key] - RWS_moments['means'][key])**2).mean(axis=tuple(range(1, RWS_moments['means'][key].ndim)))
        RWS_diffs[key] = diff


    VI_diffs = {key: [] for key in VI_moments['means'].keys()}
    for key in VI_diffs.keys():
        diff = 0
        diff += ((HMC_means[key] - VI_moments['means'][key])**2).mean(axis=tuple(range(1, VI_moments['means'][key].ndim)))
        VI_diffs[key] = diff
        
    HMC_diffs = {key: [] for key in HMC_short}

    
    for key in HMC_short.keys():
        diff = 0
        diff += ((HMC_means[key] - HMC_short[key])**2).mean(axis=tuple(range(1, HMC_short[key].ndim)))
        HMC_diffs[key] = diff


    with open(f'experiments/results/{model}/qem0.pkl', 'rb') as f:
        QEM = pickle.load(f)
    
    with open(f'experiments/results/{model}/rws0.pkl', 'rb') as f:
        RWS = pickle.load(f)
    
    with open(f'experiments/results/{model}/vi0.pkl', 'rb') as f:
        VI = pickle.load(f)
    
    with open(f'{pth[4]}', 'rb') as f:
        HMC = pickle.load(f)
        
    # print(HMC['p_lls'])
    # times
    print(QEM.keys())
    QEM_times = QEM['iter_times'][0][0].mean(1)
    RWS_times = RWS['iter_times'][0][0].mean(1)
    VI_times = VI['iter_times'][0][0].mean(1)
    HMC_times = HMC['times']['moments'].mean(1)
    
    #Cumulative sums
    QEM_times = np.cumsum(QEM_times)
    RWS_times = np.cumsum(RWS_times)
    VI_times = np.cumsum(VI_times)
    HMC_times = HMC_times
    
    #to lists
    QEM_times = QEM_times.tolist()
    RWS_times = RWS_times.tolist()
    VI_times = VI_times.tolist()
    HMC_times = HMC_times.tolist()
    

    #plot
    #Times
    methods = ['QEM', 'RWS', 'VI', 'HMC']

    QEM_diff = sum(QEM_diffs.values())
    RWS_diff = sum(RWS_diffs.values())
    VI_diff = sum(VI_diffs.values())
    
    #apply some smoothing to the differences
    QEM_diff = smooth(QEM_diff, 10)
    RWS_diff = smooth(RWS_diff, 10)
    VI_diff = smooth(VI_diff, 10)
    # for key in QEM_diffs.keys():
    ax[0,models.index(model)].plot(QEM_times, QEM_diff, label=f'QEM', color=colours[0])
    ax[0,models.index(model)].plot(RWS_times, RWS_diff, label=f'RWS', color=colours[1])
    ax[0,models.index(model)].plot(VI_times, VI_diff, label=f'VI', color=colours[2])
    
    #ylim
    ax[0,models.index(model)].set_ylim([0, QEM_diff[0]+0.5])

    
    ax[0,models.index(model)].set_title(model)
    ax[0,0].set_ylabel('MSD')
    ax[1,models.index(model)].set_xlabel('Time (s)')
    #plot time bar charts
    ax[1,models.index(model)].bar(methods, [QEM_times[-1], RWS_times[-1], VI_times[-1], HMC_times[-1]], color=colours)
    # ax[1,models.index(model)].set_title('Total time')
    ax[1,0].set_ylabel('Time (s)')
    #label each bar
    for i, v in enumerate([QEM_times[-1], RWS_times[-1], VI_times[-1], HMC_times[-1]]):
        ax[1,models.index(model)].text(i, v + 0.1, str(round(v, 2)), color='black', ha='center')
    
    
    
    
    

#add legend
ax[0, len(models)-1].legend()

plt.suptitle('Mean difference between HMC moment estimates and other methods')
plt.savefig('moments.png')
    
    

