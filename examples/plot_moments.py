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

models = ['movielens', 'bus_breakdown','radon']# 'occupancy', 'occupancy_reparam']# 'covid']# 'bus_breakdown']# 'covid']# 'radon', 'chimpanzees']

model_names = {'movielens': 'MOVIELENS', 'bus_breakdown': "BUS BREAKDOWN", 'radon': 'RADON'}
# models = ['bus_breakdown']
paths = {'movielens': ['experiments/moments/movielens/qem_0_30_0.1_moments.pkl', 'experiments/moments/movielens/rws_0_30_0.03_moments.pkl', 'experiments/moments/movielens/vi_0_30_0.03_moments.pkl', 'experiments/results/movielens/blackjax_moments0.pkl', 'experiments/results/movielens/blackjax0100.pkl', 'experiments/results/movielens/blackjax0300.pkl', 'experiments/results/movielens/blackjax01000.pkl'],
         'bus_breakdown': ['experiments/moments/bus_breakdown/qem_0_30_0.1_moments.pkl', 'experiments/moments/bus_breakdown/rws_0_30_0.03_moments.pkl', 'experiments/moments/bus_breakdown/vi_0_30_0.03_moments.pkl', 'experiments/results/bus_breakdown/blackjax_moments0.pkl', 'experiments/results/bus_breakdown/blackjax0100.pkl', 'experiments/results/bus_breakdown/blackjax0300.pkl', 'experiments/results/bus_breakdown/blackjax01000.pkl'],
         'radon': ['experiments/moments/radon/qem_0_30_0.03_moments.pkl', 'experiments/moments/radon/rws_0_30_0.03_moments.pkl', 'experiments/moments/radon/vi_0_30_0.03_moments.pkl', 'experiments/results/radon/blackjax_moments0.pkl', 'experiments/results/radon/blackjax0100.pkl', 'experiments/results/radon/blackjax0300.pkl', 'experiments/results/radon/blackjax01000.pkl']}
#reparam paths 
# paths = {'movielens': ['experiments/moments/movielens_reparam/qem_0_30_0.1_moments.pkl', 'experiments/moments/movielens_reparam/rws_0_30_0.03_moments.pkl', 'experiments/moments/movielens_reparam/vi_0_30_0.03_moments.pkl', 'experiments/results/movielens/blackjax_moments0.pkl'],
#          'bus_breakdown': ['experiments/moments/bus_breakdown_reparam/qem_0_30_0.1_moments.pkl', 'experiments/moments/bus_breakdown_reparam/rws_0_30_0.1_moments.pkl', 'experiments/moments/bus_breakdown_reparam/vi_0_30_0.1_moments.pkl', 'experiments/results/bus_breakdown_reparam/blackjax_moments0.pkl'],
#             'radon': ['experiments/moments/radon_reparam/qem_0_30_0.1_moments.pkl', 'experiments/moments/radon_reparam/rws_0_30_0.1_moments.pkl', 'experiments/moments/radon_reparam/vi_0_30_0.1_moments.pkl', 'experiments/results/radon_reparam/blackjax_moments0.pkl'],
#             'chimpanzees': ['experiments/moments/chimpanzees_reparam/qem_0_30_0.1_moments.pkl', 'experiments/moments/chimpanzees_reparam/rws_0_30_0.1_moments.pkl', 'experiments/moments/chimpanzees_reparam/vi_0_30_0.1_moments.pkl', 'experiments/results/chimpanzees_reparam/blackjax_moments0.pkl'],
#             'covid': ['experiments/moments/covid_reparam/qem_0_30_0.1_moments.pkl', 'experiments/moments/covid_reparam/rws_0_30_0.1_moments.pkl', 'experiments/moments/covid_reparam/vi_0_30_0.1_moments.pkl', 'experiments/results/covid_reparam/blackjax_moments0.pkl']}
         
# Three colours for each method
colours = ['#e7298a', '#1b9e77', '#d95f02', '#000000']

fig, ax = plt.subplots(1, len(models), figsize=(4*len(models), 4))

pred_ll_fig, pred_ll_ax = plt.subplots(1, len(models), figsize=(4*len(models), 4))
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



    # if 'bus' in model:
    #     print(HMC['alpha'].mean(0))
    #     print(QEM_moments['means']['alpha'])
    #     HMC['alpha'] = HMC['alpha'].transpose(0,2,1)
        
        
    if model == 'movielens_reparam':
        QEM_moments['means']['z'] = QEM_moments['means']['z'] * 100
        RWS_moments['means']['z'] = RWS_moments['means']['z'] * 100
        VI_moments['means']['z'] = VI_moments['means']['z'] * 100
        
    if model == 'bus_breakdown_reparam':
        QEM_moments['means']['alpha'] = QEM_moments['means']['alpha'] * 1000
        RWS_moments['means']['alpha'] = RWS_moments['means']['alpha'] * 1000
        VI_moments['means']['alpha'] = VI_moments['means']['alpha'] * 1000
        
    
    if model == 'radon_reparam':
        QEM_moments['means']['County_mean'] = QEM_moments['means']['County_mean'] * 100
        RWS_moments['means']['County_mean'] = RWS_moments['means']['County_mean'] * 100
        VI_moments['means']['County_mean'] = VI_moments['means']['County_mean'] * 100
        
        QEM_moments['means']['Beta_u'] = QEM_moments['means']['Beta_u'] * 10
        RWS_moments['means']['Beta_u'] = RWS_moments['means']['Beta_u'] * 10
        VI_moments['means']['Beta_u'] = VI_moments['means']['Beta_u'] * 10
        
        QEM_moments['means']['Beta_basement'] = QEM_moments['means']['Beta_basement'] * 1000
        RWS_moments['means']['Beta_basement'] = RWS_moments['means']['Beta_basement'] * 1000
        VI_moments['means']['Beta_basement'] = VI_moments['means']['Beta_basement'] * 1000
        
        
    HMC_short = {key: HMC[key] for key in HMC}
    
    HMC_means = {key: HMC[key].mean(0) for key in HMC_short}
 
    QEM_diffs = {key: [] for key in QEM_moments['means'].keys()}
    for key in QEM_diffs.keys():
        diff = 0
        diff += ((HMC_means[key] - QEM_moments['means'][key])**2).mean(axis=tuple(range(1, QEM_moments['means'][key].ndim)))
        QEM_diffs[key] = diff


    RWS_diffs = {key: [] for key in RWS_moments['means'].keys()}

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
        HMC100 = pickle.load(f)
    
    with open(f'{pth[5]}', 'rb') as f:
        HMC300 = pickle.load(f)
        
    with open(f'{pth[6]}', 'rb') as f:
        HMC1000 = pickle.load(f)
        
    # print(HMC['p_lls'])
    # times
    # if model == 'radon_reparam' or model == 'radon' or model == 'bus_breakdown_reparam' or model == 'bus_breakdown':
    QEM_times = QEM['iter_times'].mean(-1)[0][2]
    RWS_times = RWS['iter_times'].mean(-1)[0][2]
    VI_times = VI['iter_times'].mean(-1)[0][2]
    HMC_times100 = HMC100['times']['moments'].mean(1)
    HMC_times300 = HMC300['times']['moments'].mean(1)
    HMC_times1000 = HMC1000['times']['moments'].mean(1)
    
    QEM_plls = QEM['p_lls'].mean(-1)[0][2]
    RWS_plls = RWS['p_lls'].mean(-1)[0][2]
    VI_plls = VI['p_lls'].mean(-1)[0][2]
    
    HMC100['p_lls'] = np.logaddexp.accumulate(HMC100['p_lls']) - np.log(np.arange(1, HMC100['p_lls'].shape[0]+1)[:, None])
    HMC300['p_lls'] = np.logaddexp.accumulate(HMC300['p_lls'])  - np.log(np.arange(1, HMC100['p_lls'].shape[0]+1)[:, None])
    HMC1000['p_lls'] = np.logaddexp.accumulate(HMC1000['p_lls'])  - np.log(np.arange(1, HMC100['p_lls'].shape[0]+1)[:, None])
    
    HMC_plls100 = HMC100['p_lls'].mean(1) 
    HMC_plls300 = HMC300['p_lls'].mean(1)
    HMC_plls1000 = HMC1000['p_lls'].mean(1)
    
    

    
    #all to numpy arrays
    QEM_times = QEM_times.numpy()
    RWS_times = RWS_times.numpy()
    VI_times = VI_times.numpy()
    
    QEM_plls = QEM_plls.numpy()
    RWS_plls = RWS_plls.numpy()
    VI_plls = VI_plls.numpy()
    
    
    #Cumulative sums
    QEM_times = np.cumsum(QEM_times)
    RWS_times = np.cumsum(RWS_times)
    VI_times = np.cumsum(VI_times)
    
    #to lists
    QEM_times = QEM_times.tolist()
    RWS_times = RWS_times.tolist()
    VI_times = VI_times.tolist()
    HMC_times100 = HMC_times100.tolist()
    HMC_times300 = HMC_times300.tolist()
    HMC_times1000 = HMC_times1000.tolist()
    

    #plot
    #Times
    methods = ['QEM', 'RWS', 'VI', 'HMC']

    QEM_diff = sum(QEM_diffs.values())
    RWS_diff = sum(RWS_diffs.values())
    VI_diff = sum(VI_diffs.values())
    
    #apply some smoothing to the differences
    QEM_diff = smooth(QEM_diff, 30)
    RWS_diff = smooth(RWS_diff, 30)
    VI_diff = smooth(VI_diff, 30)
    # for key in QEM_diffs.keys():
    ax[models.index(model)].plot(QEM_times, QEM_diff, label=f'QEM', color=colours[0])
    ax[models.index(model)].plot(RWS_times, RWS_diff, label=f'RWS', color=colours[1])
    ax[models.index(model)].plot(VI_times, VI_diff, label=f'VI', color=colours[2])
    ax[models.index(model)].set_xlabel('Time (s)')
    
    #ylim
    ax[models.index(model)].set_ylim([max(QEM_diff)-np.var(QEM_diff), max(QEM_diff)+np.var(QEM_diff)])

    
    ax[models.index(model)].set_title(model_names[model])
    ax[0].set_ylabel('Mean squared error')

    
    QEM_plls = smooth(QEM_plls, 30)
    RWS_plls = smooth(RWS_plls, 30)
    VI_plls = smooth(VI_plls, 30)


    pred_ll_ax[models.index(model)].plot(QEM_times, QEM_plls, label='QEM', color=colours[0])
    pred_ll_ax[models.index(model)].plot(RWS_times, RWS_plls, label='RWS', color=colours[1])
    pred_ll_ax[models.index(model)].plot(VI_times, VI_plls, label='VI', color=colours[2])
    pred_ll_ax[models.index(model)].plot(HMC_times100, HMC_plls100, label='HMC 100', color=colours[3], alpha = 0.2)
    pred_ll_ax[models.index(model)].plot(HMC_times300, HMC_plls300, label='HMC 300', color=colours[3], alpha = 0.5)
    pred_ll_ax[models.index(model)].plot(HMC_times1000, HMC_plls1000, label='HMC 1000', color=colours[3])

    pred_ll_ax[models.index(model)].set_ylim([min(QEM_plls) - 50, None])
    pred_ll_ax[models.index(model)].set_title(model_names[model])
    pred_ll_ax[0].set_ylabel('Predictive log likelihood')
    pred_ll_ax[models.index(model)].set_xlabel('Time (s)')
    
    
    
    

#add legend
ax[len(models)-1].legend()
pred_ll_ax[len(models)-1].legend()

fig.savefig('moments.pdf', dpi=1200, bbox_inches='tight')
fig.savefig('moments.png', bbox_inches='tight')
    

pred_ll_fig.savefig('pred_lls.pdf', dpi=1200, bbox_inches='tight')
pred_ll_fig.savefig('pred_lls.png', bbox_inches='tight')
