import torch as t
from alan import Normal, HalfNormal, Bernoulli, Plate, BoundPlate, Group, Problem, Data, QEMParam, OptParam
from alan import Split, mean, mean2
from itertools import product

import pickle 
N = 20
z_mean = 33
z_var = 0.5
obs_var = 10


platesizes = {'plate_obs': N, 'plate_1': 20}
all_platesizes = {'plate_obs': N*2, 'plate_1': 20}
    
P = Plate(
        # mean = Normal(z_mean, z_var),
        logvar = Normal(0.,1.),
        plate_1 = Plate(
            obs_mean = Normal(0., lambda logvar: logvar.exp()),
            # logobs_var = Normal('mean', lambda logvar: logvar.exp()),
            plate_obs = Plate(
                obs = Normal('obs_mean', 1.)
            ),
        ),
    )

P = BoundPlate(P, platesizes)
latent_names = list(P.varname2groupvarname().keys())
latent_names.remove('obs')
moment_list = list(product(latent_names, [mean, mean2]))

Q = Plate(
        # mean = Normal(QEMParam(0.), QEMParam(1.)),
        logvar = Normal(QEMParam(0.), QEMParam(1.)),
        plate_1 = Plate(
            obs_mean = Normal(QEMParam(0.), QEMParam(1.)),
            # logobs_var = Normal(QEMParam(0.), QEMParam(1.)),
            plate_obs = Plate(
                obs = Data()
            ),
        ),
    )
Q = BoundPlate(Q, platesizes)

sample = P.sample()
all_data = {'obs': t.randn(N*2,20)}
all_data['obs'][:,1] += 10000

data = {'obs': all_data['obs'][:N,:]}

data_to_save = {data_name: data[data_name].numpy() for data_name in data}
all_data_to_save = {data_name: all_data[data_name].numpy() for data_name in all_data}
with open(f'fake_data.pkl', 'wb') as f:
    pickle.dump((platesizes, all_platesizes, data_to_save, all_data_to_save, {}, {}, latent_names), f)
            
all_data = {'obs': all_data['obs'].rename('plate_obs', 'plate_1')}

data = {'obs': all_data['obs'][:N, :].rename('plate_obs', 'plate_1')}
  
prob = Problem(P, Q, data)

K = 100
means = {name:[] for name in latent_names}
means2 = {name:[] for name in latent_names}
param_means = {name:[] for name in latent_names}
param_stds = {name:[] for name in latent_names}
for i in range(250):
    sample = prob.sample(K, reparam=False)
    elbo = sample.elbo_nograd()
    
    importance_sample = sample.importance_sample(N=N)
    extended_importance_sample = importance_sample.extend(all_platesizes)
    ll = extended_importance_sample.predictive_ll(all_data)
    if i % 10 == 0:
        print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")
    sample.update_qem_params(0.1)
    
    #save params
    for name in latent_names:
        m = sample.moments([(name, mean), (name, mean2)])
        param_means[name].append(m[0].rename(None))
        param_stds[name].append(m[1].rename(None) - m[0].rename(None)**2)
        
    for _ in range(1):
        sample = prob.sample(K, reparam=False)
        
        
        m = sample.moments(moment_list)
        temp_means = [m[i] for i in range(0,len(latent_names)*2,2)]
        temp_means2 = [m[i] for i in range(1,len(latent_names)*2,2)]
        for name, mean_val in zip(latent_names, temp_means):
            means[name].append(mean_val.rename(None))
        for name, mean_val in zip(latent_names, temp_means2):
            means2[name].append(mean_val.rename(None))
   
   
means = {name: t.stack(means[name], 0).numpy() for name in latent_names}         
# overall_mean = {}
# for name in latent_names:
#     overall_mean[name] = t.stack(means[name][-100:], 0).mean(0)
#     # means2[name] = t.stack(means2[name][-100:], 0).mean(0)
        
#take average of last 100 numpy moments



#save means 
with open('means.pkl', 'wb') as f:
    pickle.dump(means, f)

import matplotlib.pyplot as plt
#plot avg param values
fig, ax = plt.subplots(4, 2, figsize=(10,10))
ax[0,0].set_title('Means')
ax[0,1].set_title('Variances')
for i, name in enumerate(latent_names):
    ax[i,0].plot(t.stack(param_means[name], 0).reshape(50,-1).mean(-1).numpy(), label=name)
    ax[i,1].plot(t.stack(param_stds[name], 0).reshape(50,-1).mean(-1).numpy(), label=name)
    ax[i,1].legend()


plt.show()
