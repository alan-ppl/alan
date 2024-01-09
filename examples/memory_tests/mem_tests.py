import torch as t
from alan import Normal, Plate, BoundPlate, Group, Problem, Data, mean, var, mean2, checkpoint, no_checkpoint, Split

import numpy as np

import pickle
import time 
t.manual_seed(0)

#Has to be cuda for memory profiling
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
t.cuda.set_device(device)

#Model
P = Plate( 
    mu = Normal(0, 1),
    p1 = Plate(
        theta = Normal('mu', 1),
        obs = Normal('theta', 1),
    ),
)

Q = Plate( 
    mu = Normal(0, 1),
    p1 = Plate(
        theta = Normal('mu', 1),
        obs = Data(),
    ),
)


#Plate sizes to try
p_sizes = [30,100,300,1000, 3000]
#Computation modes to try
comp_modes = {'no_checkpoint':no_checkpoint, 'checkpoint':checkpoint, 'Split=10':Split('p1', 10)}

Ks = [10,30,100,300, 1000, 3000]

no_Ps = len(p_sizes)
no_Ks = len(Ks)
#2D arrays (p_sizes x Ks) 
mem_usage = {'no_checkpoint': np.zeros((no_Ps, no_Ks)), 'checkpoint': np.zeros((no_Ps, no_Ks)), 'Split=10': np.zeros((no_Ps, no_Ks))}
time_usage = {'no_checkpoint': np.zeros((no_Ps, no_Ks)), 'checkpoint': np.zeros((no_Ps, no_Ks)), 'Split=10': np.zeros((no_Ps, no_Ks))}


for p_idx in range(no_Ps):
    for K_idx in range(no_Ks):
        p_size = p_sizes[p_idx]
        K = Ks[K_idx]
        for mode in comp_modes:
            
            
            print("Plate size: {}, Computation mode: {}, K: {}".format(p_size, mode, K))
            
            platesizes = {'p1': p_size}
            

            P_bound = BoundPlate(P, platesizes)
            Q_bound = BoundPlate(Q, platesizes)

            P_sample = P_bound.sample()
            data = {'obs': P_sample['obs']}
            
            prob = Problem(P_bound, Q_bound, data)
            prob.to(device=device)
            

            # Get some initial samples (with K dims)
            sample = prob.sample(K, True)

            #Compute moments from marginals
            try:
                for _ in range(10):
                    t.cuda.reset_peak_memory_stats()
                    start = time.time()
                    marginals = sample.marginals(computation_strategy=comp_modes[mode])
                    marginal_moments = marginals.moments([('theta', mean), ('theta', mean2), ('theta', mean2), ('theta', var)])
                    end = time.time()
                    mem_usage[mode][p_idx, K_idx] += t.cuda.max_memory_allocated()/10
                    time_usage[mode][p_idx, K_idx] += (end-start)/10
            except:
                mem_usage[mode][p_idx, K_idx] = float('nan')
                time_usage[mode][p_idx, K_idx] = float('nan')

with open('mem_usage.pkl', 'wb') as f:
    pickle.dump(mem_usage, f)
    
with open('time_usage.pkl', 'wb') as f:
    pickle.dump(time_usage, f)