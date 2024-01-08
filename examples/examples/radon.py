## Radon model in 919 houses and 85 counties from Gelman et al. (2006)
import torch as t
from alan import Normal, HalfNormal, Plate, BoundPlate, Group, Problem, Data, mean, Split, OptParam, QEMParam, checkpoint, no_checkpoint, Split

from posteriordb import PosteriorDatabase
import os

t.manual_seed(123)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
computation_strategy = checkpoint
    
pdb_path = os.path.join(os.getcwd(), "posteriordb/posterior_database")
my_pdb = PosteriorDatabase(pdb_path)

posterior = my_pdb.posterior("radon_mn-radon_pooled")

data = posterior.data.values()


#Number of Houses
Houses = data["N"]
#floor measurement
floor_measure = t.tensor(data["floor_measure"])
#log radon measurements
log_radon = t.tensor(data["log_radon"])

## Model ##
#As in stan...

P_plate = Plate( 
    alpha = Normal(0, 10),
    sigma_y = HalfNormal(1),
    beta = Normal(0, 10),
    Houses = Plate(
        log_radon = Normal(lambda alpha, floor_measure, beta: alpha + floor_measure * beta , 'sigma_y'),         
    ),
)

Q_plate = Plate( 
    alpha = Normal(OptParam(0.), OptParam(10.)),
    sigma_y = HalfNormal(OptParam(1.)),
    beta = Normal(OptParam(0.), OptParam(10.)),
    Houses = Plate(
        log_radon = Data(),         
    ),
)

all_platesizes = {'Houses': Houses}

data = {'log_radon': log_radon.rename('Houses')}
inputs = {'floor_measure': floor_measure.rename('Houses')}

P_bound_plate = BoundPlate(P_plate, all_platesizes, inputs=inputs)
Q_bound_plate = BoundPlate(Q_plate, all_platesizes)

prob = Problem(P_bound_plate, Q_bound_plate, data)
prob.to(device)

opt = t.optim.Adam(prob.parameters(), lr=1e-3)
        
K = 10
print("K={}".format(K))
for i in range(1000):
    opt.zero_grad()
    sample = prob.sample(K=K)
    elbo = sample.elbo_vi(computation_strategy=computation_strategy)
    elbo.backward()
    opt.step()

    if 0 == i%200:
        print(elbo.item())
        
gs = posterior.reference_draws_info()

import pandas as pd
gs = pd.DataFrame(gs)
print(gs)