###
# From Bayesian Data Analysis, section 5.5 (Gelman et al. 2013):

# A study was performed for the Educational Testing Service to analyze the effects of special coaching programs for SAT-V 
# (Scholastic Aptitude Test-Verbal) in each of eight high schools. The outcome variable in each study was the score on a special 
# administration of the SAT-V, a standardized multiple choice test administered by the Educational Testing Service and used to 
# help colleges make admissions decisions; the scores can vary between 200 and 800, with mean about 500 and standard deviation 
# about 100. The SAT examinations are designed to be resistant to short-term efforts directed specifically toward improving 
# performance on the test; instead they are designed to reflect knowledge acquired and abilities developed over many years of 
# education. Nevertheless, each of the eight schools in this study considered its short-term coaching program to be very 
# successful at increasing SAT scores. Also, there was no prior reason to believe that any of the eight programs was more 
# effective than any other or that some were more similar in effect to each other than to any other.
###

import torch as t
from alan import Normal, HalfCauchy, Plate, BoundPlate, Group, Problem, Data, mean, Split, OptParam, QEMParam, checkpoint, no_checkpoint, Split

from posteriordb import PosteriorDatabase
import os

t.manual_seed(123)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
computation_strategy = checkpoint
    
pdb_path = os.path.join(os.getcwd(), "posteriordb/posterior_database")
my_pdb = PosteriorDatabase(pdb_path)

posterior = my_pdb.posterior("eight_schools-eight_schools_centered")

data = posterior.data.values()

#Number of schools
J = data["J"]
#Treatment effects
y = t.tensor(data["y"])
#Standard errors
sigma = t.tensor(data["sigma"])

## Model ##
#As in stan...

P_plate = Plate( 
    tau = HalfCauchy(5),
    mu = Normal(0, 5),
    J = Plate(
        theta = Normal('mu', 'tau'),
        y = Normal('theta', 'sigma'),
    ),   
)

Q_plate = Plate( 
    tau = HalfCauchy(OptParam(5.)),
    mu = Normal(OptParam(0.), OptParam(5.)),
    J = Plate(
        theta = Normal(OptParam(0.), OptParam(5.)),
        y = Data(),
    ),   
)

all_platesizes = {'J': J}

data = {'y': y.rename('J')}
inputs = {'sigma': sigma.rename('J')}

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
        
gs = posterior.reference_draws()

import pandas as pd
gs = pd.DataFrame(gs)
print(gs)