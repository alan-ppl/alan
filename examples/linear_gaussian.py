import torch as t
import functorch.dim
from functorch.dim import Dim, dims

from alan_simplified import Normal, Plate, BoundPlate, Group, Problem, IndependentSample
from alan_simplified.IndexedSample import IndexedSample

t.manual_seed(127)

P = Plate(mu = Normal(t.zeros((2,)), t.ones((2,))), 
                    p1 = Plate(obs = Normal("mu", t.ones((2,)))))
    
Q = Plate(mu = Normal("mu_mean", t.ones((2,))),
            p1 = Plate())

Q = BoundPlate(Q, params={'mu_mean': t.zeros((2,))})
platesizes = {'p1': 3}
data = {'obs': t.randn((3,2), names=('p1', None))}
prob = Problem(P, Q, platesizes, data)

K = 100
num_samples = 100

sampling_type = IndependentSample
sample = prob.sample(K, True, sampling_type)
post_idxs = sample.sample_posterior(num_samples=num_samples)
isample = IndexedSample(sample, post_idxs)


extended_data = t.tensor([0,1,2,3])
extended_platesizes = {'p1': 4}
extended_data = {'obs': extended_data.refine_names('p1')} 
ll = isample.predictive_ll(prob.P, extended_platesizes, True, extended_data)
print(ll['obs'])