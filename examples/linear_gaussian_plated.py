import torch as t
from alan_simplified import Normal, Plate, BoundPlate, Group, Problem, IndependentSample, Data

P = Plate(
    mu = Normal(t.zeros((2,)), t.ones((2,))), 
    p1 = Plate(
        psi = Normal("mu", t.ones((2,))),
        obs = Normal("psi", t.ones((2,)))
    )
)
    
Q = Plate(
    mu = Normal("mu_mean", t.ones((2,))),
    p1 = Plate(
        psi = Normal("psi_mean", t.ones((2,))),
        obs = Data()
    )
)

P = BoundPlate(P)
Q = BoundPlate(Q, params={'mu_mean': t.zeros((2,)), 'psi_mean': t.zeros((2,))})

platesizes = {'p1': 3}
data = {'obs': t.randn((3,2), names=('p1', None))}
prob = Problem(P, Q, platesizes, data)

K = 4
num_samples = 100

sampling_type = IndependentSample
sample = prob.sample(K, True, sampling_type)
print(sample.sample)

extended_platesizes = {'p1': 4}
extended_data = {'obs': t.randn((4,2), names=('p1', None))} 
ll = sample.predictive_ll(extended_platesizes, True, extended_data, 10, {})
print(ll['obs'])