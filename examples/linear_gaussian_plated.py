import torch as t
from alan import Normal, Plate, BoundPlate, Group, Problem, Data

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

platesizes = {'p1': 3}
data = {'obs': t.randn((3,2), names=('p1', None))}

P = BoundPlate(P, platesizes)
Q = BoundPlate(Q, platesizes, extra_opt_params={'mu_mean': t.zeros((2,)), 'psi_mean': t.zeros((2,))})

prob = Problem(P, Q, data)

K = 4
N = 100

sample = prob.sample(K, True)
print(sample.detached_sample)

print(sample.elbo_nograd())
print(sample.elbo_vi())
print(sample.elbo_rws())
breakpoint()

importance_sample = sample.importance_sample(N=10)
print(importance_sample.dump())

extended_platesizes = {'p1': 4}

extended_importance_sample = importance_sample.extend(extended_platesizes, None)
print(extended_importance_sample.dump())

extended_data = {'obs': t.randn((4,2), names=('p1', None))} 
ll = extended_importance_sample.predictive_ll(extended_data)
print(ll['obs'])