import torch as t
import torch.distributions as td
from functorch.dim import Dim

from alan_simplified import Normal, Bernoulli, Plate, BoundPlate, Group, Problem, IndependentSample

P = Plate(
    a = Normal(0, 1),
    b = Normal("a", 1),
    
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("a", 1),
        p2 = Plate(
            e = Normal("d", 1.),
        ),
    ),
)

Q = Plate(
    a = Normal("a_mean", 1),
    b = Normal("a", 1),
    
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("d_mean", 1),
        p2 = Plate(
        ),
    ),
)
Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})

all_platesizes = {'p1': 3, 'p2': 4}
data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

prob = Problem(P, Q, all_platesizes, data)

sampling_type = IndependentSample
sample = prob.sample(3, True, sampling_type)
L = sample.elbo()

# marginals = sample.marginals()
# conditionals = sample.conditionals()

print(L)
# print(marginals)
# print(conditionals)

post_idxs = sample.sample_posterior(num_samples=10)
print(post_idxs)


