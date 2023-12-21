import torch as t
import torch.distributions as td
from functorch.dim import Dim

from alan_simplified import Normal, Bernoulli, Plate, BoundPlate, Group, Problem, IndependentSample, Data, Mean, Var, Split

t.manual_seed(0)


P = Plate(
    ab = Group(
        a = Normal(0, 1),
        b = Normal("a", 1),
    ),
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("a", 1),
        p2 = Plate(
            e = Normal("d", 1.),
        ),
    ),
)

Q = Plate(
    ab = Group(
        a = Normal("a_mean", 1),
        b = Normal("a", 1),
    ),
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("d_mean", 1),
        p2 = Plate(
            e = Data()
        ),
    ),
)

P = BoundPlate(P)
Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})

all_platesizes = {'p1': 3, 'p2': 4}
data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

prob = Problem(P, Q, all_platesizes, data)

sampling_type = IndependentSample
sample = prob.sample(3, True, sampling_type)

marginals = sample.marginals(("ab", "c"))
importance_sample = sample.importance_sample(1000)

print(marginals.moments("a", Mean))
print(importance_sample.moments("a", Mean))

print(sample.elbo())
print(sample.elbo(split=Split('p1', 2)))
