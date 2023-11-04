import torch as t
import torch.distributions as td
from functorch.dim import Dim

from alan_simplified import Normal, Bernoulli, Plate, Group, Problem, IndependentSample

P = Plate(
    ab = Group(
        a = Normal(0, 1),
        b = Normal("a", 1),
    ),
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("a", 1),
        p2 = Plate(
            e = Normal("d", "inp"),
        ),
    ),
)

Q = Plate(
    ab = Group(
        a = Normal(0, 1),
        b = Normal("a", 1),
    ),
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("a", 1),
        p2 = Plate(
        ),
    ),
)

all_platesizes = {'p1': 3, 'p2': 4}
data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

prob = Problem(P, Q, all_platesizes, data)

samp = prob.sample(3, True, IndependentSample)
