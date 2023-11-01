import torch as t
import torch.distributions as td
from functorch.dim import Dim

from alan_simplified import Normal, Gamma, Bernoulli, Categorical, MultivariateNormal, Plate, SingleSample

plate = Plate(
    a = Normal(0, 1),
    b = Normal("a", 1),
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("a", 1)
    )
)

Kdim = Dim('K', 3)
sample = plate.sample(
    scope={},
    active_platedims={'p1': Dim('p1', 3)},
    all_platedims={},
    sampling_type=SingleSample,
    Kdim=Kdim,
    reparam=False,
)


