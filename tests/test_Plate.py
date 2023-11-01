import torch as t
import torch.distributions as td
from functorch.dim import Dim

from alan_simplified import Normal, Gamma, Bernoulli, Categorical, MultivariateNormal, Plate, SingleSample, global2local_Kdims

plate = Plate(
    a = Normal(0, 1),
    b = Normal("a", 1),
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("a", 1)
    )
)

global_Kdim = Dim('K', 3)
globalK_sample = plate.sample(
    scope={},
    active_platedims=[],
    all_platedims={'p1': Dim('p1', 3)},
    sampling_type=SingleSample,
    Kdim=global_Kdim,
    reparam=False,
)

localK_sample = global2local_Kdims(globalK_sample, global_Kdim)
