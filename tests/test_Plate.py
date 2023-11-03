import torch as t
import torch.distributions as td
from functorch.dim import Dim

from alan_simplified import Normal, Gamma, Bernoulli, Categorical, MultivariateNormal, Plate, SingleSample, global2local_Kdims

# plate = Plate(
#     a = Normal(0, 1),
#     b = Normal("a", 1),
#     c = Normal(0, lambda a: a.exp()),
#     p1 = Plate(
#         d = Normal("a", 1),
#         p2 = Plate(
#             e = Normal("d", "inp"),
#         ),
#     ),
# )
# scope = {}
# active_platedims = []
# all_platedims = {'p1': Dim('p1', 3), 'p2': Dim('p2', 4)}
# sampling_type = SingleSample
# global_Kdim = Dim('K', 3)

# globalK_sample = plate.sample(
#     scope=scope,
#     active_platedims=active_platedims,
#     all_platedims=all_platedims,
#     sampling_type=SingleSample,
#     Kdim=global_Kdim,
#     reparam=False,
# )

# localK_sample, groupvarname2Kdim = global2local_Kdims(globalK_sample, global_Kdim)

# lp = plate.log_prob(
#     sample=localK_sample,
#     scope=scope,
#     active_platedims=active_platedims,
#     all_platedims=all_platedims,
#     sampling_type=SingleSample,
#     groupvarname2Kdim=groupvarname2Kdim,
# )
