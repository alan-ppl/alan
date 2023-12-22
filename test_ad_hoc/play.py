import torch as t
from functorch.dim import Dim

from alan_simplified.TorchDimDist import TorchDimDist
import torch.distributions as td

Kdim = Dim('K', 3)
platedim = Dim('plate', 2)


print(platedim)
print(platedim.__repr__())
print(str(platedim))
# a = t.randn(3,2)[Kdim, platedim]
# print(a)
# tdd = TorchDimDist(td.Uniform, low=0, high=1)

# sample_dims = [Kdim, platedim]
# perm = tdd.sample(False, sample_dims=sample_dims, sample_shape=[]).argsort(Kdim)
# print(perm)

# perm_a = a.order(Kdim)[perm,...][Kdim]
# print(perm_a)
