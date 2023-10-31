import torch as t
import torch.distributions as td
from functorch.dim import Dim

from alan_simplified.TorchDimDist import TorchDimDist

d3 = Dim('d3', 3)
d4 = Dim('d4', 4)
d5 = Dim('d5', 5)

pt  = t.randn(())
p3  = t.randn(3)[d3]
p4  = t.randn(4)[d4]
p3_  = t.randn(3,6)[d3]
p34 = t.randn(3,4)[d3,d4]
p345 = t.randn(3,4,5)[d3,d4,d5]

tdd = TorchDimDist(td.Normal, loc=p3, scale=p4.exp())

sample = tdd.sample(True, (d3,d4,d5), [3,4])
assert set(sample.dims) == {d3,d4,d5}
assert sample.shape == t.Size([3,4])
