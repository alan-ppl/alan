import inspect
import math

import torch as t
import functorch
from functorch.dim import Dim
from torch.utils.checkpoint import checkpoint
import torch.distributions as td

from alan_simplified.TorchDimDist import TorchDimDist
from alan_simplified.utils import generic_dims, logmeanexp_dims, logsumexp_dims
from alan_simplified.reduce_Ks import reduce_Ks, sample_Ks

Kdim = Dim('K')


parent_1_Kdim = Dim('parent_1_K')
parent_2_Kdim = Dim('parent_2_K')
plate_1 = Dim('plate_1')
plate_2 = Dim('plate_2')

sizes = {'K': 2, 'parent_1_K': 3, 'parent_2_K': 4}



lps = [t.randn(2,3,4,5)[Kdim, parent_1_Kdim, parent_2_Kdim, plate_1], t.randn(2,3,5)[Kdim, parent_1_Kdim, plate_1], t.randn(2,4,5)[Kdim, parent_2_Kdim, plate_1]]

print(sample_Ks(lps, [Kdim, parent_1_Kdim, parent_2_Kdim], num_samples=10))

# for k,v in sample_Ks(lps, [Kdim, parent_1_Kdim, parent_2_Kdim], num_samples=10).items():
#     assert v.size()[0] == 10
#     assert int(v.order(*v.dims).max()) <= sizes[k]
    
active_platedims = [plate_1, plate_2]

lps = [t.randn(2,3,4,5,6)[Kdim, parent_1_Kdim, parent_2_Kdim, plate_1, plate_2], t.randn(2,3,5,6)[Kdim, parent_1_Kdim, plate_1, plate_2], t.randn(2,4,5,6)[Kdim, parent_2_Kdim, plate_1, plate_2]]

print(sample_Ks(lps, [Kdim, parent_1_Kdim, parent_2_Kdim], num_samples=10))

# for k,v in sample_Ks(lps, [Kdim, parent_1_Kdim, parent_2_Kdim], num_samples=10).items():
#     assert v.size()[0] == 10
#     assert int(v.order(*v.dims).max()) <= sizes[k]