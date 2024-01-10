from .Plate import Plate, tensordict2tree, flatten_tree, empty_tree
from .utils import *
from .moments import RawMoment, torchdim_moments_mixin, named_moments_mixin

class SampleNonMP:
    def __init__(
            problem,
            sample,
            groupvarname2Kdim,
            reparam,
        ):
        
        self.problem = problem
        self.groupvarname2Kdim = groupvarname2Kdim
        self.sampler = sampler
        self.reparam = reparam

        if self.reparam:
            self.reparam_sample = sample
            self.detached_sample = detach_dict(sample)
        else:
            self.detached_sample = sample

