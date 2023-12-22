from .utils import *
from .moments import user_facing_moments_mixin

class Marginals:
    def __init__(
            self, 
            samples:dict[str, Tensor], 
            weights:dict[tuple[str], Tensor], 
            all_platedims: dict[str, Dim],
            varname2groupvarname: dict[str, tuple[str]]):
        """
        samples and weights are flat dicts of torchdim Tensors.
        
        But there's some subtlety as to the keys.
        samples is indexed by a single varname.
        weights is indexed by frozenset[groupvarname] (frozenset so its hashable, and we don't care about ordering).

        That's because weights really depends on the K-dimensions, not the underlying variables.
        Moreover, we could compute the joint marginal over multiple K-dimensions, not just one.
        """
        self.samples = samples
        self.weights = weights
        self.all_platedims = all_platedims
        self.varname2groupvarname = varname2groupvarname

    def _moments(self, moms):

        result = []
        for varnames, m in moms:
            samples = tuple(self.samples[varname] for varname in varnames)
            groupvarnames = frozenset([self.varname2groupvarname[varname] for varname in varnames])

            weights = self.weights[groupvarnames]

            result.append(m.from_marginals(samples, weights, self.all_platedims))

        return result

    moments = user_facing_moments_mixin
