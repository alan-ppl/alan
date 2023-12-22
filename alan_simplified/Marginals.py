from .utils import *
from .moments import torchdim_moments_mixin, named_moments_mixin

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

    def _moments_uniform_input(self, moms):
        assert isinstance(moms, list)

        result = []
        for varnames, m in moms:
            samples = tuple(self.samples[varname] for varname in varnames)
            groupvarnames = frozenset([self.varname2groupvarname[varname] for varname in varnames])

            weights = self.weights[groupvarnames]

            result.append(m.from_marginals(samples, weights, self.all_platedims))

        return result

    _moments = torchdim_moments_mixin
    moments = named_moments_mixin

    def ess(self):
        result = {}
        set_all_platedims = set(self.all_platedims.values())

        for (varnames, w) in self.weights.items():
            Kdims = tuple(set(generic_dims(w)).difference(set_all_platedims))
            assert 1 <= len(Kdims)
            result[varnames] = 1/((w**2).sum(Kdims))
        return result

    def min_ess(self):
        ess_dict = self.ess()
        min_ess = [ess.min() for ess in ess_dict.values()]
        return min(min_ess)

