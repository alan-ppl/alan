from .utils import *
from .moments import uniformise_moment_args, postproc_moment_outputs

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

    def moments(self, *raw_moms):
        moms = uniformise_moment_args(raw_moms)

        for varnames in moms.keys():
            assert isinstance(varnames, tuple)
            assert 1 == len(varnames)
            assert isinstance(varnames[0], str)

        result = {}
        for varnames, moment_specs in moms.items():
            samples = tuple(self.samples[varname] for varname in varnames)
            groupvarnames = frozenset([self.varname2groupvarname[varname] for varname in varnames])

            weights = self.weights[groupvarnames]

            moments = [] 
            for moment_spec in moment_specs:
                moments.append(moment_spec.from_marginals(samples, weights, self.all_platedims))
            result[varnames] = tuple(moments)

        return postproc_moment_outputs(result, raw_moms)
