from .utils import *
from .moments import uniformise_moment_args

class Marginals
    def __init__(self, samples:dict[str, Tensor], weights:dict[tuple(str), Tensor], all_platedims: dict[str, Dim]):
        """
        samples and weights are flat dicts of torchdims
        samples is indexed just by the variable name as a string.
        weights could in principle be indexed by a tuple of variable names (because we could have a joint distribution).
        likewise, moms (argument to .moments) is a dict that could be indexed by a tuple of variable names.
        """
        self.samples = samples
        self.weights = weights
        self.all_platedims = all_platedims

    def moments(self, *moms):
        moms = uniformise_moment_args(moms)

        for k in moms.keys():
            assert isinstance(k, tuple)
            assert 1 == len(k)
            assert isinstance(k[0], str)

        result = {}
        for argnames, moment_specs in moms:
            samples = [self.samples[argname] for argname in argnames]
            weights = self.weights[argnames]

            moments = [] 
            for moment_spec in moment_specs:
                moments.append(moment_spec.from_marginals(samples, weights, self.all_platedims))
            result[argnames] = moments

        return result
        





