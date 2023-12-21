from .utils import *
from .Plate import flatten_tree
from .moments import uniformise_moment_args, postproc_moment_outputs

class AbstractImportanceSample():
    def dump(self):
        """
        User-facing method that returns a flat dict of named tensors, with N as the first dimension.
        """
        return dim2named_dict(self.samples_flatdict)

    def moments(self, *raw_moms):
        moms = uniformise_moment_args(raw_moms)

        result = {}
        for varnames, moment_specs in moms.items():
            samples = tuple(self.samples_flatdict[varname] for varname in varnames)

            moments = [] 
            for moment_spec in moment_specs:
                moments.append(moment_spec.from_samples(samples, self.Ndim))
            result[varnames] = tuple(moments)

        return postproc_moment_outputs(result, raw_moms)

class ImportanceSample(AbstractImportanceSample):
    def __init__(self, problem, samples_tree, Ndim):
        """
        samples is tree-structured torchdim (as we might need to use it for extended).
        """
        self.problem = problem
        self.samples_tree = samples_tree
        self.samples_flatdict = flatten_tree(samples_tree)
        self.Ndim = Ndim

    def extend(self, extend_platesizes:dict[str, int]):
        """
        User-facing method that extends sample by drawing the rest from the prior.

        Note: doesn't take any extended data!!!

        returns ExtendedImportanceSample
        """
        #Returns ExtendedImportanceSample
        pass


class ExtendedImportanceSample:
    def __init__(self, problem, samples_tree, Ndim, extended_platedims):
        """
        samples is tree-structured torchdim (as we might need to use it for extended).
        """
        self.problem = problem
        self.samples_tree = samples_tree
        self.samples_flatdict = flatten_tree(samples_tree)
        self.Ndim = Ndim
        self.extended_platedims = extended_platedims

    def predictive_ll(self, data:dict[str, Tensor]):
        pass

