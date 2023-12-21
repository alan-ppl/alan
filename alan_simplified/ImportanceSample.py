from .utils import *
from .Plate import flatten_tree

class ImportanceSamples:
    def __init__(self, problem, samples, Ndim):
        """
        samples is tree-structured torchdim (as we might need to use it for extended).
        """
        self.problem = problem
        self.samples = samples
        self.Ndim = Ndim

    def dump(self):
        """
        User-facing method that returns a flat dict of named tensors, with N as the first dimension.
        """
        samples_flatdict = flatten_tree(self.samples)
        return samples_flatdict

