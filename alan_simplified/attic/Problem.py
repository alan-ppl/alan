import torch as t
from typing import Union
from .Plate import Plate
from .BoundPlate import BoundPlate
from .SamplingType import SingleSample
from .global2local_Kdims import global2local_Kdims

from .utils import *

from typing import Callable, Any

PlateBoundPlate = Union[Plate, BoundPlate]

class Problem():
    def __init__(self, P: PlateBoundPlate, Q: PlateBoundPlate, all_platesizes: dict[str, int], data: dict[str, t.Tensor]):
        self.P = P
        self.Q = Q
        self.all_platedims = {name: Dim(name, size) for name, size in all_platesizes.items()}
        self.data = named2dim_tensordict(self.all_platedims, data)

    def sample(self, K: int, reparam:bool):
        """
        Returns: 
            globalK_sample: sample with different K-dimension for each variable.
            logPQ: log-prob.
        """
        #TODO: Error checking that P and Q make sense.
        #  data appears in P but not Q.
        #  all_platesizes is complete.

        Q_scope = {}
        sampling_type = SingleSample

        #Sample from Q
        global_Kdim = Dim('K', K)
        globalK_sample = self.Q.sample(
            scope={},
            active_platedims=[],
            all_platedims=self.all_platedims,
            sampling_type=SingleSample,
            Kdim=global_Kdim,
            reparam=reparam,
        )

        localK_sample, groupvarname2Kdim = global2local_Kdims(globalK_sample, global_Kdim)
        return localK_sample, groupvarname2Kdim

    def logPQ(self, localK_sample, groupvarname2Kdim):
        #Compute logQ
        logQ = self.Q.log_prob(
            sample=localK_sample,
            scope={},
            active_platedims=[],
            all_platedims=self.all_platedims,
            sampling_type=SingleSample,
            groupvarname2Kdim=groupvarname2Kdim,
        )

        #Subtract log K
        logQ = logQ.minus_logK()

        #Compute logP
        #Need to put data in the right place on the tree!
        logP = self.P.log_prob(
            sample={**localK_sample, **self.data},
            scope={},
            active_platedims=[],
            all_platedims=self.all_platedims,
            sampling_type=SingleSample,
            groupvarname2Kdim=groupvarname2Kdim,
        )

        #Take the difference of logP and logQ
        logPQ = logP.minus_logQ(logQ)

        return Sample(globalK_sample, logPQ)

def flatten_sample(samples:dict):
    result = {}
    for name, sample in samples.items():
        if isinstance(sample, Tensor):
            result[name] = sample
        elif isinstance(sample, GroupSample):
            for n, s in sample.items():
                result[n] = s
        else:
            assert isinstance(sample, dict)
            result = {**result, **flatten_sample(sample)}
    return result

#class Sample():
#    def __init__(self, sample:dict, logPQ:LP_Plate, all_platedims:list[str, dim]):
#        self.sample = sample
#        self.flat_sample = flatten_sample(samples)
#
#        self.logPQ = logPQ
#        self.all_platedims = all_platedims
#        self.all_platedims_set = set(all_platedims.values())
#
#    def is_platedim(dim):
#        return dim in self.all_platedims_set
#
#    def elbo(self):
#        return self.logPQ.sum()
#
#    def check_arguments(self, argnames:list[str]):
#        """
#        If there are multiple variables in a function we're computing the moments 
#        of, then we have to think a bit.  First, the 
#        plates that the variables are in have to make sense.  Its a bit like
#        scoping in the generative model.  All the variables used need to be 
#        "nested" wrt to each other.  Specifically, we can test nesting by:
#          If you have two variables with the same number of platedims, they must
#          have the same platedims.
#          If one variable has fewer platedims than another, the platedims must be
#          entirely contained in the other.
#
#        Returns a variable in the lowest-level plate, whose log-prob can validly
#        be modified.
#        """
#
#        platedimss = []
#        for argname in argnames:
#            arg = self.flat_sample[argname]
#            platedimss.append([dim for dim in generic_dims() if self.is_platedim(dim)])
#        number_of_platedims = [len(platedims) for platedims in platedimss]
#
#        #sorts from the smallest to highest number of platedims.
#        sorted_platedimss = sorted(argnames, key=number_of_platedims)
#        sorted_argnames = sorted(argnames, key=number_of_platedims)
#
#        for i in range(1, len(sorted_platedims)):
#            prev_platedims = sorted_platedims[i]
#            curr_platedims = sorted_platedims[i]
#
#            for platedim in prev_platedims:
#                if platedim not in curr_platedims:
#                    raise Exception("Asking for moments that depend on an incompatible set of variables. Specifically, we can only compute moments that depend on variables that are part of nested plates.  In other words, there needs to be somewhere in the generative model where all variables are in-scope")
#
#        return sorted_argnames[-1]
#
#        
#
#    def moments_with_argnames(self, argnamess_fs:list[tuple[list[str], Callable]]):
#        """
#        Compute importance weighted moments.
#
#        Arguments:
#            fs: list of tuples. First part of tuple is a list of argument names,
#                second part of tuple is a function.
#            
#        Returns:
#            list: List of computed moments
#
#        """
#        argnamess = [argnamess_fs[0] for argnames_f in argnamess_fs]
#        #Check that we have a valid combination of arguments.
#        attach_argname = [self.check_arguments(argnames) for argnames in argnamess]
#
#        Js = []
#        factors = []
#        for argnames, f in argnamess_fs:
#            m = f(*[self.flat_sample[argname] for argname in argnames])
#
#            platedims = [dim for dim in generic_dims(m) if self.is_platedim(dim)]
#
#            #J drops the K-dimensions, but keeps plate + unnamed dimensions.
#            sizes = [*[dim.size for dim in platedims], *m.shape]
#        
#            J_tensor   = t.zeros(sizes, dtype=m.dtype, device=m.device, requires_grad=True)
#            Js.append(J)
#            dims       = [*platedims, Ellipsis]
#            J_torchdim = J[dims]
#            factors.append(m*J_torchdim)
#
#
#
#        #Compute result with torchdim Js !!!!
#        #result = self.logPQ(extra
#        #But differentiate wrt non-torchdim Js
#        named_Es = list(t.autograd.grad(result, named_Js))
#
#        if callable(fs[0]):
#            named_Es = named_Es[0]
#
#        return named_Es
