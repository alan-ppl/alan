from functorch.dim import Dim

from .SamplingType import SamplingType
from .Split import Split
from .Plate import Plate, tensordict2tree
from .utils import *
from .logpq import logPQ_plate

class Sample():
    def __init__(
            self,
            problem,
            sample: dict,
            groupvarname2Kdim: dict[str, Dim],
            sampling_type: SamplingType,
            split:Split
        ):
        self.problem = problem
        self.sample = sample
        self.groupvarname2Kdim = groupvarname2Kdim
        self.sampling_type = sampling_type
        self.split = split

    @property 
    def P(self):
        return self.problem.P

    @property 
    def Q(self):
        return self.problem.Q

    @property 
    def all_platedims(self):
        return self.problem.all_platedims

    def elbo(self, extra_log_factors=None):

        if extra_log_factors is None:
            extra_log_factors = {}
        extra_log_factors = named2dim_dict(extra_log_factors, self.all_platedims)
        extra_log_factors = tensordict2tree(self.P, extra_log_factors)

        lp = logPQ_plate(
            name=None,
            P=self.P, 
            Q=self.Q, 
            sample=self.sample,
            inputs_params_P=self.P.inputs_params(self.all_platedims),
            inputs_params_Q=self.Q.inputs_params(self.all_platedims),
            data=self.problem.data,
            extra_log_factors=extra_log_factors,
            scope_P={}, 
            scope_Q={}, 
            active_platedims=[],
            all_platedims=self.all_platedims,
            groupvarname2Kdim=self.groupvarname2Kdim,
            sampling_type=self.sampling_type,
            split=self.split)

        return lp

#    def marginals(self, sample:dict, groupvarname2Kdim:dict[str, Dim]):
#        extra_log_factors = {}
#
#    def elf_marginals(sample):
#        dims = generic_dims(sample)
#
#        def marginals(samples:dict, groupvarname2Kdim:dict[str, Dim], sampling_type:SamplingType):
#            #This is an ordered dict.
#            samples = flatten_tree(samples) 
#            #Everything here is a list.
#            dimss = [generic_dims(sample) for sample in samples.values()]
#            shapess = [[dim.size for dim in dims] for dims in dimss]
#            Js_tensor = [t.zeros(shape, device=sample.device, requires_grad=True) for sample, shape in zip(samples, shapes)]
#            Js_torchdim_list = [generic_getitem(J, dims) for (J, dims) in zip(Js_tensor, dimss)]
#            #Back to a flat dict
#            Js_torchdim_dict = {name: J for (name, J) in zip(samples.keys(), Js_torchdim}
#            #And back to a tree
#            Js_torchdim_tree = tensordict2tree(self.P, Js_torchdim)
#            #Compute loss
#            L = self.elbo(
#                sample=sample,
#                groupvarname2Kdim=gropuvarname2Kdim,
#                sampling_type=sampling_type,
#                extra_log_factors=Js_torchdim_tree,
#            )
#            marginals_list = 
#
